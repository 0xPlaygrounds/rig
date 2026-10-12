//! FH-20: transcript search, as in `less`.
//!
//! Ctrl+F opens a search bar under the transcript (a [`TuiPanel`] that
//! takes the keys). What is typed searches as it is typed, ignoring case
//! unless the query has a capital letter; `re:` makes the rest a regular
//! expression. The search covers what the transcript shows, read as data:
//! what the user sent, the replies, reasoning, each tool call's arguments
//! and its whole output, plugins' transcript entries, the notices the view
//! shows, and the reply streaming in, which is searched again as it grows.
//!
//! The matches are lit in the transcript ([`TranscriptView::highlights`])
//! and the view scrolls to the match shown ([`ScrollTo`]), expanding the
//! item it is in. Enter takes it; `n` (or Enter again) and `N` (or
//! Shift+Enter) move, wrapping around, and the bar counts (`3/17`). Tab
//! searches every agent, naming the agent of the match and showing it. Esc
//! closes the bar and unlights the matches; unless a match was taken, it
//! also collapses what the search expanded and scrolls back to where the
//! view was. The search only reads: it never changes a conversation.

use std::collections::BTreeSet;
use std::ops::Range;

use regex::{Regex, RegexBuilder};
use rig_harness::prelude::*;
use rig_tui::crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::{Style, Stylize};
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::{
    AppKeyBindingsExt, Highlight, ItemRow, KeyBinding, KeyFocus, KeyPress, KeyPressed, PanelCanvas,
    Placement, RequestRedraw, ScrollTo, TerminalEvent, TranscriptItem, TranscriptView, TuiPanel,
    TuiPlugin, TuiSystems,
};

/// Rows the view keeps above a match it scrolls to.
const MARGIN: usize = 3;

/// Whose transcript is searched.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Scope {
    /// The agent shown.
    #[default]
    Shown,
    /// Every agent.
    All,
}

/// Where a block is drawn, as far as the view can be scrolled to it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Place {
    /// In this item: a message, a tool call with its output, or an entry.
    Item(TranscriptItem),
    /// The reply streaming in, at the bottom.
    Streaming,
    /// A notice, which the view has no item for.
    Notice,
}

/// One searchable text of a transcript.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    /// The agent whose transcript it is in.
    pub agent: Entity,
    /// That agent's name.
    pub agent_name: String,
    /// Where it is drawn.
    pub place: Place,
    /// Its text.
    pub text: String,
}

/// A match: in which block, on which of its lines, and where on it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    /// The block, by its index in the search's blocks.
    pub block: usize,
    /// The line of the block.
    pub line: usize,
    /// The bytes of the line matched.
    pub range: Range<usize>,
}

/// The query as a regular expression, or `None` for an empty one: `re:`
/// starts a regular expression, anything else is matched as typed; case
/// is ignored unless the query has a capital letter.
pub fn compile(query: &str) -> Result<Option<Regex>, regex::Error> {
    let (pattern, literal) = match query.strip_prefix("re:") {
        Some(pattern) => (pattern.to_owned(), false),
        None => (regex::escape(query), true),
    };
    if pattern.is_empty() {
        return Ok(None);
    }
    let typed = if literal { query } else { &pattern };
    // A capital after `\` is a class such as `\W`, not a capital letter.
    let mut escaped = false;
    let capital = typed.chars().any(|letter| {
        let capital = !escaped && letter.is_uppercase();
        escaped = !escaped && letter == '\\';
        capital
    });
    RegexBuilder::new(&pattern)
        .case_insensitive(!capital)
        .build()
        .map(Some)
}

/// Every non-empty match of `regex` in `blocks`, in order, line by line.
pub fn find(regex: &Regex, blocks: &[Block]) -> Vec<Hit> {
    let mut hits = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        for (line, text) in block.text.lines().enumerate() {
            let found = regex.find_iter(text).filter(|found| !found.is_empty());
            hits.extend(found.map(|found| Hit {
                block: index,
                line,
                range: found.range(),
            }));
        }
    }
    hits
}

/// The texts of `conversation` in the order the transcript draws them,
/// each with its item, and the entries (title, text) after the message
/// they follow.
fn texts_of<'a>(
    conversation: &'a Conversation,
    entries: &[(Entity, &'a TranscriptEntry)],
) -> Vec<(TranscriptItem, String)> {
    let last = conversation.messages().len().checked_sub(1);
    // Where the view draws an entry: after its message, or after the one
    // standing for it once an edit took it out.
    let follows = |entry: &TranscriptEntry| {
        let after = entry.after?;
        let standing = last.map(|last| conversation.start_of(after).min(last));
        conversation.index_of(after).or(standing)
    };
    let entries_after = |at: Option<usize>, texts: &mut Vec<_>| {
        for (entity, entry) in entries.iter().filter(|(_, entry)| follows(entry) == at) {
            let text = format!("{}\n{}", entry.title, entry.text);
            texts.push((TranscriptItem::Entry(*entity), text));
        }
    };
    let mut texts = Vec::new();
    entries_after(None, &mut texts);
    for (at, entry) in conversation.messages().iter().enumerate() {
        let id = conversation.id(at);
        let message = |text: &str| id.map(|id| (TranscriptItem::Message(id), text.to_owned()));
        match entry {
            message::Message::System { .. } => {}
            message::Message::User { content } => {
                texts.extend(content.iter().filter_map(|part| match part {
                    UserContent::Text(text) => message(&text.text),
                    UserContent::ToolResult(result) => {
                        let lines: Vec<&str> = result
                            .content
                            .iter()
                            .filter_map(|content| content.as_text())
                            .collect();
                        Some((TranscriptItem::Call(result.call.clone()), lines.join("\n")))
                    }
                    _ => None,
                }));
            }
            message::Message::Assistant(reply) => {
                texts.extend(reply.content.iter().filter_map(|part| match part {
                    AssistantContent::Text(text) => message(&text.text),
                    AssistantContent::Reasoning(reasoning) => message(&reasoning.text),
                    AssistantContent::ToolCall(call) => {
                        let arguments = serde_json::Value::Object(call.function.arguments.clone());
                        let text = format!("{} {arguments}", call.function.name);
                        Some((TranscriptItem::Call(call.id.clone()), text))
                    }
                    _ => None,
                }));
            }
        }
        entries_after(Some(at), &mut texts);
    }
    texts
}

/// The open search bar, on its [`TuiPanel`]'s entity.
#[derive(Component, Debug)]
pub struct TranscriptSearch {
    /// What is typed.
    pub query: String,
    /// Whose transcript is searched.
    pub scope: Scope,
    regex: Option<Regex>,
    error: bool,
    blocks: Vec<Block>,
    hits: Vec<Hit>,
    current: Option<usize>,
    /// Typing edits the query; after Enter, `n` and `N` move.
    editing: bool,
    /// Whether a match was taken with Enter (or `n`, `N`).
    taken: bool,
    /// The query or scope changed since the last search.
    dirty: bool,
    /// The agent shown and where the view was scrolled when it opened.
    back: Option<(Entity, Option<(TranscriptItem, usize)>)>,
    /// The match the view was last scrolled to.
    went: Option<MatchKey>,
    /// The item the search expanded, which was collapsed before.
    expanded: Option<TranscriptItem>,
    /// The highlights it lit.
    lit: Vec<Highlight>,
}

/// What tells a match apart across searches.
type MatchKey = (Entity, Place, usize, usize);

impl TranscriptSearch {
    fn new(back: Option<(Entity, Option<(TranscriptItem, usize)>)>) -> Self {
        Self {
            query: String::new(),
            scope: Scope::Shown,
            regex: None,
            error: false,
            blocks: Vec::new(),
            hits: Vec::new(),
            current: None,
            editing: true,
            taken: false,
            dirty: true,
            back,
            went: None,
            expanded: None,
            lit: Vec::new(),
        }
    }

    /// Every match, in transcript order.
    pub fn hits(&self) -> &[Hit] {
        &self.hits
    }

    /// The match shown, with its block.
    pub fn current(&self) -> Option<(&Block, &Hit)> {
        let hit = self.hits.get(self.current?)?;
        Some((self.blocks.get(hit.block)?, hit))
    }

    /// The count the bar shows, such as `3/17`, or `bad regex`.
    pub fn counter(&self) -> String {
        if self.error {
            return "bad regex".to_owned();
        }
        let at = self.current.map_or(0, |current| current + 1);
        format!("{at}/{}", self.hits.len())
    }

    /// What tells match `at` apart.
    fn key(&self, at: Option<usize>) -> Option<MatchKey> {
        let hit = self.hits.get(at?)?;
        let block = self.blocks.get(hit.block)?;
        Some((block.agent, block.place.clone(), hit.line, hit.range.start))
    }

    /// Searches `blocks` again with the query, keeping the match shown
    /// where it still is; else, while typing, the latest match.
    fn search(&mut self, blocks: Vec<Block>) {
        let was = self.key(self.current);
        if self.dirty {
            self.dirty = false;
            let compiled = compile(&self.query);
            self.error = compiled.is_err();
            self.regex = compiled.ok().flatten();
        }
        self.blocks = blocks;
        self.hits = self
            .regex
            .as_ref()
            .map(|regex| find(regex, &self.blocks))
            .unwrap_or_default();
        let kept = (0..self.hits.len()).find(|&at| was.is_some() && self.key(Some(at)) == was);
        let last = self.hits.len().checked_sub(1);
        self.current = match kept {
            Some(kept) if !self.editing => Some(kept),
            _ if self.editing => last,
            _ => self.current.zip(last).map(|(at, last)| at.min(last)),
        };
    }

    /// Takes the match `by` matches on, wrapping around.
    fn step(&mut self, by: isize) {
        let count = self.hits.len();
        if count == 0 {
            return;
        }
        self.taken = true;
        self.editing = false;
        let count = isize::try_from(count).unwrap_or(isize::MAX);
        let current = self.current.and_then(|at| isize::try_from(at).ok());
        let next = match current {
            Some(current) => (current + by).rem_euclid(count),
            None if by >= 0 => 0,
            None => count - 1,
        };
        self.current = usize::try_from(next).ok();
    }

    /// Each text the matches are, once, lit in the transcript.
    fn highlights(&self) -> Vec<Highlight> {
        let lines: Vec<Vec<&str>> = self
            .blocks
            .iter()
            .map(|block| block.text.lines().collect())
            .collect();
        let texts: BTreeSet<String> = self
            .hits
            .iter()
            .filter_map(|hit| lines.get(hit.block)?.get(hit.line)?.get(hit.range.clone()))
            .map(str::to_lowercase)
            .collect();
        let style = Style::new().black().on_yellow();
        texts
            .into_iter()
            .map(|text| Highlight {
                text,
                style,
                ..Highlight::default()
            })
            .collect()
    }
}

/// Transcript search in the terminal view, on Ctrl+F.
#[derive(Default)]
pub struct TranscriptSearchPlugin;

impl Plugin for TranscriptSearchPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<TuiPlugin>() {
            app.add_plugins(TuiPlugin);
        }
        // Shadows the input's Ctrl+F (a character forward, as Right does).
        app.add_key_binding(
            KeyBinding::new(KeyPress::ctrl('f'), "Search the transcript"),
            open,
        );
        app.add_systems(Update, type_in_bar)
            .add_systems(PostUpdate, draw.in_set(TuiSystems::Draw));
    }
}

/// Ctrl+F: opens the bar with the keys, unless it is open.
fn open(
    pressed: On<KeyPressed>,
    bars: Query<(), With<TranscriptSearch>>,
    shown: Res<Shown>,
    view: Res<TranscriptView>,
    mut focus: ResMut<KeyFocus>,
    mut commands: Commands,
) {
    if pressed.released || !bars.is_empty() {
        return;
    }
    let back = shown.0.map(|agent| (agent, view.top().cloned()));
    let bar = (
        TranscriptSearch::new(back),
        TuiPanel::new(Placement::Bottom(Constraint::Length(1))),
    );
    focus.0 = Some(commands.spawn(bar).id());
}

/// What the bar searches, read live.
#[derive(bevy_ecs::system::SystemParam)]
struct Transcripts<'w, 's> {
    agents: Query<
        'w,
        's,
        (
            Entity,
            &'static Name,
            &'static AgentId,
            &'static Conversation,
            Option<&'static ActiveTurn>,
            Option<&'static TranscriptEntries>,
        ),
        With<Agent>,
    >,
    entries: Query<'w, 's, &'static TranscriptEntry>,
    calls: Query<'w, 's, &'static Calls>,
    partials: Query<'w, 's, &'static Partial>,
    changed: Query<
        'w,
        's,
        (),
        Or<(
            Changed<Conversation>,
            Changed<Partial>,
            Changed<TranscriptEntry>,
        )>,
    >,
    shown: Res<'w, Shown>,
}

impl Transcripts<'_, '_> {
    /// Whether a conversation, a reply streaming in, an entry or the agent
    /// shown changed since the last frame; the notices are the view's.
    fn changed(&self) -> bool {
        !self.changed.is_empty() || self.shown.is_changed()
    }

    /// The blocks of the agent shown, or of every agent: its texts, the
    /// notices about it (those about every agent in the agent shown's),
    /// then the reply streaming in.
    fn blocks(&self, scope: Scope, view: &TranscriptView) -> Vec<Block> {
        let mut agents: Vec<_> = self
            .agents
            .iter()
            .filter(|(agent, ..)| scope == Scope::All || Some(*agent) == self.shown.0)
            .collect();
        agents.sort_by_key(|(agent, ..)| *agent);
        let mut blocks = Vec::new();
        for (agent, name, id, conversation, turn, entries) in agents {
            let block = |place, text: &str| Block {
                agent,
                agent_name: name.as_str().to_owned(),
                place,
                text: text.to_owned(),
            };
            let entries: Vec<_> = entries
                .into_iter()
                .flat_map(|entries| entries.iter())
                .filter_map(|entity| Some((entity, self.entries.get(entity).ok()?)))
                .collect();
            for (item, text) in texts_of(conversation, &entries) {
                blocks.push(block(Place::Item(item), &text));
            }
            let about = |notice: &&rig_tui::ShownNotice| match &notice.agent {
                Some(about) => about == id,
                None => self.shown.0 == Some(agent),
            };
            for notice in view.notices().iter().filter(about) {
                blocks.push(block(Place::Notice, &notice.text));
            }
            let streaming = turn
                .and_then(|turn| self.calls.get(turn.turn()).ok())
                .and_then(|calls| calls.iter().find_map(|call| self.partials.get(call).ok()));
            for text in streaming
                .iter()
                .flat_map(|partial| [&partial.reasoning, &partial.text])
            {
                if !text.is_empty() {
                    blocks.push(block(Place::Streaming, text));
                }
            }
        }
        blocks
    }
}

/// Types into the open bar, in order, then searches again when the query
/// or what it searches changed, and shows the match in the transcript. It
/// closes when it loses the keys, as on an Esc. The keys of the frame it
/// opened in were the input's.
fn type_in_bar(
    mut events: MessageReader<TerminalEvent>,
    mut bars: Query<(Entity, &mut TranscriptSearch, Ref<TuiPanel>)>,
    focus: Res<KeyFocus>,
    transcripts: Transcripts,
    mut view: ResMut<TranscriptView>,
    mut redraw: MessageWriter<RequestRedraw>,
    mut commands: Commands,
) {
    let Ok((bar, mut search, opened)) = bars.single_mut() else {
        events.clear();
        return;
    };
    if opened.is_added() {
        events.clear();
    }
    if focus.0 != Some(bar) && !opened.is_added() {
        close(bar, &mut search, &mut view, &mut commands);
        return;
    }
    let (mut closing, mut typed) = (false, false);
    for TerminalEvent(event) in events.read() {
        match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => {
                let shift = key.modifiers.contains(KeyModifiers::SHIFT);
                let plain = !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT);
                match key.code {
                    KeyCode::Esc => closing = true,
                    KeyCode::Enter if shift => search.step(-1),
                    KeyCode::Enter if search.editing => {
                        search.editing = false;
                        search.taken = search.current.is_some();
                    }
                    KeyCode::Enter => search.step(1),
                    KeyCode::Tab => {
                        search.scope = match search.scope {
                            Scope::Shown => Scope::All,
                            Scope::All => Scope::Shown,
                        };
                        search.dirty = true;
                    }
                    KeyCode::Backspace => {
                        search.editing = true;
                        search.query.pop();
                        search.dirty = true;
                    }
                    KeyCode::Char('n') if plain && !search.editing => search.step(1),
                    KeyCode::Char('N') if plain && !search.editing => search.step(-1),
                    KeyCode::Char(typed) if plain => {
                        search.editing = true;
                        search.query.push(typed);
                        search.dirty = true;
                    }
                    _ => continue,
                }
            }
            Event::Paste(text) => {
                search.editing = true;
                let line = text.lines().next().unwrap_or_default();
                search.query.push_str(line);
                search.dirty = true;
            }
            _ => continue,
        }
        if closing {
            break;
        }
        typed = true;
        if search.dirty {
            let blocks = transcripts.blocks(search.scope, &view);
            search.search(blocks);
        }
    }
    if closing {
        close(bar, &mut search, &mut view, &mut commands);
        return;
    }
    if !search.dirty && (transcripts.changed() || view.is_changed()) {
        let blocks = transcripts.blocks(search.scope, &view);
        search.search(blocks);
        typed = true;
    }
    if typed {
        show(&mut search, &mut view, &mut commands);
        redraw.write(RequestRedraw);
    }
}

/// Lights the matches in the transcript and, when the match shown moved,
/// scrolls there, expanding its item in place of the one expanded before.
fn show(search: &mut TranscriptSearch, view: &mut ResMut<TranscriptView>, commands: &mut Commands) {
    let lit = search.highlights();
    if lit != search.lit {
        view.highlights
            .retain(|highlight| !search.lit.contains(highlight));
        view.highlights.extend(lit.iter().cloned());
        search.lit = lit;
    }
    let key = search.key(search.current);
    if key == search.went {
        return;
    }
    search.went = key;
    let Some((block, hit)) = search.current() else {
        return;
    };
    let (agent, place, line) = (block.agent, block.place.clone(), hit.line);
    if let Some(item) = search.expanded.take() {
        view.expanded.remove(&item);
    }
    let item = match place {
        Place::Item(item) => {
            if view.expanded.insert(item.clone()) {
                search.expanded = Some(item.clone());
            }
            Some(item)
        }
        Place::Streaming => None,
        // The view has no item for a notice: it stays where it is.
        Place::Notice => return,
    };
    let line = line.saturating_sub(MARGIN);
    commands.trigger(ScrollTo {
        agent,
        item,
        row: ItemRow::Top(line),
    });
}

/// Closes the bar and unlights the matches; with no match taken, also
/// collapses what it expanded and scrolls back to where the view was.
fn close(
    bar: Entity,
    search: &mut TranscriptSearch,
    view: &mut ResMut<TranscriptView>,
    commands: &mut Commands,
) {
    commands.entity(bar).despawn();
    if !search.lit.is_empty() {
        view.highlights
            .retain(|highlight| !search.lit.contains(highlight));
    }
    if search.taken {
        return;
    }
    if let Some(item) = search.expanded.take() {
        view.expanded.remove(&item);
    }
    if let Some((agent, top)) = search.back.take() {
        let (item, line) = top.map_or((None, 0), |(item, line)| (Some(item), line));
        commands.trigger(ScrollTo {
            agent,
            item,
            row: ItemRow::Top(line),
        });
    }
}

/// Draws the bar: the query, the count, and where the match is.
fn draw(mut bars: Query<(&TranscriptSearch, &mut PanelCanvas)>) {
    for (search, mut canvas) in &mut bars {
        let mut spans = vec![
            Span::from(format!("/{}", search.query)).bold(),
            Span::from(format!("  {}", search.counter())).yellow(),
        ];
        if search.scope == Scope::All {
            let agent = search.current().map(|(block, _)| block.agent_name.as_str());
            spans.push(Span::from(format!(
                "  all agents · {}",
                agent.unwrap_or("-")
            )));
        }
        let keys = if search.editing {
            "  Enter take · Tab scope · Esc close"
        } else {
            "  n next · N previous · Esc close"
        };
        spans.push(Span::from(keys).dim());
        canvas.render(Line::from(spans));
    }
}

#[cfg(test)]
mod tests;
