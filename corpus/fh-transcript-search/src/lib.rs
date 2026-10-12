//! FH-20: transcript search, as in `less`.
//!
//! Ctrl+F opens a search bar under the transcript (a [`TuiPanel`] that
//! takes the keys). What is typed searches as it is typed, ignoring case
//! unless the query has a capital letter; `re:` makes the rest a regular
//! expression. It searches the transcript as the view drew it
//! ([`DrawnTranscript`]): every row, wrapped, as its renderer drew it,
//! the notices where they are drawn and the reply streaming in. What a
//! collapsed item cuts, and the transcripts of agents not shown, are not
//! drawn; there it searches their text as data: what the user sent, the
//! replies, reasoning, a call's arguments and whole output, plugins'
//! entries and the notices, in the order the view draws them.
//!
//! The matches are lit in the transcript ([`TranscriptView::highlights`],
//! matching case as the search does). Moving to one scrolls the view to
//! its row ([`ScrollTo`] with [`ItemRow::Top`]); one not drawn expands
//! its item, and the frame that draws it gives its row. Enter takes it;
//! `n` (or Enter again) and `N` (or Shift+Enter) move, wrapping around,
//! and the bar counts (`3/17`). Tab
//! searches every agent, naming the agent of the match and showing it. Esc
//! closes the bar and unlights the matches; unless a match was taken, it
//! also collapses what the search expanded and scrolls back to where the
//! view was. The search only reads: it never changes a conversation.

use std::collections::{BTreeSet, HashMap, HashSet};

use regex::{Regex, RegexBuilder};
use rig_harness::prelude::*;
use rig_harness::rig_core::message::ToolResult as CallResult;
use rig_tui::crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::{Style, Stylize};
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::{
    AppKeyBindingsExt, DrawnTranscript, Highlight, ItemRow, KeyBinding, KeyFocus, KeyPress,
    KeyPressed, PanelCanvas, Placement, RequestRedraw, ScrollTo, ShownNotice, TerminalEvent,
    TranscriptItem, TranscriptSlot, TranscriptView, TuiPanel, TuiPlugin, TuiSystems,
};

/// Whose transcript is searched.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Scope {
    /// The agent shown.
    #[default]
    Shown,
    /// Every agent.
    All,
}

/// A match.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    /// The agent whose transcript it is in.
    pub agent: Entity,
    /// The innermost item it is drawn in (a call rather than its message),
    /// or `None` for rows of no item, such as the reply streaming in.
    pub item: Option<TranscriptItem>,
    /// Which of its item's matches it is, top to bottom.
    pub nth: usize,
    /// Its row in its item as last drawn, or `None` where it is not drawn:
    /// in what a collapsed item cuts, or in an agent not shown.
    pub row: Option<usize>,
    /// The text it matched.
    pub text: String,
}

/// The query as a regular expression, and whether it matches case, or
/// `None` for an empty one: `re:` starts a regular expression, anything
/// else is matched as typed; case is ignored unless the query has a
/// capital letter.
pub fn compile(query: &str) -> Result<Option<(Regex, bool)>, regex::Error> {
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
        .map(|regex| Some((regex, capital)))
}

/// The non-empty matches of `regex` in `text`, line by line.
fn matches<'t>(regex: &'t Regex, text: &'t str) -> impl Iterator<Item = &'t str> + 't {
    let found = text.lines().flat_map(|line| regex.find_iter(line));
    found
        .filter(|found| !found.is_empty())
        .map(|found| found.as_str())
}

/// The matches of `regex` in the transcript of `agent`, top to bottom.
/// Where `drawn` is its transcript, they are the matches in the rows the
/// view drew, then, after the rows of each item not `expanded`, the
/// matches of its `texts` past the ones it drew: those it cuts. Else they
/// are the matches in its `texts` (each with its item, in the order they
/// are drawn).
pub fn find(
    regex: &Regex,
    agent: Entity,
    texts: &[(Option<TranscriptItem>, String)],
    drawn: Option<&DrawnTranscript>,
    expanded: &HashSet<TranscriptItem>,
) -> Vec<Hit> {
    let drawn = drawn.filter(|drawn| drawn.agent() == Some(agent));
    let hit = |item: Option<TranscriptItem>, row, text: &str| Hit {
        agent,
        item,
        nth: 0,
        row,
        text: text.to_owned(),
    };
    // Each match with its place in drawing order.
    let mut placed = Vec::new();
    // How many of its texts' matches each item drew.
    let mut drawn_in: HashMap<Option<TranscriptItem>, usize> = HashMap::new();
    for (at, row) in drawn
        .iter()
        .flat_map(|drawn| drawn.rows().iter().enumerate())
    {
        let item = drawn.and_then(|drawn| drawn.item_at(at)).cloned();
        let rows = item
            .as_ref()
            .zip(drawn)
            .and_then(|(item, drawn)| drawn.rows_of(item));
        let start = rows.map_or(at, |rows| rows.start);
        for text in matches(regex, &row.text) {
            *drawn_in.entry(item.clone()).or_default() += 1;
            placed.push((at, hit(item.clone(), at.checked_sub(start), text)));
        }
    }
    for (at, (item, text)) in texts.iter().enumerate() {
        let whole = item.as_ref().is_none_or(|item| expanded.contains(item));
        let order = match drawn {
            Some(_) if whole => continue,
            Some(drawn) => item
                .as_ref()
                .and_then(|item| drawn.rows_of(item))
                .map_or(usize::MAX, |rows| rows.end.saturating_sub(1)),
            None => at,
        };
        for text in matches(regex, text) {
            match drawn_in.get_mut(item).filter(|left| **left > 0) {
                Some(left) => *left -= 1,
                None => placed.push((order, hit(item.clone(), None, text))),
            }
        }
    }
    placed.sort_by_key(|(order, _)| *order);
    let mut counted: HashMap<Option<TranscriptItem>, usize> = HashMap::new();
    let mut hits: Vec<Hit> = placed.into_iter().map(|(_, hit)| hit).collect();
    for hit in &mut hits {
        let count = counted.entry(hit.item.clone()).or_default();
        hit.nth = *count;
        *count += 1;
    }
    hits
}

/// A tool result's text.
fn result_text(result: &CallResult) -> String {
    let lines: Vec<&str> = result
        .content
        .iter()
        .filter_map(|content| content.as_text())
        .collect();
    lines.join("\n")
}

/// The texts of the transcript of the agent `id` with `conversation`, its
/// `entries` and the view's `notices`, each with its item, in the order
/// the view draws them ([`TranscriptSlot`]): a call with its arguments and
/// result, inside the reply it is in.
fn texts_of(
    id: &AgentId,
    conversation: &Conversation,
    entries: &[(Entity, &TranscriptEntry)],
    notices: &[ShownNotice],
) -> Vec<(Option<TranscriptItem>, String)> {
    // In a slot, entries go before notices, but inside a message notices
    // go first; both go before the message's item there.
    let inside = |slot| matches!(slot, TranscriptSlot::Inside(..));
    let mut texts = Vec::new();
    for (entity, entry) in entries {
        let slot = TranscriptSlot::of_entry(entry, conversation);
        let text = format!("{}\n{}", entry.title, entry.text);
        texts.push((
            (slot, u8::from(inside(slot))),
            TranscriptItem::Entry(*entity),
            text,
        ));
    }
    for (at, notice) in notices.iter().enumerate() {
        if let Some(slot) = notice.slot(id, conversation) {
            let order = (slot, u8::from(!inside(slot)));
            texts.push((order, TranscriptItem::Notice(at), notice.text.clone()));
        }
    }
    let messages = conversation.messages();
    for (at, entry) in messages.iter().enumerate() {
        let Some(message) = conversation.id(at).map(TranscriptItem::Message) else {
            continue;
        };
        let order = |part| (TranscriptSlot::Inside(at, part), 2);
        match entry {
            message::Message::System { .. } => {}
            message::Message::User { content } => {
                // A result of the call before it is drawn with that call.
                let previous = at.checked_sub(1).and_then(|before| messages.get(before));
                let answers = |result: &CallResult| {
                    match previous {
                    Some(message::Message::Assistant(reply)) => reply.content.iter().any(
                        |part| matches!(part, AssistantContent::ToolCall(call) if call.id == result.call),
                    ),
                    _ => false,
                }
                };
                for (part, content) in content.iter().enumerate() {
                    let text = match content {
                        UserContent::Text(text) => text.text.clone(),
                        UserContent::ToolResult(result) if !answers(result) => result_text(result),
                        _ => continue,
                    };
                    texts.push((order(part), message.clone(), text));
                }
            }
            message::Message::Assistant(reply) => {
                let next = at.checked_add(1).and_then(|after| messages.get(after));
                let results = match next {
                    Some(message::Message::User { content }) => content.iter().collect(),
                    _ => Vec::new(),
                };
                for (part, content) in reply.content.iter().enumerate() {
                    let (item, text) = match content {
                        AssistantContent::Text(text) => (message.clone(), text.text.clone()),
                        AssistantContent::Reasoning(reasoning) => {
                            (message.clone(), reasoning.text.clone())
                        }
                        AssistantContent::ToolCall(call) => {
                            let arguments =
                                serde_json::Value::Object(call.function.arguments.clone());
                            let result = results.iter().find_map(|content| match content {
                                UserContent::ToolResult(result) if result.call == call.id => {
                                    Some(result_text(result))
                                }
                                _ => None,
                            });
                            let text = format!(
                                "{} {arguments}\n{}",
                                call.function.name,
                                result.unwrap_or_default()
                            );
                            (TranscriptItem::Call(call.id.clone()), text)
                        }
                        _ => continue,
                    };
                    texts.push((order(part), item, text));
                }
            }
        }
    }
    texts.sort_by_key(|(order, ..)| *order);
    let texts = texts.into_iter();
    texts.map(|(_, item, text)| (Some(item), text)).collect()
}

/// The open search bar, on its [`TuiPanel`]'s entity.
#[derive(Component, Debug)]
pub struct TranscriptSearch {
    /// What is typed.
    pub query: String,
    /// Whose transcript is searched.
    pub scope: Scope,
    regex: Option<Regex>,
    match_case: bool,
    error: bool,
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
    /// The match the view was last scrolled to, and its row then.
    went: Option<(MatchKey, Option<usize>)>,
    /// The item the search expanded, which was collapsed before.
    expanded: Option<TranscriptItem>,
    /// The highlights it lit.
    lit: Vec<Highlight>,
}

/// What tells a match apart across searches.
type MatchKey = (Entity, Option<TranscriptItem>, usize);

impl TranscriptSearch {
    fn new(back: Option<(Entity, Option<(TranscriptItem, usize)>)>) -> Self {
        Self {
            query: String::new(),
            scope: Scope::Shown,
            regex: None,
            match_case: false,
            error: false,
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

    /// The match shown.
    pub fn current(&self) -> Option<&Hit> {
        self.hits.get(self.current?)
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
        Some((hit.agent, hit.item.clone(), hit.nth))
    }

    /// Searches `transcripts` again with the query, keeping the match
    /// shown where it still is; else, while typing, the latest match.
    fn search(&mut self, transcripts: &Transcripts, view: &TranscriptView) {
        let was = self.key(self.current);
        if self.dirty {
            self.dirty = false;
            let compiled = compile(&self.query);
            self.error = compiled.is_err();
            let compiled = compiled.ok().flatten();
            self.match_case = compiled.as_ref().is_some_and(|(_, case)| *case);
            self.regex = compiled.map(|(regex, _)| regex);
        }
        self.hits = self
            .regex
            .as_ref()
            .map(|regex| transcripts.find(regex, self.scope, view))
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
        let texts: BTreeSet<String> = self
            .hits
            .iter()
            .map(|hit| match self.match_case {
                true => hit.text.clone(),
                false => hit.text.to_lowercase(),
            })
            .collect();
        let style = Style::new().black().on_yellow();
        texts
            .into_iter()
            .map(|text| Highlight {
                text,
                style,
                match_case: self.match_case,
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
        app.init_resource::<DrawnTranscript>()
            .add_systems(Update, type_in_bar)
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
    drawn: Res<'w, DrawnTranscript>,
}

impl Transcripts<'_, '_> {
    /// Whether a conversation, a reply streaming in, an entry, the agent
    /// shown or what the view drew changed since the last frame; the
    /// notices are the view's.
    fn changed(&self) -> bool {
        !self.changed.is_empty() || self.shown.is_changed() || self.drawn.is_changed()
    }

    /// The matches of `regex` in the transcript of the agent shown, or of
    /// every agent: as drawn where the view drew it, else in its texts,
    /// with the reply streaming in last.
    fn find(&self, regex: &Regex, scope: Scope, view: &TranscriptView) -> Vec<Hit> {
        let mut agents: Vec<_> = self
            .agents
            .iter()
            .filter(|(agent, ..)| scope == Scope::All || Some(*agent) == self.shown.0)
            .collect();
        agents.sort_by_key(|(agent, ..)| *agent);
        let mut hits = Vec::new();
        for (agent, id, conversation, turn, entries) in agents {
            let entries: Vec<_> = entries
                .into_iter()
                .flat_map(|entries| entries.iter())
                .filter_map(|entity| Some((entity, self.entries.get(entity).ok()?)))
                .collect();
            let mut texts = texts_of(id, conversation, &entries, view.notices());
            let streaming = turn
                .and_then(|turn| self.calls.get(turn.turn()).ok())
                .and_then(|calls| calls.iter().find_map(|call| self.partials.get(call).ok()));
            let streaming = streaming
                .iter()
                .flat_map(|partial| [&partial.reasoning, &partial.text]);
            texts.extend(streaming.map(|text| (None, text.clone())));
            let drawn = Some(&*self.drawn);
            hits.extend(find(regex, agent, &texts, drawn, &view.expanded));
        }
        hits
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
            search.search(&transcripts, &view);
        }
    }
    if closing {
        close(bar, &mut search, &mut view, &mut commands);
        return;
    }
    if !search.dirty && (transcripts.changed() || view.is_changed()) {
        search.search(&transcripts, &view);
        typed = true;
    }
    if typed {
        show(&mut search, &mut view, &mut commands);
        redraw.write(RequestRedraw);
    }
}

/// Lights the matches in the transcript, ahead of other plugins'
/// highlights, and, when the match shown moved, scrolls to it: to its row
/// where it is drawn, else to its item, expanded in place of the one
/// expanded before (the next frame draws the match, and the search, which
/// runs again on what is drawn, scrolls to its row).
fn show(search: &mut TranscriptSearch, view: &mut ResMut<TranscriptView>, commands: &mut Commands) {
    let lit = search.highlights();
    if lit != search.lit {
        view.highlights
            .retain(|highlight| !search.lit.contains(highlight));
        view.highlights.splice(0..0, lit.iter().cloned());
        search.lit = lit;
    }
    let went = search
        .key(search.current)
        .zip(Some(search.current().and_then(|hit| hit.row)));
    if went == search.went {
        return;
    }
    search.went = went;
    let Some(hit) = search.current().cloned() else {
        return;
    };
    if search.expanded != hit.item
        && let Some(item) = search.expanded.take()
    {
        view.expanded.remove(&item);
    }
    let row = match (&hit.item, hit.row) {
        (Some(_), Some(row)) => ItemRow::Top(row),
        (Some(item), None) => {
            if view.expanded.insert(item.clone()) {
                search.expanded = Some(item.clone());
            }
            ItemRow::Reveal
        }
        (None, _) => ItemRow::Reveal,
    };
    let (agent, item) = (hit.agent, hit.item);
    commands.trigger(ScrollTo { agent, item, row });
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
        let (item, row) = top.map_or((None, 0), |(item, row)| (Some(item), row));
        let row = ItemRow::Top(row);
        commands.trigger(ScrollTo { agent, item, row });
    }
}

/// Draws the bar: the query, the count, and where the match is.
fn draw(mut bars: Query<(&TranscriptSearch, &mut PanelCanvas)>, names: Query<&Name>) {
    for (search, mut canvas) in &mut bars {
        let mut spans = vec![
            Span::from(format!("/{}", search.query)).bold(),
            Span::from(format!("  {}", search.counter())).yellow(),
        ];
        if search.scope == Scope::All {
            let agent = search.current().and_then(|hit| names.get(hit.agent).ok());
            let agent = agent.map(Name::as_str);
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
