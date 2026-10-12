//! FH-20: transcript search, as in `less`.
//!
//! Ctrl+F opens a search bar under the transcript (a [`TuiPanel`] that
//! takes the keys). What is typed searches as it is typed, ignoring case
//! unless the query has a capital letter; `re:` makes the rest a regular
//! expression. The search covers what the transcript shows, read as data
//! from the agents: what the user sent, the replies, reasoning, each tool
//! call's arguments and its whole output (also where the transcript cuts
//! it short), the notices of this build, and the reply streaming in, which
//! is searched again as it grows.
//!
//! Enter takes the match shown and goes on to the next, as `n` then does;
//! Shift+Enter and `N` go back; both wrap around, and the bar counts
//! (`3/17`). Tab searches every agent instead of the one shown, naming the
//! agent of each match and showing that agent when a match of it is taken.
//! The bar shows the block of the match around it, the matches lit. Esc
//! closes it and, unless a match was taken, shows the agent shown before.
//!
//! The search only reads: it never changes a conversation or the session.

use std::ops::Range;

use regex::{Regex, RegexBuilder};
use rig_harness::prelude::*;
use rig_tui::crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::{Style, Stylize};
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::ratatui::widgets::{Block as Border, Paragraph};
use rig_tui::{
    AppKeyBindingsExt, Focus, KeyBinding, KeyFocus, KeyPress, KeyPressed, PanelCanvas, Placement,
    RequestRedraw, TerminalEvent, TuiPanel, TuiPlugin, TuiSystems,
};

/// Lines of the matched block the bar shows around the match.
const CONTEXT_LINES: u16 = 7;

/// Notices kept for searching, the oldest dropped first.
const KEPT_NOTICES: usize = 500;

/// Columns kept before a match far right on its line.
const LEAD: usize = 12;

/// Whose transcript is searched.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Scope {
    /// The agent shown.
    #[default]
    Shown,
    /// Every agent.
    All,
}

/// What part of a transcript a block is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// What the user, an agent or a plugin sent.
    Sent,
    /// The model's answer.
    Reply,
    /// The model's reasoning.
    Reasoning,
    /// A call's arguments, with its tool.
    Arguments(String),
    /// A call's output, with its tool.
    Output(String),
    /// A notice shown in the transcript.
    Notice,
    /// The reply streaming in.
    Streaming,
}

impl Source {
    /// What it is called in the bar.
    pub fn label(&self) -> String {
        match self {
            Self::Sent => "message".to_owned(),
            Self::Reply => "reply".to_owned(),
            Self::Reasoning => "reasoning".to_owned(),
            Self::Arguments(tool) => format!("{tool} call"),
            Self::Output(tool) => format!("{tool} output"),
            Self::Notice => "notice".to_owned(),
            Self::Streaming => "streaming".to_owned(),
        }
    }
}

/// One searchable text of a transcript: a content item of a message, a
/// notice, or the reply streaming in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    /// The agent whose transcript it is in.
    pub agent: Entity,
    /// That agent's name.
    pub agent_name: String,
    /// The message it is in, by index; `None` for a notice or the reply
    /// streaming in.
    pub message: Option<usize>,
    /// Its place in the message, or among the blocks after it.
    pub item: usize,
    /// The user's turn it belongs to, counted from 1; 0 before the first.
    pub turn: usize,
    /// What it is.
    pub source: Source,
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
            hits.extend(
                regex
                    .find_iter(text)
                    .filter(|found| !found.is_empty())
                    .map(|found| Hit {
                        block: index,
                        line,
                        range: found.range(),
                    }),
            );
        }
    }
    hits
}

/// The blocks of `agent`'s transcript, in the order it shows them: each
/// content item of its messages, with the notices written after each
/// message, then the reply streaming in.
pub fn blocks_of(
    agent: Entity,
    agent_name: &str,
    conversation: &Conversation,
    notices: &[SeenNotice],
    streaming: Option<&Partial>,
) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut turn = 0;
    let block = |message, item, turn, source, text: &str| Block {
        agent,
        agent_name: agent_name.to_owned(),
        message,
        item,
        turn,
        source,
        text: text.to_owned(),
    };
    let notices_at = |count: usize, turn: usize, blocks: &mut Vec<Block>| {
        let written = notices
            .iter()
            .filter(|notice| notice.written_in == Some(agent) && notice.after == count);
        for (item, notice) in written.enumerate() {
            blocks.push(block(None, item, turn, Source::Notice, &notice.text));
        }
    };
    let messages = conversation.messages();
    notices_at(0, turn, &mut blocks);
    for (index, entry) in messages.iter().enumerate() {
        match entry {
            message::Message::System { .. } => {}
            message::Message::User { content } => {
                if content
                    .iter()
                    .any(|item| matches!(item, UserContent::Text(_)))
                {
                    turn += 1;
                }
                for (item, part) in content.iter().enumerate() {
                    let (source, text) = match part {
                        UserContent::Text(text) => (Source::Sent, text.text.clone()),
                        UserContent::ToolResult(result) => {
                            let text: Vec<&str> = result
                                .content
                                .iter()
                                .filter_map(|content| content.as_text())
                                .collect();
                            (Source::Output(result.name.to_string()), text.join("\n"))
                        }
                        _ => continue,
                    };
                    blocks.push(block(Some(index), item, turn, source, &text));
                }
            }
            message::Message::Assistant(reply) => {
                for (item, part) in reply.content.iter().enumerate() {
                    let (source, text) = match part {
                        AssistantContent::Text(text) => (Source::Reply, text.text.clone()),
                        AssistantContent::Reasoning(reasoning) => {
                            (Source::Reasoning, reasoning.text.clone())
                        }
                        AssistantContent::ToolCall(call) => {
                            let arguments =
                                serde_json::Value::Object(call.function.arguments.clone());
                            let tool = call.function.name.to_string();
                            (
                                Source::Arguments(tool.clone()),
                                format!("{tool} {arguments}"),
                            )
                        }
                        _ => continue,
                    };
                    blocks.push(block(Some(index), item, turn, source, &text));
                }
            }
        }
        notices_at(index + 1, turn, &mut blocks);
    }
    if let Some(partial) = streaming {
        let parts = [
            (Source::Reasoning, &partial.reasoning),
            (Source::Streaming, &partial.text),
        ];
        for (item, (source, text)) in parts.into_iter().enumerate() {
            if !text.is_empty() {
                blocks.push(block(None, usize::MAX - 1 + item, turn, source, text));
            }
        }
    }
    blocks
}

/// A notice as the search keeps it: in whose transcript it was written,
/// and after how many of that agent's messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeenNotice {
    /// The agent whose transcript shows it: the one it is about, else
    /// the one shown when it came.
    pub written_in: Option<Entity>,
    /// How many messages that agent had then.
    pub after: usize,
    /// Its text.
    pub text: String,
}

/// The notices of this build, oldest first, for searching.
#[derive(Resource, Default, Debug)]
pub struct SeenNotices(pub Vec<SeenNotice>);

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
    /// The agent shown when it opened.
    origin: Option<Entity>,
    /// The query or scope changed since the last search.
    dirty: bool,
}

impl TranscriptSearch {
    fn new(origin: Option<Entity>) -> Self {
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
            origin,
            dirty: true,
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

    /// The lines of the match's block around it, matches lit and the one
    /// shown most, `rows` at most and `width` columns wide.
    pub fn context(&self, rows: usize, width: usize) -> Vec<Line<'static>> {
        let (Some((block, hit)), Some(regex)) = (self.current(), &self.regex) else {
            return Vec::new();
        };
        let lines: Vec<&str> = block.text.lines().collect();
        let first = hit
            .line
            .saturating_sub(rows / 2)
            .min(lines.len().saturating_sub(rows));
        lines
            .iter()
            .enumerate()
            .skip(first)
            .take(rows)
            .map(|(at, text)| {
                let shown = (at == hit.line).then_some(&hit.range);
                lit(regex, text, shown, width)
            })
            .collect()
    }

    /// Searches `blocks` again with the query, keeping the match shown
    /// where it still is; else, while typing, the latest match.
    fn search(&mut self, blocks: Vec<Block>) {
        let was = self.current().map(|(block, hit)| key(block, hit));
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
        let kept = was.and_then(|was| {
            self.hits.iter().position(|hit| {
                self.blocks
                    .get(hit.block)
                    .is_some_and(|block| key(block, hit) == was)
            })
        });
        let last = self.hits.len().checked_sub(1);
        self.current = match kept {
            Some(kept) if !self.editing => Some(kept),
            _ if self.editing => last,
            _ => self
                .current
                .zip(last)
                .map(|(current, last)| current.min(last)),
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
        let current = self
            .current
            .and_then(|current| isize::try_from(current).ok());
        let next = match current {
            Some(current) => (current + by).rem_euclid(count),
            None if by >= 0 => 0,
            None => count - 1,
        };
        self.current = usize::try_from(next).ok();
    }
}

/// What tells a match apart across searches.
fn key(block: &Block, hit: &Hit) -> (Entity, Option<usize>, usize, usize, usize) {
    (
        block.agent,
        block.message,
        block.item,
        hit.line,
        hit.range.start,
    )
}

/// `text` with the matches of `regex` lit, the `shown` one most, starting
/// a little before that one when it would be past `width`.
fn lit(regex: &Regex, text: &str, shown: Option<&Range<usize>>, width: usize) -> Line<'static> {
    let start = shown
        .filter(|range| range.end > width)
        .map_or(0, |range| range.start.saturating_sub(LEAD));
    let start = (0..=start)
        .rev()
        .find(|at| text.is_char_boundary(*at))
        .unwrap_or(0);
    let mut spans = Vec::new();
    if start > 0 {
        spans.push(Span::from("…").dim());
    }
    let mut at = start;
    for found in regex.find_iter(text).filter(|found| !found.is_empty()) {
        if found.start() < start {
            continue;
        }
        spans.push(Span::from(
            text.get(at..found.start()).unwrap_or_default().to_owned(),
        ));
        let style = if shown == Some(&found.range()) {
            Style::new().black().on_yellow().bold()
        } else {
            Style::new().reversed()
        };
        spans.push(Span::styled(found.as_str().to_owned(), style));
        at = found.end();
    }
    spans.push(Span::from(text.get(at..).unwrap_or_default().to_owned()));
    Line::from(spans)
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
        app.init_resource::<SeenNotices>()
            .add_systems(Update, (keep_notices, type_in_bar).chain())
            .add_systems(PostUpdate, draw.in_set(TuiSystems::Draw));
    }
}

/// Keeps each notice, placed where its agent's transcript ended.
fn keep_notices(
    mut notices: MessageReader<Notice>,
    agents: Query<&Conversation>,
    shown: Option<Res<Shown>>,
    mut seen: ResMut<SeenNotices>,
) {
    if notices.is_empty() {
        return;
    }
    for notice in notices.read() {
        let written_in = notice.agent.or(shown.as_ref().and_then(|shown| shown.0));
        let after = written_in
            .and_then(|agent| agents.get(agent).ok())
            .map_or(0, |conversation| conversation.messages().len());
        seen.0.push(SeenNotice {
            written_in,
            after,
            text: notice.text.clone(),
        });
    }
    let excess = seen.0.len().saturating_sub(KEPT_NOTICES);
    seen.0.drain(..excess);
}

/// Ctrl+F: opens the bar with the keys, unless it is open.
fn open(
    pressed: On<KeyPressed>,
    bars: Query<(), With<TranscriptSearch>>,
    shown: Option<Res<Shown>>,
    mut focus: ResMut<KeyFocus>,
    mut commands: Commands,
) {
    if pressed.released || !bars.is_empty() {
        return;
    }
    let origin = shown.and_then(|shown| shown.0);
    let height = Constraint::Length(CONTEXT_LINES + 3);
    let bar = (
        TranscriptSearch::new(origin),
        TuiPanel::new(Placement::Bottom(height)),
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
            &'static Conversation,
            Option<&'static ActiveTurn>,
        ),
        With<Agent>,
    >,
    calls: Query<'w, 's, &'static Calls>,
    partials: Query<'w, 's, &'static Partial>,
    changed: Query<'w, 's, (), Or<(Changed<Conversation>, Changed<Partial>)>>,
    notices: Res<'w, SeenNotices>,
    shown: Option<Res<'w, Shown>>,
}

impl Transcripts<'_, '_> {
    /// Whether anything searched changed since the last frame.
    fn changed(&self) -> bool {
        !self.changed.is_empty()
            || self.notices.is_changed()
            || self.shown.as_ref().is_some_and(|shown| shown.is_changed())
    }

    /// The blocks of the agent shown, or of every agent.
    fn blocks(&self, scope: Scope) -> Vec<Block> {
        let shown = self.shown.as_ref().and_then(|shown| shown.0);
        let mut agents: Vec<_> = self
            .agents
            .iter()
            .filter(|(agent, ..)| scope == Scope::All || Some(*agent) == shown)
            .collect();
        agents.sort_by_key(|(agent, ..)| *agent);
        agents
            .into_iter()
            .flat_map(|(agent, name, conversation, turn)| {
                let streaming = turn
                    .and_then(|turn| self.calls.get(turn.turn()).ok())
                    .and_then(|calls| calls.iter().find_map(|call| self.partials.get(call).ok()));
                blocks_of(
                    agent,
                    name.as_str(),
                    conversation,
                    &self.notices.0,
                    streaming,
                )
            })
            .collect()
    }
}

/// Types into the open bar, in order, then searches again when the query
/// or what it searches changed. It closes when it loses the keys, as on
/// an Esc, showing the agent shown before unless a match was taken. The
/// keys of the frame it opened in were the input's.
fn type_in_bar(
    mut events: MessageReader<TerminalEvent>,
    mut bars: Query<(Entity, &mut TranscriptSearch, Ref<TuiPanel>)>,
    focus: Res<KeyFocus>,
    transcripts: Transcripts,
    mut redraw: MessageWriter<RequestRedraw>,
    mut commands: Commands,
) {
    let Ok((bar, mut search, opened)) = bars.single_mut() else {
        events.clear();
        return;
    };
    let shown = transcripts.shown.as_ref().and_then(|shown| shown.0);
    let close = |search: &TranscriptSearch, commands: &mut Commands| {
        commands.entity(bar).despawn();
        if let Some(origin) = search
            .origin
            .filter(|origin| !search.taken && shown != Some(*origin))
        {
            commands.trigger(Focus { entity: origin });
        }
    };
    if focus.0 != Some(bar) && !opened.is_added() {
        close(&search, &mut commands);
        return;
    }
    if opened.is_added() {
        events.clear();
    }
    let (moved, was_taken) = (search.current, search.taken);
    for TerminalEvent(event) in events.read() {
        match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => {
                let shift = key.modifiers.contains(KeyModifiers::SHIFT);
                let plain = !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT);
                match key.code {
                    KeyCode::Esc => {
                        close(&search, &mut commands);
                        return;
                    }
                    KeyCode::Enter if shift => search.step(-1),
                    KeyCode::Enter if search.editing => {
                        search.editing = false;
                        search.taken = true;
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
                search
                    .query
                    .push_str(text.lines().next().unwrap_or_default());
                search.dirty = true;
            }
            _ => continue,
        }
        if search.dirty {
            let blocks = transcripts.blocks(search.scope);
            search.search(blocks);
        }
        redraw.write(RequestRedraw);
    }
    if !search.dirty && transcripts.changed() {
        let blocks = transcripts.blocks(search.scope);
        search.search(blocks);
        redraw.write(RequestRedraw);
    }
    // A match taken in another agent's transcript shows that agent.
    let jumped = search.taken && (search.current != moved || !was_taken);
    if let Some((block, _)) = search.current().filter(|_| jumped)
        && Some(block.agent) != shown
    {
        commands.trigger(Focus {
            entity: block.agent,
        });
    }
}

/// Draws the bar: the match's block around it, under a line naming where
/// it is, and the query with the count on the border.
fn draw(mut bars: Query<(&TranscriptSearch, &mut PanelCanvas)>) {
    for (search, mut canvas) in &mut bars {
        let area = canvas.area();
        let width = usize::from(area.width.saturating_sub(2));
        let mut lines = Vec::new();
        if let Some((block, _)) = search.current() {
            let place = format!(
                "{} · turn {} · {}",
                block.agent_name,
                block.turn,
                block.source.label()
            );
            lines.push(Line::from(place).dim().bold());
            lines.extend(search.context(usize::from(CONTEXT_LINES), width));
        }
        let scope = match search.scope {
            Scope::Shown => "this agent",
            Scope::All => "all agents",
        };
        let keys = if search.editing {
            "Enter take · Tab scope · Esc close"
        } else {
            "n next · N previous · Esc close"
        };
        let title = Line::from(vec![
            Span::from(format!(" /{} ", search.query)).bold(),
            Span::from(format!(" {} ", search.counter())).yellow(),
            Span::from(format!(" {scope} · {keys} ")).dim(),
        ]);
        canvas.render(Paragraph::new(lines).block(Border::bordered().title(title)));
    }
}

#[cfg(test)]
mod tests;
