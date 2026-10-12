//! The transcript's wrapped rows, kept per message. A frame re-renders and
//! re-wraps only the messages that changed (a message's look also depends
//! on its neighbours: a tool call is drawn with the result the next message
//! carries, and on the notices written inside it, and on what is expanded),
//! re-wraps without re-rendering when the width changed, and copies out
//! only the rows on screen. Each message keeps where its items are (the
//! message, its calls), so the view can scroll to one, select it and say
//! which is at the top. Before, every frame and every streamed delta
//! re-wrapped the whole history.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use bevy_ecs::prelude::*;
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::Line;
use rig_core::completion::{AssistantContent, Message};
use rig_core::message::{CallId, ToolCall, ToolFunction, ToolResult, UserContent};
use rig_core::streaming::{PartKind, StreamEvent, parse_partial_arguments};
use rig_core::tool::ToolResultContext;

use super::markdown;
use super::render::lit;
use super::renderers::{
    CallState, EntryView, RESULT_LINES, RenderEntry, RenderToolCall, ToolCallView, excerpt,
    result_style, result_text,
};
use super::view::{DrawnRow, DrawnTranscript, Highlight, ItemRow, ShownNotice, TranscriptItem};
use super::wrap::wrap;
use rig_ecs::agent::{Conversation, NoticeLevel, Partial};
use rig_ecs::prelude::{Origin, OriginKind, STOPPED};
use rig_ecs::stage::{Published, Settled, Started};
use rig_ecs::tools::ToolRequest;
use rig_ecs::turn::StreamDelta;
use rig_harness::transcript::TranscriptEntry;
use rig_tools::attach::attached_file;

/// The tool renderers by tool name.
pub(crate) type Renderers<'a> = HashMap<&'a str, &'a Arc<RenderToolCall>>;

/// The entry renderers by origin.
pub(crate) type EntryRenderers<'a> = HashMap<&'a str, &'a Arc<RenderEntry>>;

/// An item's rows (or lines) within a part.
pub(crate) type Marks = Vec<(TranscriptItem, Range<usize>)>;

/// What this run knows of the tool calls that settled in it: what each
/// published and how long it ran, by call id.
#[derive(Resource, Default)]
pub(crate) struct CallFacts(HashMap<CallId, (Option<ToolResultContext>, Option<Duration>)>);

impl CallFacts {
    /// What `call` published and how long it ran, as far as known.
    fn of(&self, call: &CallId) -> (Option<&ToolResultContext>, Option<Duration>) {
        self.0
            .get(call)
            .map_or((None, None), |(facts, took)| (facts.as_ref(), *took))
    }
}

/// Notes what a tool call published and how long it ran, as it settles.
pub(crate) fn note_call_facts(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, Option<&Published>, Option<&Started>)>,
    mut facts: ResMut<CallFacts>,
) {
    let Ok((request, published, started)) = calls.get(settled.entity) else {
        return;
    };
    let took = started.and_then(|started| SystemTime::now().duration_since(started.at).ok());
    let published = published.map(|Published(context)| context.clone());
    facts.0.insert(request.call.id.clone(), (published, took));
}

/// The tool calls a model call is streaming, by the model call, in the
/// order they opened: each as far as its arguments came, or finished.
#[derive(Resource, Default)]
pub(crate) struct StreamingCalls(HashMap<Entity, Vec<(rig_core::streaming::Part, ToolCall)>>);

/// The JSON of a streaming call's arguments so far, by model call and part.
#[derive(Default)]
pub(crate) struct Fragments(HashMap<(Entity, rig_core::streaming::Part), String>);

impl StreamingCalls {
    /// The calls `effect` streams.
    pub(crate) fn of(&self, effect: Entity) -> &[(rig_core::streaming::Part, ToolCall)] {
        self.0.get(&effect).map_or(&[], Vec::as_slice)
    }
}

/// Follows the tool calls models stream: a call opens with its tool's
/// name, its arguments grow, and it ends with the finished call. A model
/// call's go once it is in the conversation, its [`Partial`] gone.
pub(crate) fn follow_streams(
    mut deltas: MessageReader<StreamDelta>,
    live: Query<(), With<Partial>>,
    mut streaming: ResMut<StreamingCalls>,
    mut fragments: Local<Fragments>,
) {
    let calls = streaming.bypass_change_detection();
    let mut changed = false;
    for delta in deltas.read() {
        let effect = delta.effect;
        match &delta.event {
            StreamEvent::Start {
                part,
                kind: PartKind::ToolCall,
                name: Some(name),
            } => {
                let id = CallId::from_wire(format!("streaming-{}", part.index()));
                let call = ToolCall::new(id, ToolFunction::new(name.clone(), Default::default()));
                calls.0.entry(effect).or_default().push((*part, call));
            }
            StreamEvent::Arguments { part, json } => {
                let text = fragments.0.entry((effect, *part)).or_default();
                text.push_str(json);
                let found = calls.0.get_mut(&effect).into_iter().flatten();
                if let Some((_, call)) = found.into_iter().find(|(at, _)| at == part) {
                    call.function.arguments = parse_partial_arguments(text);
                }
            }
            StreamEvent::End {
                part,
                content: AssistantContent::ToolCall(done),
            } => {
                let found = calls.0.get_mut(&effect).into_iter().flatten();
                if let Some((_, call)) = found.into_iter().find(|(at, _)| at == part) {
                    *call = done.clone();
                }
            }
            _ => continue,
        }
        changed = true;
    }
    let before = calls.0.len();
    calls.0.retain(|effect, _| live.contains(*effect));
    fragments.0.retain(|(effect, _), _| live.contains(*effect));
    if changed || calls.0.len() != before {
        streaming.set_changed();
    }
}

/// What the messages' look depends on besides the messages.
pub(crate) struct Look<'a> {
    /// The tool renderers by tool name.
    pub(crate) renderers: &'a Renderers<'a>,
    /// The entry renderers by origin.
    pub(crate) entries: &'a EntryRenderers<'a>,
    /// The state of each call of the running turn.
    pub(crate) calls: &'a BTreeMap<CallId, CallState>,
    /// The output so far of each of its running calls that reports it.
    pub(crate) outputs: &'a BTreeMap<CallId, String>,
    /// The notices and entries drawn inside a message, by its index, each
    /// with the content item it goes before.
    pub(crate) inside: HashMap<usize, Vec<(usize, Inside<'a>)>>,
    /// The items drawn whole.
    pub(crate) expanded: &'a HashSet<TranscriptItem>,
    /// What settled calls published and how long they ran.
    pub(crate) facts: &'a CallFacts,
}

/// What is drawn inside a message, before one of its items.
#[derive(Clone, Copy)]
pub(crate) enum Inside<'a> {
    /// A notice written there, and its index in the notices shown.
    Notice(usize, &'a ShownNotice),
    /// An entry whose message was taken out, after the start of what
    /// stands for it.
    Entry(Entity, &'a TranscriptEntry),
}

impl Look<'_> {
    /// The notices and entries inside message `index`.
    fn inside(&self, index: usize) -> &[(usize, Inside<'_>)] {
        self.inside.get(&index).map_or(&[], Vec::as_slice)
    }

    /// Hashes the notices and entries inside message `index`.
    fn hash_inside(&self, index: usize, hasher: &mut DefaultHasher) {
        for (item, inside) in self.inside(index) {
            item.hash(hasher);
            match inside {
                Inside::Notice(at, notice) => {
                    at.hash(hasher);
                    level_code(notice.level).hash(hasher);
                    notice.text.hash(hasher);
                }
                Inside::Entry(entity, entry) => {
                    entity.hash(hasher);
                    (&entry.origin, &entry.title, &entry.text).hash(hasher);
                    let expanded = TranscriptItem::Entry(*entity);
                    self.expanded.contains(&expanded).hash(hasher);
                }
            }
        }
    }

    /// Where the call `call` stands: as its effect says while it has one,
    /// else done once it has `result`.
    fn state(&self, call: &CallId, result: Option<&ToolResult>) -> CallState {
        match self.calls.get(call) {
            Some(state) => state.clone(),
            None if result.is_some() => CallState::Done,
            None => CallState::Running(None),
        }
    }

    /// Hashes what message `index` (with `id`) looks like besides its
    /// messages: its notices, whether it is expanded, and its calls' states,
    /// output so far, expansion and running times.
    fn hash_message(
        &self,
        (index, id): (usize, Option<rig_ecs::prelude::MessageId>),
        message: &Message,
        hasher: &mut DefaultHasher,
    ) {
        self.hash_inside(index, hasher);
        id.map(|id| self.expanded.contains(&TranscriptItem::Message(id)))
            .hash(hasher);
        if let Message::Assistant(assistant) = message {
            for item in assistant.content.iter() {
                if let AssistantContent::ToolCall(call) = item {
                    self.calls.get(&call.id).hash(hasher);
                    self.outputs.get(&call.id).hash(hasher);
                    let expanded = TranscriptItem::Call(call.id.clone());
                    self.expanded.contains(&expanded).hash(hasher);
                    self.facts.of(&call.id).1.hash(hasher);
                }
            }
        }
    }

    /// A hash of all of it but the renderers, whose changes
    /// [`Transcript::clear`] handles.
    fn digest(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        let mut indices: Vec<&usize> = self.inside.keys().collect();
        indices.sort();
        for index in indices {
            index.hash(&mut hasher);
            self.hash_inside(*index, &mut hasher);
        }
        self.calls.hash(&mut hasher);
        self.outputs.hash(&mut hasher);
        // Sets in no order: a sum of each one's hash.
        let unordered = |item: &dyn Fn(&mut DefaultHasher)| {
            let mut hasher = DefaultHasher::new();
            item(&mut hasher);
            hasher.finish()
        };
        let expanded = self
            .expanded
            .iter()
            .map(|item| unordered(&|h| item.hash(h)));
        expanded.fold(0u64, u64::wrapping_add).hash(&mut hasher);
        let facts = self.facts.0.iter();
        let facts = facts.map(|(call, (_, took))| unordered(&|h| (call, took).hash(h)));
        facts.fold(0u64, u64::wrapping_add).hash(&mut hasher);
        hasher.finish()
    }
}

/// A number for each notice level.
fn level_code(level: NoticeLevel) -> u8 {
    match level {
        NoticeLevel::Info => 0,
        NoticeLevel::Warn => 1,
        NoticeLevel::Error => 2,
    }
}

/// One message's look: its lines and where its items are in them, and
/// those lines wrapped to the width with where its items are in the rows.
struct Cached {
    fingerprint: u64,
    lines: Vec<Line<'static>>,
    line_marks: Marks,
    rows: Vec<Line<'static>>,
    marks: Marks,
}

/// The wrapped rows of the shown agent's messages.
#[derive(Default)]
pub(crate) struct Transcript {
    agent: Option<Entity>,
    width: u16,
    /// The [`Look::digest`] the rows were made with.
    look: u64,
    messages: Vec<Cached>,
}

/// A part of the transcript, in order: a message's cached rows, or rows
/// made for this frame (notices, an entry, the reply streaming in), with
/// where the items among them are.
pub(crate) enum Part {
    Message(usize),
    Rows(Vec<Line<'static>>, Marks),
}

impl Part {
    /// `lines`, with the items `marks` says are in them, wrapped to `width`.
    pub(crate) fn rows(
        lines: &[Line<'static>],
        marks: &[(TranscriptItem, Range<usize>)],
        width: usize,
    ) -> Self {
        let (rows, marks) = wrap_marked(lines, marks, width);
        Self::Rows(rows, marks)
    }
}

/// The rows of the transcript on screen, and where it is.
pub(crate) struct Visible {
    /// The rows shown.
    pub(crate) rows: Vec<Line<'static>>,
    /// What is below them.
    pub(crate) below: Below,
    /// The first one's row in the whole transcript.
    pub(crate) first: usize,
    /// Every item's rows in the whole transcript, top to bottom; a call's
    /// are within its message's.
    pub(crate) marks: Marks,
    /// The item at the top and the row of it there, while the view is
    /// scrolled up.
    pub(crate) top: Option<(TranscriptItem, usize)>,
}

impl Transcript {
    /// Brings the rows up to date with the conversation of `agent` and
    /// `look` at `width`.
    /// `changed` says whether the messages may have changed since the last
    /// call; when neither they nor the look did, only a new agent or width
    /// costs anything.
    pub(crate) fn update(
        &mut self,
        agent: Entity,
        conversation: &Conversation,
        changed: bool,
        look: &Look<'_>,
        width: u16,
    ) {
        if self.agent != Some(agent) {
            self.agent = Some(agent);
            self.messages.clear();
        }
        let usable = usize::from(width.max(1));
        if self.width != width {
            self.width = width;
            for cached in &mut self.messages {
                (cached.rows, cached.marks) =
                    wrap_marked(&cached.lines, &cached.line_marks, usable);
            }
        }
        let messages = conversation.messages();
        let digest = look.digest();
        let looks_alike = std::mem::replace(&mut self.look, digest) == digest;
        if !changed && looks_alike && self.messages.len() == messages.len() {
            return;
        }
        self.messages.truncate(messages.len());
        for (index, message) in messages.iter().enumerate() {
            let previous = index.checked_sub(1).and_then(|before| messages.get(before));
            let next = messages.get(index + 1);
            let id = conversation.id(index);
            let fingerprint = fingerprint([previous, Some(message), next], look, (index, id));
            if self
                .messages
                .get(index)
                .is_some_and(|cached| cached.fingerprint == fingerprint)
            {
                continue;
            }
            let (lines, line_marks) =
                message_lines(message, (conversation, index), previous, next, look);
            let (rows, marks) = wrap_marked(&lines, &line_marks, usable);
            let cached = Cached {
                fingerprint,
                lines,
                line_marks,
                rows,
                marks,
            };
            match self.messages.get_mut(index) {
                Some(slot) => *slot = cached,
                None => self.messages.push(cached),
            }
        }
    }

    /// Drops every cached row, after the renderers changed.
    pub(crate) fn clear(&mut self) {
        self.messages.clear();
    }

    /// A part's rows.
    fn rows<'a>(&'a self, part: &'a Part) -> &'a [Line<'static>] {
        match part {
            Part::Message(index) => self
                .messages
                .get(*index)
                .map(|cached| cached.rows.as_slice())
                .unwrap_or_default(),
            Part::Rows(rows, _) => rows.as_slice(),
        }
    }

    /// Where a part's items are in its rows.
    fn marks(&self, part: &Part) -> Marks {
        match part {
            Part::Message(index) => self
                .messages
                .get(*index)
                .map(|cached| cached.marks.clone())
                .unwrap_or_default(),
            Part::Rows(_, marks) => marks.clone(),
        }
    }

    /// The rows of `parts` that fit `height` rows at `width`, where
    /// `scroll` places them, what is below them, and where the items are.
    /// A jump to an item's match counts the matches of `highlights`.
    pub(crate) fn visible(
        &self,
        parts: &[Part],
        (width, height): (u16, usize),
        scroll: &mut Scroll,
        highlights: &[Highlight],
    ) -> Visible {
        let counts: Vec<usize> = parts.iter().map(|part| self.rows(part).len()).collect();
        let mut marks = Marks::new();
        let mut start = 0;
        for (part, count) in parts.iter().zip(&counts) {
            let shifted = self.marks(part).into_iter();
            marks.extend(shifted.map(|(item, rows)| (item, rows.start + start..rows.end + start)));
            start += count;
        }
        scroll.find(&marks);
        let (mut top, mut below) = scroll.place(&counts, height, width);
        if let Some((item, to)) = scroll.jump.take() {
            let rows = marks.iter().find(|(marked, _)| *marked == item);
            let row = rows.map(|(_, rows)| {
                let lit = match to {
                    ItemRow::Lit(n) => self.lit_row(parts, rows, highlights, n),
                    ItemRow::Top(_) | ItemRow::Reveal => None,
                };
                lit.unwrap_or(rows.start)
            });
            if let Some(row) = row.filter(|row| !(top..top + height).contains(row)) {
                scroll.at = Some(row);
                (top, below) = scroll.place(&counts, height, width);
            }
        }
        let mut shown = Vec::with_capacity(height);
        let mut start = 0;
        for (part, count) in parts.iter().zip(&counts) {
            let end = start + count;
            if end > top && start < top + height {
                let from = top.saturating_sub(start);
                let to = (top + height - start).min(*count);
                shown.extend(
                    self.rows(part)
                        .get(from..to)
                        .unwrap_or_default()
                        .iter()
                        .cloned(),
                );
            }
            start = end;
            if start >= top + height {
                break;
            }
        }
        let at_top = scroll.held.is_some().then(|| {
            // The innermost item there: the one that starts last.
            let holding = marks.iter().filter(|(_, rows)| rows.contains(&top));
            let item = holding.max_by_key(|(_, rows)| rows.start);
            item.map(|(item, rows)| (item.clone(), top - rows.start))
        });
        Visible {
            rows: shown,
            below,
            first: top,
            marks,
            top: at_top.flatten(),
        }
    }

    /// The transcript of `agent` as `parts` and `visible` drew it, with the
    /// matches of `highlights` in each row, counted only when one is set.
    pub(crate) fn drawn(
        &self,
        agent: Option<Entity>,
        parts: &[Part],
        visible: &Visible,
        highlights: &[Highlight],
    ) -> DrawnTranscript {
        let lighting = highlights
            .iter()
            .any(|highlight| !highlight.text.is_empty());
        let rows = parts.iter().flat_map(|part| self.rows(part));
        let rows = rows.map(|row| DrawnRow {
            text: row.to_string(),
            lit: if lighting {
                lit(row, highlights).1
            } else {
                Vec::new()
            },
        });
        let on_screen = visible.first..visible.first + visible.rows.len();
        DrawnTranscript::new(agent, rows.collect(), visible.marks.clone(), on_screen)
    }

    /// The row among `rows` of `parts` with match `n` of `highlights`.
    fn lit_row(
        &self,
        parts: &[Part],
        rows: &Range<usize>,
        highlights: &[Highlight],
        n: usize,
    ) -> Option<usize> {
        let drawn = parts.iter().flat_map(|part| self.rows(part));
        let mut before = 0;
        for (at, row) in rows.clone().zip(drawn.skip(rows.start)) {
            before += lit(row, highlights).1.iter().sum::<usize>();
            if before > n {
                return Some(at);
            }
        }
        None
    }
}

/// `lines` wrapped to `width`, and `marks` of lines moved to the rows
/// they wrap to.
fn wrap_marked(
    lines: &[Line<'static>],
    marks: &[(TranscriptItem, Range<usize>)],
    width: usize,
) -> (Vec<Line<'static>>, Marks) {
    let mut rows = Vec::with_capacity(lines.len());
    // The first row of each line, and past the last.
    let mut starts = Vec::with_capacity(lines.len() + 1);
    for line in lines {
        starts.push(rows.len());
        rows.extend(wrap(line, width));
    }
    starts.push(rows.len());
    let row = |line: usize| starts.get(line).copied().unwrap_or(rows.len());
    let marks = marks
        .iter()
        .map(|(item, lines)| (item.clone(), row(lines.start)..row(lines.end)))
        .collect();
    (rows, marks)
}

/// Where the transcript is scrolled: following its bottom, or held at the
/// row the user scrolled to, which then stays put as rows arrive below.
#[derive(Default)]
pub(crate) struct Scroll {
    /// The row held at the top of the screen; `None` follows the bottom.
    held: Option<Held>,
    /// Rows the keys moved the view since the last frame, up positive.
    moved: isize,
    /// An item to go to at the next frame, and which row of it.
    jump: Option<(TranscriptItem, ItemRow)>,
    /// The row to put at the top at the next frame, where a jump found it.
    at: Option<usize>,
}

/// The top row of a view scrolled up.
struct Held {
    /// Its index in the whole transcript.
    top: usize,
    /// The part it is in, its row there and that part's rows, to find it
    /// again after a resize re-wraps everything above it.
    part: usize,
    row: usize,
    rows: usize,
    /// The transcript's width and rows at the last frame.
    width: u16,
    total: usize,
    /// Whether rows arrived below since the view was held.
    grew: bool,
}

/// What is below a view scrolled up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Below {
    /// Nothing: the view follows the bottom.
    Nothing,
    /// Rows that were there when the view was scrolled up.
    More,
    /// Rows that arrived since.
    New,
}

impl Scroll {
    /// Moves the view `rows` up.
    pub(crate) fn up(&mut self, rows: usize) {
        self.moved = self
            .moved
            .saturating_add(isize::try_from(rows).unwrap_or(isize::MAX));
    }

    /// Moves the view `rows` down; reaching the bottom follows it again.
    pub(crate) fn down(&mut self, rows: usize) {
        self.moved = self
            .moved
            .saturating_sub(isize::try_from(rows).unwrap_or(isize::MAX));
    }

    /// Follows the bottom again.
    pub(crate) fn follow(&mut self) {
        *self = Self::default();
    }

    /// Shows `row` of `item` at the next frame.
    pub(crate) fn to(&mut self, item: TranscriptItem, row: ItemRow) {
        self.jump = Some((item, row));
    }

    /// Finds the row a jump to put a row of an item at the top goes to
    /// among `marks`: the item's first row and the row asked for, within
    /// the item. An item not there leaves the view where it is.
    fn find(&mut self, marks: &[(TranscriptItem, Range<usize>)]) {
        if let Some((item, ItemRow::Top(line))) = &self.jump {
            let found = marks.iter().find(|(marked, _)| marked == item);
            self.at = found.map(|(_, rows)| {
                let last = rows.end.saturating_sub(1).max(rows.start);
                (rows.start + line).min(last)
            });
            self.jump = None;
        }
    }

    /// The top row of a view `height` rows high over parts of `counts`
    /// rows at `width`, after the keys' moves or a jump; holds it while it
    /// is above the bottom.
    fn place(&mut self, counts: &[usize], height: usize, width: u16) -> (usize, Below) {
        let total: usize = counts.iter().sum();
        let bottom = total.saturating_sub(height);
        let moved = std::mem::take(&mut self.moved);
        let (top, grew) = match (self.at.take(), &self.held) {
            (Some(at), _) => (at, false),
            (None, None) => (bottom, false),
            (None, Some(held)) if held.width == width => {
                (held.top, held.grew || total > held.total)
            }
            (None, Some(held)) => {
                let start: usize = counts.iter().take(held.part).sum();
                let rows = counts.get(held.part).copied().unwrap_or(0);
                let row = (held.row * rows / held.rows.max(1)).min(rows.saturating_sub(1));
                (start + row, held.grew)
            }
        };
        let top = top.saturating_add_signed(moved.saturating_neg());
        if top >= bottom {
            self.held = None;
            return (bottom, Below::Nothing);
        }
        let (mut part, mut start) = (0, 0);
        for (index, count) in counts.iter().enumerate() {
            part = index;
            if start + count > top {
                break;
            }
            start += count;
        }
        self.held = Some(Held {
            top,
            part,
            row: top - start,
            rows: counts.get(part).copied().unwrap_or(0),
            width,
            total,
            grew,
        });
        (top, if grew { Below::New } else { Below::More })
    }
}

/// A hash of what message `index`'s look depends on: it and its
/// neighbours, and `look`'s part for it.
fn fingerprint(
    messages: [Option<&Message>; 3],
    look: &Look<'_>,
    at: (usize, Option<rig_ecs::prelude::MessageId>),
) -> u64 {
    let mut hasher = DefaultHasher::new();
    if let [_, Some(message), _] = messages {
        look.hash_message(at, message, &mut hasher);
    }
    for message in messages {
        match message {
            Some(message) => {
                hasher.write_u8(1);
                hasher.write(&serde_json::to_vec(message).unwrap_or_default());
            }
            None => hasher.write_u8(0),
        }
    }
    hasher.finish()
}

/// The tool results a message carries.
fn tool_results(message: Option<&Message>) -> impl Iterator<Item = &ToolResult> {
    let content = match message {
        Some(Message::User { content }) => Some(content.iter()),
        _ => None,
    };
    content.into_iter().flatten().filter_map(|item| match item {
        UserContent::ToolResult(result) => Some(result),
        _ => None,
    })
}

/// Whether `message` is a reply holding the call `result` answers.
fn answers(message: Option<&Message>, result: &ToolResult) -> bool {
    match message {
        Some(Message::Assistant(assistant)) => assistant
            .content
            .iter()
            .any(|item| matches!(item, AssistantContent::ToolCall(call) if call.id == result.call)),
        _ => false,
    }
}

/// Draws `message`, the `at`-th of `conversation`, with the notices and
/// entries inside it before their items, and says which lines are the
/// message and which each of its calls and entries. A tool call's result, which comes
/// in the `next` message, is drawn with the call, so the results of a
/// reply's calls are not drawn after all of its calls.
fn message_lines(
    message: &Message,
    (conversation, at): (&Conversation, usize),
    previous: Option<&Message>,
    next: Option<&Message>,
    look: &Look<'_>,
) -> (Vec<Line<'static>>, Marks) {
    let mut lines = Vec::new();
    let mut marks = Marks::new();
    let id = conversation.id(at);
    let expanded = id.is_some_and(|id| look.expanded.contains(&TranscriptItem::Message(id)));
    let inside = look.inside(at);
    let inside_before = |item: usize, lines: &mut Vec<Line<'static>>, marks: &mut Marks| {
        for (_, inside) in inside.iter().filter(|(before, _)| *before == item) {
            match inside {
                Inside::Notice(at, notice) => notice_lines((*at, notice), lines, marks),
                Inside::Entry(entity, entry) => {
                    let item = TranscriptItem::Entry(*entity);
                    let view = EntryView {
                        origin: &entry.origin,
                        title: &entry.title,
                        text: &entry.text,
                        expanded: look.expanded.contains(&item),
                    };
                    let start = lines.len();
                    lines.push(Line::default());
                    lines.extend(entry_lines(&view, look.entries));
                    marks.push((item, start..lines.len()));
                }
            }
        }
    };
    match message {
        Message::System { .. } => {}
        Message::User { content } => {
            for (item_at, item) in content.iter().enumerate() {
                inside_before(item_at, &mut lines, &mut marks);
                match item {
                    UserContent::Text(text) => {
                        if let Some((label, count)) = attached_file(&text.text) {
                            lines.push(Line::from(format!("  [{label}, {count} lines]")).dim());
                            continue;
                        }
                        if text.text == STOPPED {
                            lines.push(Line::from("  [stopped before it was answered]").dim());
                            continue;
                        }
                        lines.push(Line::default());
                        match conversation.origin(at, item_at) {
                            Some(origin) => {
                                lines.extend(delivered_lines(origin, &text.text, expanded, look));
                            }
                            None => user_lines(&text.text, &mut lines),
                        }
                    }
                    // Drawn under its call already.
                    UserContent::ToolResult(result) if answers(previous, result) => {}
                    UserContent::ToolResult(result) => lines.extend(excerpt(
                        &result_text(result),
                        if expanded { usize::MAX } else { RESULT_LINES },
                        result_style(result.is_error),
                    )),
                    UserContent::Image(_) => lines.push(Line::from("  [image]").dim()),
                    _ => lines.push(Line::from("  [attachment]").dim()),
                }
            }
        }
        Message::Assistant(assistant) => {
            for (item_at, item) in assistant.content.iter().enumerate() {
                inside_before(item_at, &mut lines, &mut marks);
                match item {
                    AssistantContent::Text(text) => {
                        lines.push(Line::default());
                        lines.extend(markdown::render(&text.text));
                    }
                    AssistantContent::Reasoning(reasoning) => {
                        lines.extend(plain_lines(&reasoning.text, Style::new().dim().italic()));
                    }
                    AssistantContent::ToolCall(call) => {
                        let result = tool_results(next).find(|result| result.call == call.id);
                        let (facts, took) = look.facts.of(&call.id);
                        let item = TranscriptItem::Call(call.id.clone());
                        let view = ToolCallView {
                            call,
                            result,
                            state: look.state(&call.id, result),
                            expanded: look.expanded.contains(&item),
                            facts,
                            took,
                            output: look.outputs.get(&call.id).map(String::as_str),
                        };
                        let start = lines.len();
                        lines.extend(call_lines(&view, look.renderers));
                        marks.push((item, start..lines.len()));
                    }
                    _ => {}
                }
            }
        }
    }
    if let Some(id) = id {
        marks.insert(0, (TranscriptItem::Message(id), 0..lines.len()));
    }
    (lines, marks)
}

/// A tool call's lines, by its tool's renderer or the default look.
pub(crate) fn call_lines(view: &ToolCallView<'_>, renderers: &Renderers<'_>) -> Vec<Line<'static>> {
    match renderers.get(view.name()) {
        Some(render) => render(view),
        None => view.default_lines(),
    }
}

/// An entry's lines, by its origin's renderer or the default look.
pub(crate) fn entry_lines(
    view: &EntryView<'_>,
    renderers: &EntryRenderers<'_>,
) -> Vec<Line<'static>> {
    match renderers.get(view.origin) {
        Some(render) => render(view),
        None => view.default_lines(),
    }
}

/// What the user typed, under a `›`.
fn user_lines(text: &str, lines: &mut Vec<Line<'static>>) {
    let style = Style::new().cyan().bold();
    for (index, line) in text.lines().enumerate() {
        let prefix = if index == 0 { "› " } else { "  " };
        lines.push(Line::styled(
            format!("{prefix}{}", line.replace('\t', "    ")),
            style,
        ));
    }
}

/// Text an agent or a plugin sent, without the header the model reads:
/// drawn by the renderer of the plugin that sent it, if it has one, else
/// as where it came from and its start.
fn delivered_lines(
    origin: &Origin,
    text: &str,
    expanded: bool,
    look: &Look<'_>,
) -> Vec<Line<'static>> {
    let body = origin
        .header()
        .and_then(|header| text.strip_prefix(&header))
        .map_or(text, |body| body.trim_start_matches('\n'));
    let label = origin.label();
    let name = match &origin.kind {
        OriginKind::Plugin(name) => name.as_str(),
        _ => "",
    };
    let view = EntryView {
        origin: name,
        title: &label,
        text: body,
        expanded,
    };
    entry_lines(&view, look.entries)
}

/// The lines of the notice `at` in the notices shown, coloured by its
/// level, marked as that item.
pub(crate) fn notice_lines(
    (at, notice): (usize, &ShownNotice),
    lines: &mut Vec<Line<'static>>,
    marks: &mut Marks,
) {
    let style = match notice.level {
        NoticeLevel::Info => Style::new().fg(Color::Magenta),
        NoticeLevel::Warn => Style::new().fg(Color::Yellow),
        NoticeLevel::Error => Style::new().fg(Color::Red),
    };
    let start = lines.len();
    lines.extend(plain_lines(&notice.text, style));
    marks.push((TranscriptItem::Notice(at), start..lines.len()));
}

/// `text`'s lines in one style.
pub(crate) fn plain_lines(text: &str, style: Style) -> Vec<Line<'static>> {
    text.lines()
        .map(|line| Line::styled(line.replace('\t', "    "), style))
        .collect()
}

#[cfg(test)]
mod tests;
