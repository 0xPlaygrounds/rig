//! View state, kept apart from the agent core: the agent shown (the
//! harness's [`Shown`]), the input's completion, scrolling, the pickers and questions waiting to be
//! shown, and the transcript as plugins read and steer it
//! ([`TranscriptView`]): what is expanded and selected, the text lit,
//! where it is scrolled, and recent notices, which are kept in the
//! session's [`PluginData`] for the next build.

use std::collections::HashSet;
use std::ops::Range;

use bevy_ecs::prelude::*;
use bevy_log::warn;
use bevy_reflect::prelude::*;
use ratatui::style::Style;
use rig_core::completion::Message;
use rig_core::message::CallId;
use serde::{Deserialize, Serialize};

use super::complete::Completion;
use super::editor::TuiInput;
use super::jsonl::{self, Kept};
use super::keys::KeyPress;
use super::panel::Focused;
use super::pick::Choosing;
use super::transcript::Scroll;
use rig_ecs::agent::{Agent, AgentId, Conversation, Notice, NoticeLevel};
use rig_ecs::inbox::Recalled;
use rig_ecs::prelude::{Aside, AsideId, MessageId};
use rig_harness::prelude::{ExitReason, PluginData};
use rig_harness::primary::Shown;

/// Notices kept for display, and read back by the next build.
const KEPT_NOTICES: usize = 200;

/// The file of the session's [`PluginData`] that keeps the notices shown.
const NOTICES_FILE: &str = "tui/notices.jsonl";

/// The file of the session's [`PluginData`] that keeps what was typed and
/// not sent for the build a reload starts.
const DRAFT_FILE: &str = "tui/draft.txt";

/// Ask the view to show the agent and send what is typed to it, such as
/// a subagent picked with `/agents`.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Focus {
    /// The agent.
    pub entity: Entity,
}

/// The terminal view's state. Never saved and never read by the core.
#[derive(Resource, Default)]
pub(crate) struct TuiView {
    /// The open completion list.
    pub(crate) completion: Option<Completion>,
    /// Where the token starts whose completion was closed with Esc.
    pub(crate) dismissed: Option<usize>,
    /// Where the transcript is scrolled.
    pub(crate) scroll: Scroll,
    /// The pickers and questions to show, in the order they came; the
    /// first is shown over the transcript and takes the keys.
    pub(crate) queue: Vec<Entity>,
    /// The one shown, with what is typed in it.
    pub(crate) choosing: Option<Choosing>,
    /// The keys pressed so far of a binding of several.
    pub(crate) pending: Vec<KeyPress>,
    /// A refused command put back in the input: sent again unchanged, it
    /// goes to the model as a message.
    pub(crate) refused: Option<String>,
    /// The open questions shown beside the work, oldest first; the first
    /// is shown.
    pub(crate) inline: Vec<Entity>,
    /// The items of the transcript drawn last, top to bottom, which the
    /// selection moves through.
    pub(crate) items: Vec<TranscriptItem>,
}

/// An item of a transcript, as plugins name it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TranscriptItem {
    /// A message, by id.
    Message(MessageId),
    /// A tool call, by its call id.
    Call(CallId),
    /// A plugin's [`Note`](rig_harness::note::Note), by its aside's id.
    Note(AsideId),
    /// A notice, by its index in [`TranscriptView::notices`].
    Notice(usize),
    /// A content block of a message, an image, audio, a video or a
    /// document, by the message's id and the block's place in it
    /// ([`ContentView`](crate::ContentView)).
    Content(MessageId, usize),
}

/// The terminal view's transcript, as plugins read and steer it. A change
/// to it draws the transcript again. It is there in any run.
///
/// ```no_run
/// use rig_harness::prelude::*;
/// use rig_tui::ratatui::style::{Style, Stylize};
/// use rig_tui::{Highlight, ItemRow, ScrollTo, TranscriptItem, TranscriptView};
///
/// /// Lights a search's matches and shows the first in the message `hit`.
/// fn show_match(agent: Entity, hit: MessageId, mut view: ResMut<TranscriptView>, mut commands: Commands) {
///     let style = Style::new().on_yellow();
///     let flaky = Highlight { text: "flaky".into(), style, ..Highlight::default() };
///     view.highlights = vec![flaky];
///     commands.trigger(ScrollTo { agent, item: Some(TranscriptItem::Message(hit)), row: ItemRow::Lit(0) });
/// }
/// ```
#[derive(Resource, Default)]
pub struct TranscriptView {
    /// Items drawn whole; the rest are drawn cut, as their look says.
    pub expanded: HashSet<TranscriptItem>,
    /// The item the user selected, drawn marked; Ctrl+O expands it.
    pub selected: Option<TranscriptItem>,
    /// Text lit where the transcript draws it, such as a search's
    /// matches; text a wrap splits across two rows is lit on both.
    pub highlights: Vec<Highlight>,
    /// Labels drawn inside replies' text, such as a seam where a plugin
    /// joined two replies. Never saved, never sent to a model.
    pub marks: Vec<TextMark>,
    /// The item and the row of it at the top while the view is scrolled
    /// up; `None` while it follows the newest output.
    top: Option<(TranscriptItem, usize)>,
    /// The transcript's width at the last frame.
    width: Option<u16>,
    /// Recent notices, oldest first.
    notices: Vec<ShownNotice>,
    /// Where they are kept for the next build of the session.
    notices_file: Option<Kept>,
}

/// Whose text a [`TextMark`] is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MarkIn {
    /// Text block `block` (its index among the content) of an assistant
    /// message.
    Message {
        /// The message.
        id: MessageId,
        /// The text block's index among the message's content: not always
        /// 0, as reasoning comes first when there is any.
        block: usize,
    },
    /// The reply a model call is streaming (its `Partial` text), by the
    /// call's entity.
    Streaming(Entity),
}

/// A short label the transcript draws inside a reply's text, before byte
/// `at` of it, markdown kept on both sides: a mark past the end goes at
/// the end, one inside a character goes before it.
///
/// ```
/// use rig_tui::ratatui::style::{Style, Stylize};
/// use rig_tui::{MarkIn, TextMark, TranscriptView};
///
/// fn mark_seam(view: &mut TranscriptView, call: bevy_ecs::entity::Entity, so_far: &str) {
///     view.marks.push(TextMark {
///         r#in: MarkIn::Streaming(call),
///         at: so_far.len(),
///         label: " ⟳ continued ".into(),
///         style: Style::new().dim(),
///     });
/// }
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextMark {
    /// Whose text it is in.
    pub r#in: MarkIn,
    /// The byte of the text it goes before.
    pub at: usize,
    /// What it shows.
    pub label: String,
    /// Its style.
    pub style: Style,
}

/// Text to light in the transcript: wherever its `text` is drawn, or
/// only its `spans`. Where matches of two highlights overlap, the one
/// placed first in [`TranscriptView::highlights`] lights them and the
/// other's match there is not lit, so a highlight placed first, in a style
/// of its own, sets its text apart from the rest's.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Highlight {
    /// The text.
    pub text: String,
    /// The style laid over it.
    pub style: Style,
    /// Whether its case must match too; else it is matched without regard
    /// to case.
    pub match_case: bool,
    /// The matches a plugin found itself, such as a regular expression's,
    /// each lit and counted as a match of `text` is; with any, `text` is
    /// not sought. A span the transcript does not draw is not lit.
    pub spans: Vec<LineSpan>,
}

/// A match a plugin found in a line of the transcript, as
/// [`DrawnTranscript::lines`] gives them, to light with a [`Highlight`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct LineSpan {
    /// The item whose lines `line` counts, from its first; `None` counts
    /// the whole transcript's, for rows of no item.
    pub item: Option<TranscriptItem>,
    /// The line, by its index in the item's
    /// ([`DrawnTranscript::item`]'s [`lines`](DrawnTranscript::lines)).
    pub line: usize,
    /// Its bytes in the line's text; a match a wrap split is lit on each
    /// of its rows and counted in the row it starts in.
    pub bytes: Range<usize>,
}

/// Scrolls the terminal view to `agent`'s transcript, showing that agent
/// if another is shown, so that `row` of `item` is shown; `item: None`
/// follows the newest output again. An item not drawn leaves the view
/// where it is. [`TranscriptView::top`] reads where it is.
#[derive(Event, Clone, Debug)]
pub struct ScrollTo {
    /// The agent.
    pub agent: Entity,
    /// The item to show, or `None` for the newest output.
    pub item: Option<TranscriptItem>,
    /// Which row of the item to show, and where.
    pub row: ItemRow,
}

/// A row of a [`TranscriptItem`] to scroll to, counted in the rows the
/// view draws, which the view works out, wrapping included.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemRow {
    /// This row of the item, put at the top as far as the transcript
    /// reaches, as [`TranscriptView::top`] reads it back.
    Top(usize),
    /// The item's first row, put at the top when it is not on screen.
    Reveal,
    /// The row of the item's match `n` (from 0) of the
    /// [`TranscriptView::highlights`], counted top to bottom in the rows
    /// drawn, put at the top when it is not on screen. With fewer matches
    /// drawn, the item is revealed.
    Lit(usize),
}

/// The transcript as the view drew it at its last frame: every row laid
/// out, on screen or not, as drawn (wrapped, cut where its item is
/// collapsed, as its renderer drew it), with the items they belong to,
/// those drawn cut, and the matches the view lit in each. A plugin counts
/// and scrolls over this, not over the source text.
/// [`TranscriptLayout`](crate::TranscriptLayout) lays one out without
/// drawing it, for any agent, cut as drawn or whole; a plugin's test, with
/// no terminal to draw, inserts such a one as this resource.
///
/// It costs nothing until a plugin adds it with
/// `app.init_resource::<DrawnTranscript>()`. From then on, each frame that
/// draws the transcript copies the text of every row of it, and with
/// [`TranscriptView::highlights`] set counts their matches in each, so its
/// work and memory grow with the transcript; a streaming reply draws at
/// most about 30 frames a second. An item laid out again is also drawn
/// whole once, to tell whether it is cut. Frames that draw the same rows
/// leave it unchanged, so `Res::is_changed` says what is drawn changed.
///
/// ```no_run
/// use rig_harness::prelude::*;
/// use rig_tui::{DrawnTranscript, ItemRow, ScrollTo, TranscriptItem};
///
/// /// Shows the first item with a lit match, by its first one.
/// fn first_match(drawn: Res<DrawnTranscript>, mut commands: Commands) {
///     let Some(agent) = drawn.agent().filter(|_| drawn.is_changed()) else {
///         return;
///     };
///     let mut lit = drawn.items().iter().filter(|(item, _)| drawn.lit_in(item) > 0);
///     if let Some((item, _)) = lit.next() {
///         commands.trigger(ScrollTo { agent, item: Some(item.clone()), row: ItemRow::Lit(0) });
///     }
/// }
/// ```
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct DrawnTranscript {
    agent: Option<Entity>,
    rows: Vec<DrawnRow>,
    items: Vec<(TranscriptItem, Range<usize>)>,
    cut: Vec<TranscriptItem>,
    on_screen: Range<usize>,
}

/// A row of the [`DrawnTranscript`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DrawnRow {
    /// Its text, as drawn.
    pub text: String,
    /// How it continues the row before it, when the view wrapped one
    /// line into both; [`DrawnTranscript::lines`] rejoins them.
    pub continues: Option<Continuation>,
    /// How many matches of each highlight the view lit starting in it,
    /// by its index in [`TranscriptView::highlights`] (where two overlap,
    /// the one placed first lit it); a match a wrap splits counts in the
    /// row it starts in. Empty with no highlights.
    pub lit: Vec<usize>,
}

/// A line of a [`DrawnTranscript`] as it was before the view wrapped it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DrawnLine {
    /// Its rows in the transcript.
    pub rows: Range<usize>,
    /// Its text: its rows rejoined, as each one's [`Continuation`] says.
    pub text: String,
    /// Where each of its rows starts in `text`, by byte; the blanks a wrap
    /// dropped end the row before.
    pub starts: Vec<usize>,
}

impl DrawnLine {
    /// The row of the transcript byte `at` of its text is drawn in.
    pub fn row_at(&self, at: usize) -> usize {
        let before = self.starts.partition_point(|&start| start <= at);
        self.rows.start + before.saturating_sub(1)
    }
}

/// How a [`DrawnRow`] continues the row before it, the view having
/// wrapped one line into both: the line is the row before, then `gap`,
/// then this row's text past its `indent`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Continuation {
    /// The blanks the wrap dropped at the break.
    pub gap: String,
    /// How many spaces the row starts with that the line does not have,
    /// such as those keeping a list item's wrapped text under its text.
    pub indent: usize,
}

impl DrawnRow {
    /// How many matches of every highlight the view lit in it.
    pub fn lit_total(&self) -> usize {
        self.lit.iter().sum()
    }
}

impl DrawnTranscript {
    /// The transcript `agent`'s `rows` make, `items` among them, `cut` of
    /// them drawn cut, with `on_screen` on screen.
    pub(crate) fn new(
        agent: Option<Entity>,
        rows: Vec<DrawnRow>,
        items: Vec<(TranscriptItem, Range<usize>)>,
        cut: Vec<TranscriptItem>,
        on_screen: Range<usize>,
    ) -> Self {
        Self {
            agent,
            rows,
            items,
            cut,
            on_screen,
        }
    }

    /// The agent whose transcript it is; `None` with no agent shown, when
    /// the view draws the notices about every agent.
    pub fn agent(&self) -> Option<Entity> {
        self.agent
    }

    /// Its rows, top to bottom.
    pub fn rows(&self) -> &[DrawnRow] {
        &self.rows
    }

    /// Its lines as they were before the view wrapped them, top to
    /// bottom, each with its rows. Text a wrap split across two rows is
    /// whole here, for a search or a copy.
    pub fn lines(&self) -> impl Iterator<Item = DrawnLine> + '_ {
        let mut start = 0;
        let lines = self.rows.chunk_by(|_, next| next.continues.is_some());
        lines.map(move |rows| {
            let mut text = String::new();
            let mut starts = Vec::with_capacity(rows.len());
            for row in rows {
                match &row.continues {
                    Some(joint) => {
                        text.push_str(&joint.gap);
                        starts.push(text.len());
                        text.push_str(row.text.get(joint.indent..).unwrap_or(&row.text));
                    }
                    None => {
                        starts.push(text.len());
                        text.push_str(&row.text);
                    }
                }
            }
            let line = start..start + rows.len();
            start = line.end;
            DrawnLine {
                rows: line,
                text,
                starts,
            }
        })
    }

    /// Each item's rows, top to bottom by first row. An item's rows hold
    /// those of the items drawn inside it: a message's, its calls and the
    /// notices and notes written inside it. Rows of no item, such as the
    /// reply streaming in, are in none.
    pub fn items(&self) -> &[(TranscriptItem, Range<usize>)] {
        &self.items
    }

    /// The rows of `item`, or `None` when it is not drawn.
    pub fn rows_of(&self, item: &TranscriptItem) -> Option<Range<usize>> {
        let found = self.items.iter().find(|(drawn, _)| drawn == item);
        found.map(|(_, rows)| rows.clone())
    }

    /// The innermost item `row` belongs to, such as a call rather than
    /// its message.
    pub fn item_at(&self, row: usize) -> Option<&TranscriptItem> {
        let holding = self.items.iter().filter(|(_, rows)| rows.contains(&row));
        holding
            .max_by_key(|(_, rows)| rows.start)
            .map(|(item, _)| item)
    }

    /// How many matches of every highlight the view lit in `item`'s rows:
    /// `n` in [`ItemRow::Lit`] counts these, so it finds a row for each `n`
    /// below this.
    pub fn lit_in(&self, item: &TranscriptItem) -> usize {
        let rows = self.rows_of(item).unwrap_or_default();
        let rows = self.rows.get(rows).unwrap_or_default();
        rows.iter().map(DrawnRow::lit_total).sum()
    }

    /// Whether `item` is drawn cut: collapsed, it draws less than it does
    /// expanded, such as a shell call's last lines of output. Expanding it
    /// draws it whole. A message is cut when its own text is, not when
    /// only a call or a note in it is.
    pub fn is_cut(&self, item: &TranscriptItem) -> bool {
        self.cut.contains(item)
    }

    /// The rows on screen; none in one laid out without drawing it.
    pub fn on_screen(&self) -> Range<usize> {
        self.on_screen.clone()
    }

    /// Only `item`'s rows, as a transcript of their own: its rows from 0,
    /// the items drawn in it with which are cut, and its rows on screen.
    /// `None` when it is not drawn.
    pub fn item(&self, item: &TranscriptItem) -> Option<Self> {
        let at = self.items.iter().position(|(drawn, _)| drawn == item)?;
        let rows = self.items.get(at).map(|(_, rows)| rows.clone())?;
        let start = rows.start;
        let within = |inner: &Range<usize>| rows.start <= inner.start && inner.end <= rows.end;
        // An item holding it comes before it, though it may have the same
        // rows, such as a reply that only calls a tool.
        let items: Vec<(TranscriptItem, Range<usize>)> = self
            .items
            .iter()
            .skip(at)
            .filter(|(_, inner)| within(inner))
            .map(|(item, inner)| (item.clone(), inner.start - start..inner.end - start))
            .collect();
        let cut = self
            .cut
            .iter()
            .filter(|cut| items.iter().any(|(item, _)| item == *cut));
        let cut = cut.cloned().collect();
        let on_screen = self.on_screen.start.max(rows.start)..self.on_screen.end.min(rows.end);
        let on_screen = if on_screen.is_empty() {
            0..0
        } else {
            on_screen.start - start..on_screen.end - start
        };
        Some(Self {
            agent: self.agent,
            rows: self.rows.get(rows).unwrap_or_default().to_vec(),
            items,
            cut,
            on_screen,
        })
    }
}

/// A notice placed in a transcript.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShownNotice {
    /// The agent it is about, or `None` for every agent.
    pub agent: Option<AgentId>,
    /// Where it was written in its agent's transcript (the one shown, for
    /// a notice about every agent); `None` before that agent had a
    /// message, or with no agent shown.
    at: Option<Anchor>,
    /// Whether it informs, warns or reports a failure.
    #[serde(with = "level")]
    pub level: NoticeLevel,
    /// The text.
    pub text: String,
}

/// Where a notice was written: after the first `items` content items of
/// message `message` of `agent`'s conversation, which it stays after as
/// messages are added, merged or spliced.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Anchor {
    agent: AgentId,
    message: MessageId,
    items: usize,
}

/// Where a notice or a note is drawn in an agent's transcript, by the
/// index of a message in its [`Conversation`]. Slots order as they are
/// drawn: the top, then each message with the slots inside it by item,
/// then the slot after it. Those in the same slot are drawn in the order
/// they came.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TranscriptSlot {
    /// Before the first message.
    Top,
    /// After this message.
    After(usize),
    /// Inside this message, before this content item.
    Inside(usize, usize),
}

impl TranscriptSlot {
    /// Where `aside` is drawn in `conversation`: after the entry it
    /// follows while that is there, else after the first item of the entry
    /// that stands for it now, such as the summary of a compaction that
    /// took it out.
    pub fn of_aside(aside: &Aside, conversation: &Conversation) -> Self {
        let Some(after) = aside.after else {
            return Self::Top;
        };
        if let Some(index) = conversation.index_of(after) {
            return Self::After(index);
        }
        let Some(last) = conversation.messages().len().checked_sub(1) else {
            return Self::Top;
        };
        let standing = conversation.start_of(after).min(last);
        match items(conversation, standing) {
            0 | 1 => Self::After(standing),
            _ => Self::Inside(standing, 1),
        }
    }

    /// Its place in drawing order.
    fn order(self) -> Option<(usize, bool, usize)> {
        match self {
            Self::Top => None,
            Self::Inside(message, item) => Some((message, false, item)),
            Self::After(message) => Some((message, true, 0)),
        }
    }
}

impl Ord for TranscriptSlot {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.order().cmp(&other.order())
    }
}

impl PartialOrd for TranscriptSlot {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl ShownNotice {
    /// Whether it is shown with no agent shown: it is about every agent.
    pub(crate) fn is_for_every_agent(&self) -> bool {
        self.agent.is_none()
    }

    /// The agent whose transcript draws it: the one it is about, or for a
    /// notice about every agent the one it was written in. `None` for one
    /// about every agent written before any had a message, which every
    /// transcript draws at its top. With no agent shown, the view draws
    /// every notice about every agent.
    pub fn drawn_in(&self) -> Option<&AgentId> {
        let written = self.at.as_ref().map(|at| &at.agent);
        written.or(self.agent.as_ref())
    }

    /// Where it is drawn in the transcript of the agent `id` with
    /// `conversation`, or `None` when it is no part of it: it is about
    /// another agent, or about every agent and was written in another's.
    /// One whose message is gone goes where that message was.
    pub fn slot(&self, id: &AgentId, conversation: &Conversation) -> Option<TranscriptSlot> {
        if self.agent.as_ref().is_some_and(|agent| agent != id) {
            return None;
        }
        let Some(at) = &self.at else {
            return Some(TranscriptSlot::Top);
        };
        if &at.agent != id {
            return None;
        }
        Some(match conversation.index_of(at.message) {
            Some(index) if items(conversation, index) > at.items => {
                TranscriptSlot::Inside(index, at.items)
            }
            Some(index) => TranscriptSlot::After(index),
            None => match conversation.start_of(at.message).checked_sub(1) {
                Some(index) => TranscriptSlot::After(index),
                None => TranscriptSlot::Top,
            },
        })
    }
}

/// The content items of message `index`.
fn items(conversation: &Conversation, index: usize) -> usize {
    match conversation.messages().get(index) {
        Some(Message::User { content }) => content.len(),
        Some(Message::Assistant(assistant)) => assistant.content.len(),
        _ => 0,
    }
}

/// The end of `id`'s conversation, where a notice written now goes.
fn anchor(id: &AgentId, conversation: &Conversation) -> Option<Anchor> {
    let last = conversation.messages().len().checked_sub(1)?;
    Some(Anchor {
        agent: id.clone(),
        message: conversation.id(last)?,
        items: items(conversation, last),
    })
}

/// A [`NoticeLevel`] as a word.
mod level {
    use rig_ecs::agent::NoticeLevel;
    use serde::{Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S: Serializer>(level: &NoticeLevel, out: S) -> Result<S::Ok, S::Error> {
        out.serialize_str(match level {
            NoticeLevel::Info => "info",
            NoticeLevel::Warn => "warn",
            NoticeLevel::Error => "error",
        })
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        input: D,
    ) -> Result<NoticeLevel, D::Error> {
        Ok(match String::deserialize(input)?.as_str() {
            "warn" => NoticeLevel::Warn,
            "error" => NoticeLevel::Error,
            _ => NoticeLevel::Info,
        })
    }
}

impl TranscriptView {
    /// The item and its row at the top while the view is scrolled up;
    /// `None` while it follows the newest output.
    pub fn top(&self) -> Option<&(TranscriptItem, usize)> {
        self.top.as_ref()
    }

    /// The transcript's width in columns at the last frame the view drew;
    /// `None` before one, as in a run without the terminal. The view is
    /// changed when it changes, so `Res::is_changed` on it says so.
    pub fn width(&self) -> Option<u16> {
        self.width
    }

    /// The notices shown, oldest first, those an earlier build of the
    /// session showed included.
    pub fn notices(&self) -> &[ShownNotice] {
        &self.notices
    }

    /// Notes where `view` is scrolled and how wide the transcript is, as
    /// it was drawn: drawing's own bookkeeping, not a change to redraw
    /// for, but for a new width, which a plugin laying the transcript out
    /// at it watches for.
    pub(crate) fn note_drawn(
        view: &mut impl DetectChangesMut<Inner = Self>,
        top: Option<(TranscriptItem, usize)>,
        width: u16,
    ) {
        let noted = view.bypass_change_detection();
        noted.top = top;
        if noted.width.replace(width) != Some(width) {
            view.set_changed();
        }
    }

    /// Expands `item` when collapsed, collapses it when expanded.
    pub(crate) fn toggle(&mut self, item: TranscriptItem) {
        if !self.expanded.remove(&item) {
            self.expanded.insert(item);
        }
    }

    /// Keeps `notice` for display and for the next build, dropping the
    /// oldest past [`KEPT_NOTICES`].
    fn keep(&mut self, notice: ShownNotice) {
        if let Some(file) = &self.notices_file
            && let Err(failure) = jsonl::append(file, [&notice])
        {
            warn!("could not keep a notice for the next build: {failure}");
        }
        self.notices.push(notice);
        let excess = self.notices.len().saturating_sub(KEPT_NOTICES);
        self.notices.drain(..excess);
    }
}

/// Reads back the notices an earlier build of the session showed, and
/// keeps the next ones in the session's [`PluginData`].
pub(crate) fn load_notices(data: Option<Res<PluginData>>, mut view: ResMut<TranscriptView>) {
    let Some(data) = data else {
        return;
    };
    let file = Kept(data.session.clone(), NOTICES_FILE);
    view.notices = jsonl::tail(&file, KEPT_NOTICES);
    view.notices_file = Some(file);
}

/// Keeps what was typed and not sent for the build a reload starts, such
/// as a command typed during the rebuild that only the new build knows.
/// While a question is answered in the input, what was typed before it.
pub(crate) fn keep_draft(
    reason: Option<Res<ExitReason>>,
    data: Option<Res<PluginData>>,
    (input, view): (Res<TuiInput>, Option<Res<TuiView>>),
) {
    let (Some(reason), Some(data)) = (reason, data) else {
        return;
    };
    let answering = view.as_deref().and_then(|view| view.choosing.as_ref());
    let draft = answering.and_then(|choosing| choosing.draft.as_deref());
    let text = draft.unwrap_or(input.text());
    if *reason != ExitReason::Reload || text.is_empty() {
        return;
    }
    if let Err(failure) = data.session.write_file(DRAFT_FILE, text.as_bytes()) {
        warn!("could not keep the input for the next build: {failure}");
    }
}

/// Puts back in the input what the build before a reload kept.
pub(crate) fn load_draft(data: Option<Res<PluginData>>, mut input: ResMut<TuiInput>) {
    let Some(data) = data else {
        return;
    };
    if let Ok(text) = data.session.read_file(DRAFT_FILE) {
        data.session.remove_file(DRAFT_FILE).ok();
        input.set(String::from_utf8_lossy(&text).into_owned());
    }
}

/// Keeps [`Focused`] on the agent shown, and on no other.
pub(crate) fn mark_focused(
    shown: Res<Shown>,
    marked: Query<Entity, With<Focused>>,
    mut commands: Commands,
) {
    if !shown.is_changed() {
        return;
    }
    for agent in &marked {
        if Some(agent) != shown.0 {
            commands.entity(agent).try_remove::<Focused>();
        }
    }
    if let Some(agent) = shown.0
        && !marked.contains(agent)
    {
        commands.entity(agent).try_insert(Focused);
    }
}

/// Shows the agent a [`Focus`] names.
pub(crate) fn on_focus(
    focus: On<Focus>,
    agents: Query<(), With<Agent>>,
    mut shown: ResMut<Shown>,
    mut view: ResMut<TuiView>,
) {
    if !agents.contains(focus.entity) || shown.0 == Some(focus.entity) {
        return;
    }
    shown.0 = Some(focus.entity);
    view.scroll.follow();
    view.completion = None;
}

/// Shows the agent a [`ScrollTo`] names, scrolled to its item, which the
/// next frame finds.
pub(crate) fn on_scroll_to(
    scroll: On<ScrollTo>,
    agents: Query<(), With<Agent>>,
    mut shown: ResMut<Shown>,
    mut view: ResMut<TuiView>,
) {
    if !agents.contains(scroll.agent) {
        return;
    }
    if shown.0 != Some(scroll.agent) {
        shown.0 = Some(scroll.agent);
        view.completion = None;
    }
    match &scroll.item {
        Some(item) => view.scroll.to(item.clone(), scroll.row),
        None => view.scroll.follow(),
    }
}

/// Keeps the latest notices for display, each placed where its agent's
/// conversation ended when it arrived. One about an agent gone before it
/// is read, such as a helper retired in the pass its model call failed,
/// has no transcript left to be drawn in, and is not taken for one about
/// every agent.
pub(crate) fn collect_notices(
    mut notices: MessageReader<Notice>,
    agents: Query<(&AgentId, &Conversation)>,
    shown: Res<Shown>,
    mut view: ResMut<TranscriptView>,
) {
    if notices.is_empty() {
        return;
    }
    for notice in notices.read() {
        let about = match notice.agent {
            Some(agent) => match agents.get(agent) {
                Ok(about) => Some(about),
                Err(_) => continue,
            },
            None => None,
        };
        let written = about.or_else(|| shown.0.and_then(|agent| agents.get(agent).ok()));
        view.keep(ShownNotice {
            agent: about.map(|(id, _)| id.clone()),
            at: written.and_then(|(id, conversation)| anchor(id, conversation)),
            level: notice.level,
            text: notice.text.clone(),
        });
    }
}

/// Puts what was typed and not sent back in the input, before what is
/// typed there now, and says why a refused command was.
pub(crate) fn recall_messages(
    mut recalled: MessageReader<Recalled>,
    agents: Query<(&AgentId, &Conversation)>,
    shown: Res<Shown>,
    (mut view, mut transcript): (ResMut<TuiView>, ResMut<TranscriptView>),
    mut input: ResMut<TuiInput>,
) {
    for recalled in recalled.read() {
        if shown.0 != Some(recalled.agent) {
            continue;
        }
        if let Some(why) = &recalled.why {
            let agent = agents.get(recalled.agent).ok();
            transcript.keep(ShownNotice {
                agent: agent.map(|(id, _)| id.clone()),
                at: agent.and_then(|(id, conversation)| anchor(id, conversation)),
                level: NoticeLevel::Error,
                text: format!("{why} Enter sends it to the model as it is."),
            });
            view.refused = Some(recalled.text.clone());
        }
        let typed = input.take();
        let text = if typed.trim().is_empty() {
            recalled.text.clone()
        } else {
            format!("{}\n\n{typed}", recalled.text)
        };
        input.set(text);
    }
}

#[cfg(test)]
mod tests;
