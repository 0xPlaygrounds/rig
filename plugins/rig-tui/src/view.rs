//! View state, kept apart from the agent core: the agent shown (the
//! harness's [`Shown`]), the input's completion, scrolling, the pickers and questions waiting to be
//! shown, and the transcript as plugins read and steer it
//! ([`TranscriptView`]): what is expanded and selected, the text lit,
//! where it is scrolled, and recent notices, which are kept in the
//! session's directory for the next build.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use bevy_ecs::prelude::*;
use bevy_log::warn;
use bevy_reflect::prelude::*;
use ratatui::style::Style;
use rig_core::completion::Message;
use rig_core::message::CallId;
use serde::{Deserialize, Serialize};

use super::complete::Completion;
use super::editor::TuiInput;
use super::jsonl;
use super::keys::KeyPress;
use super::panel::Focused;
use super::pick::Choosing;
use super::transcript::Scroll;
use rig_ecs::agent::{Agent, AgentId, Conversation, Notice, NoticeLevel};
use rig_ecs::inbox::Recalled;
use rig_ecs::prelude::MessageId;
use rig_harness::prelude::{ExitReason, SessionPaths};
use rig_harness::primary::Shown;

/// Notices kept for display, and read back by the next build.
const KEPT_NOTICES: usize = 200;

/// The file that keeps the notices shown, in the view's own directory of
/// the session's: a top-level `*.jsonl` there is an agent's log.
const NOTICES_FILE: &str = "tui/notices.jsonl";

/// The file that keeps what was typed and not sent for the build a reload
/// starts, in the view's own directory of the session's.
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
    /// The open `/` or `@` completion list.
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
    /// A plugin's [`TranscriptEntry`](rig_harness::transcript::TranscriptEntry).
    Entry(Entity),
}

/// The terminal view's transcript, as plugins read and steer it. A change
/// to it draws the transcript again. It is there in any run.
///
/// ```no_run
/// use rig_harness::prelude::*;
/// use rig_tui::ratatui::style::{Style, Stylize};
/// use rig_tui::{Highlight, ScrollTo, TranscriptItem, TranscriptView};
///
/// /// Lights a search's matches and shows the message of the first.
/// fn show_match(agent: Entity, hit: MessageId, mut view: ResMut<TranscriptView>, mut commands: Commands) {
///     view.highlights = vec![Highlight { text: "flaky".into(), style: Style::new().on_yellow() }];
///     commands.trigger(ScrollTo { agent, item: Some(TranscriptItem::Message(hit)), line: 0 });
/// }
/// ```
#[derive(Resource, Default)]
pub struct TranscriptView {
    /// Items drawn whole; the rest are drawn cut, as their look says.
    pub expanded: HashSet<TranscriptItem>,
    /// The item the user selected, drawn marked; Ctrl+O expands it.
    pub selected: Option<TranscriptItem>,
    /// Text lit wherever the transcript draws it, such as a search's
    /// matches.
    pub highlights: Vec<Highlight>,
    /// The item and the row of it at the top while the view is scrolled
    /// up; `None` while it follows the newest output.
    top: Option<(TranscriptItem, usize)>,
    /// Recent notices, oldest first.
    notices: Vec<ShownNotice>,
    /// Where they are kept for the next build of the session.
    notices_file: Option<PathBuf>,
}

/// Text to light in the transcript, matched without regard to case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Highlight {
    /// The text.
    pub text: String,
    /// The style laid over it.
    pub style: Style,
}

/// Scrolls the terminal view to `agent`'s transcript, showing that agent
/// if another is shown, so that row `line` of `item` is at the top, as far
/// as the transcript reaches; `item: None` follows the newest output again.
/// [`TranscriptView::top`] reads where it is.
#[derive(Event, Clone, Debug)]
pub struct ScrollTo {
    /// The agent.
    pub agent: Entity,
    /// The item to show, or `None` for the newest output.
    pub item: Option<TranscriptItem>,
    /// The row of the item to put at the top.
    pub line: usize,
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

/// Where a notice is drawn in a transcript.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Slot {
    /// Before the first message.
    Top,
    /// After this message.
    After(usize),
    /// Inside this message, before this content item.
    Inside(usize, usize),
}

impl ShownNotice {
    /// Whether it is shown with no agent shown: it is about every agent.
    pub(crate) fn is_for_every_agent(&self) -> bool {
        self.agent.is_none()
    }

    /// Where it goes in the transcript of the agent `id` with
    /// `conversation`, or `None` when it is no part of it: it is about
    /// another agent, or about every agent and was written in another's.
    /// One whose message is gone goes where that message was.
    pub(crate) fn slot(&self, id: &AgentId, conversation: &Conversation) -> Option<Slot> {
        if self.agent.as_ref().is_some_and(|agent| agent != id) {
            return None;
        }
        let Some(at) = &self.at else {
            return Some(Slot::Top);
        };
        if &at.agent != id {
            return None;
        }
        Some(match conversation.index_of(at.message) {
            Some(index) if items(conversation, index) > at.items => Slot::Inside(index, at.items),
            Some(index) => Slot::After(index),
            None => match conversation.start_of(at.message).checked_sub(1) {
                Some(index) => Slot::After(index),
                None => Slot::Top,
            },
        })
    }
}

/// Where an entry that follows the message `after` is drawn: after it
/// while it is in `conversation`, else after the message that stands for
/// it now, such as the summary of a compaction that took it out.
pub(crate) fn entry_slot(after: Option<MessageId>, conversation: &Conversation) -> Slot {
    let last = conversation.messages().len().checked_sub(1);
    let at = after.and_then(|after| {
        let standing = last.map(|last| conversation.start_of(after).min(last));
        conversation.index_of(after).or(standing)
    });
    at.map_or(Slot::Top, Slot::After)
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

    /// The notices shown, oldest first, those an earlier build of the
    /// session showed included.
    pub fn notices(&self) -> &[ShownNotice] {
        &self.notices
    }

    /// Notes where the view is scrolled, as it was drawn.
    pub(crate) fn set_top(&mut self, top: Option<(TranscriptItem, usize)>) {
        self.top = top;
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
/// keeps the next ones in the session's directory.
pub(crate) fn load_notices(paths: Option<Res<SessionPaths>>, mut view: ResMut<TranscriptView>) {
    let Some(paths) = paths else {
        return;
    };
    let file = paths.path().join(NOTICES_FILE);
    view.notices = jsonl::tail(&file, KEPT_NOTICES);
    view.notices_file = Some(file);
}

/// Keeps what was typed and not sent for the build a reload starts, such
/// as a command typed during the rebuild that only the new build knows.
/// While a question is answered in the input, what was typed before it.
pub(crate) fn keep_draft(
    reason: Option<Res<ExitReason>>,
    paths: Option<Res<SessionPaths>>,
    (input, view): (Res<TuiInput>, Option<Res<TuiView>>),
) {
    let (Some(reason), Some(paths)) = (reason, paths) else {
        return;
    };
    let answering = view.as_deref().and_then(|view| view.choosing.as_ref());
    let draft = answering.and_then(|choosing| choosing.draft.as_deref());
    let text = draft.unwrap_or(input.text());
    if *reason != ExitReason::Reload || text.is_empty() {
        return;
    }
    let file = paths.path().join(DRAFT_FILE);
    let kept = file
        .parent()
        .map_or(Ok(()), fs::create_dir_all)
        .and_then(|()| fs::write(&file, text));
    if let Err(failure) = kept {
        warn!("could not keep the input for the next build: {failure}");
    }
}

/// Puts back in the input what the build before a reload kept.
pub(crate) fn load_draft(paths: Option<Res<SessionPaths>>, mut input: ResMut<TuiInput>) {
    let Some(paths) = paths else {
        return;
    };
    let file = paths.path().join(DRAFT_FILE);
    if let Ok(text) = fs::read_to_string(&file) {
        fs::remove_file(&file).ok();
        input.set(text);
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
        Some(item) => view.scroll.to(item.clone(), scroll.line),
        None => view.scroll.follow(),
    }
}

/// Keeps the latest notices for display, each placed where its agent's
/// conversation ended when it arrived.
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
        let about = notice.agent.and_then(|agent| agents.get(agent).ok());
        let written = notice
            .agent
            .or(shown.0)
            .and_then(|agent| agents.get(agent).ok());
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
