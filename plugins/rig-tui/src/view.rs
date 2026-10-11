//! View state, kept apart from the agent core: the agent shown (the
//! harness's [`Shown`]), the input's completion, scrolling, the pickers and questions waiting to be
//! shown, and recent notices, which are kept in the session's directory
//! for the next build.

use std::path::PathBuf;

use bevy_ecs::prelude::*;
use bevy_log::warn;
use bevy_reflect::prelude::*;
use rig_core::completion::Message;
use serde::{Deserialize, Serialize};

use super::complete::Completion;
use super::editor::TuiInput;
use super::jsonl;
use super::keys::KeyPress;
use super::panel::Focused;
use super::pick::Choosing;
use super::transcript::Scroll;
use rig_ecs::agent::{Agent, AgentId, Conversation, MessageId, Notice, NoticeLevel};
use rig_ecs::inbox::Recalled;
use rig_harness::prelude::SessionPaths;
use rig_harness::primary::Shown;

/// Notices kept for display, and read back by the next build.
const KEPT_NOTICES: usize = 200;

/// The file in the session's directory that keeps the notices shown.
const NOTICES_FILE: &str = "notices.jsonl";

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
    /// Recent notices, oldest first.
    pub(crate) notices: Vec<ShownNotice>,
    /// Where they are kept for the next build of the session.
    notices_file: Option<PathBuf>,
    /// A refused command put back in the input: sent again unchanged, it
    /// goes to the model as a message.
    pub(crate) refused: Option<String>,
}

/// A notice placed in a transcript.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ShownNotice {
    /// The agent it is about, or `None` for every agent.
    agent: Option<AgentId>,
    /// Where it was written in its agent's transcript (the one shown, for
    /// a notice about every agent); `None` before that agent had a
    /// message, or with no agent shown.
    at: Option<Anchor>,
    /// Whether it informs, warns or reports a failure.
    #[serde(with = "level")]
    pub(crate) level: NoticeLevel,
    /// The text.
    pub(crate) text: String,
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

impl TuiView {
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
pub(crate) fn load_notices(paths: Option<Res<SessionPaths>>, mut view: ResMut<TuiView>) {
    let Some(paths) = paths else {
        return;
    };
    let file = paths.path().join(NOTICES_FILE);
    view.notices = jsonl::tail(&file, KEPT_NOTICES);
    view.notices_file = Some(file);
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

/// Keeps the latest notices for display, each placed where its agent's
/// conversation ended when it arrived.
pub(crate) fn collect_notices(
    mut notices: MessageReader<Notice>,
    agents: Query<(&AgentId, &Conversation)>,
    shown: Res<Shown>,
    mut view: ResMut<TuiView>,
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
    mut view: ResMut<TuiView>,
    mut input: ResMut<TuiInput>,
) {
    for recalled in recalled.read() {
        if shown.0 != Some(recalled.agent) {
            continue;
        }
        if let Some(why) = &recalled.why {
            let agent = agents.get(recalled.agent).ok();
            view.keep(ShownNotice {
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
