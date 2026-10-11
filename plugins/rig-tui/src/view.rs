//! View state, kept apart from the agent core: the focused agent, the
//! input's completion, scrolling, the pickers and questions waiting to be
//! shown, and recent notices.

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;

use super::complete::Completion;
use super::editor::TuiInput;
use super::keys::KeyPress;
use super::panel::Focused;
use super::pick::Choosing;
use super::transcript::Scroll;
use rig_ecs::agent::{Agent, Conversation, Notice, NoticeLevel};
use rig_ecs::inbox::Recalled;
use rig_harness::primary::{PrimaryQuery, primary};

/// Notices kept for display.
const KEPT_NOTICES: usize = 50;

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
    /// The agent shown and typed to.
    pub(crate) agent: Option<Entity>,
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
    /// A refused command put back in the input: sent again unchanged, it
    /// goes to the model as a message.
    pub(crate) refused: Option<String>,
}

/// A notice placed in a transcript.
pub(crate) struct ShownNotice {
    /// The agent it is about, or `None` for every agent.
    agent: Option<Entity>,
    /// The length of that agent's conversation (the focused one's, for an
    /// app notice) when it arrived; it is drawn after that many messages.
    pub(crate) after: usize,
    /// Whether it reports a failure.
    pub(crate) level: NoticeLevel,
    /// The text.
    pub(crate) text: String,
}

impl ShownNotice {
    /// Whether it belongs in `agent`'s transcript.
    pub(crate) fn is_for(&self, agent: Option<Entity>) -> bool {
        self.agent.is_none() || self.agent == agent
    }
}

/// Focuses the first agent by id when the focused one is gone, one the
/// user started before any subagent.
pub(crate) fn focus_agent(mut view: ResMut<TuiView>, agents: PrimaryQuery) {
    if view.agent.is_some_and(|agent| agents.contains(agent)) {
        return;
    }
    view.agent = primary(&agents);
}

/// Keeps [`Focused`] on the agent shown, and on no other.
pub(crate) fn mark_focused(
    view: Res<TuiView>,
    marked: Query<Entity, With<Focused>>,
    mut commands: Commands,
) {
    if !view.is_changed() {
        return;
    }
    for agent in &marked {
        if Some(agent) != view.agent {
            commands.entity(agent).try_remove::<Focused>();
        }
    }
    if let Some(agent) = view.agent
        && !marked.contains(agent)
    {
        commands.entity(agent).try_insert(Focused);
    }
}

/// Shows the agent a [`Focus`] names.
pub(crate) fn on_focus(
    focus: On<Focus>,
    agents: Query<(), With<Agent>>,
    mut view: ResMut<TuiView>,
) {
    if !agents.contains(focus.entity) || view.agent == Some(focus.entity) {
        return;
    }
    view.agent = Some(focus.entity);
    view.scroll.follow();
    view.completion = None;
}

/// Keeps the latest notices for display, each placed after the messages
/// its agent had when it arrived.
pub(crate) fn collect_notices(
    mut notices: MessageReader<Notice>,
    conversations: Query<&Conversation>,
    mut view: ResMut<TuiView>,
) {
    if notices.is_empty() {
        return;
    }
    for notice in notices.read() {
        let after = notice
            .agent
            .or(view.agent)
            .and_then(|agent| conversations.get(agent).ok())
            .map_or(0, |conversation| conversation.messages().len());
        view.notices.push(ShownNotice {
            agent: notice.agent,
            after,
            level: notice.level,
            text: notice.text.clone(),
        });
    }
    let excess = view.notices.len().saturating_sub(KEPT_NOTICES);
    view.notices.drain(..excess);
}

/// Puts what was typed and not sent back in the input, before what is
/// typed there now, and says why a refused command was.
pub(crate) fn recall_messages(
    mut recalled: MessageReader<Recalled>,
    conversations: Query<&Conversation>,
    mut view: ResMut<TuiView>,
    mut input: ResMut<TuiInput>,
) {
    for recalled in recalled.read() {
        if view.agent != Some(recalled.agent) {
            continue;
        }
        if let Some(why) = &recalled.why {
            view.notices.push(ShownNotice {
                agent: Some(recalled.agent),
                after: conversations
                    .get(recalled.agent)
                    .map_or(0, |conversation| conversation.messages().len()),
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
