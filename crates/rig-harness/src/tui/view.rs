//! View state, kept apart from the agent core: the focused agent, the input
//! editor and its completion, scrolling, the open overlay and recent
//! notices.

use bevy_ecs::prelude::*;

use super::complete::Completion;
use super::editor::Editor;
use crate::host::reload::ReloadFailed;
use crate::view::{Focus, PickItem, PickRequest};
use rig_ecs::agent::{Agent, Conversation, Notice, NoticeLevel, PrimaryQuery, primary};
use rig_ecs::inbox::Recalled;

/// Notices kept for display.
const KEPT_NOTICES: usize = 50;

/// The terminal view's state. Never saved and never read by the core.
#[derive(Resource, Default)]
pub(crate) struct TuiView {
    /// The agent shown and typed to.
    pub(crate) agent: Option<Entity>,
    /// The input being typed, and the prompt history.
    pub(crate) editor: Editor,
    /// The open `/` or `@` completion list.
    pub(crate) completion: Option<Completion>,
    /// Where the token starts whose completion was closed with Esc.
    pub(crate) dismissed: Option<usize>,
    /// Lines scrolled up from the bottom of the transcript.
    pub(crate) scroll: usize,
    /// What is shown over the transcript and takes the keys.
    pub(crate) overlay: Option<Overlay>,
    /// Recent notices, oldest first.
    pub(crate) notices: Vec<ShownNotice>,
}

/// What is shown over the transcript, one at a time.
pub(crate) enum Overlay {
    /// A list to pick from.
    Picker(Picker),
    /// A failed rebuild's output, until Esc or Enter.
    ReloadFailure(String),
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

/// A filterable list to choose one item from.
pub(crate) struct Picker {
    /// The agent the choice is for.
    pub(crate) agent: Entity,
    /// The title.
    pub(crate) title: String,
    /// Every item.
    items: Vec<PickItem>,
    /// The filter typed so far.
    pub(crate) filter: String,
    /// The selected position among the visible items.
    pub(crate) selected: usize,
}

impl Picker {
    /// The items whose label holds every word of the filter, ignoring case.
    pub(crate) fn visible(&self) -> Vec<&PickItem> {
        let filter = self.filter.to_lowercase();
        self.items
            .iter()
            .filter(|item| {
                let label = item.label.to_lowercase();
                filter.split_whitespace().all(|word| label.contains(word))
            })
            .collect()
    }

    /// The command line of the selected item.
    pub(crate) fn chosen(&self) -> Option<String> {
        self.visible()
            .get(self.selected)
            .map(|item| item.command.clone())
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
    view.scroll = 0;
    view.completion = None;
}

/// Opens the picker a command asked for.
pub(crate) fn open_pickers(mut requests: MessageReader<PickRequest>, mut view: ResMut<TuiView>) {
    for request in requests.read() {
        view.overlay = Some(Overlay::Picker(Picker {
            agent: request.agent,
            title: request.title.clone(),
            items: request.items.clone(),
            filter: String::new(),
            selected: request.selected,
        }));
    }
}

/// Shows the output of a failed `/reload` until it is dismissed.
pub(crate) fn show_reload_failures(
    mut failures: MessageReader<ReloadFailed>,
    mut view: ResMut<TuiView>,
) {
    if let Some(failure) = failures.read().last() {
        view.overlay = Some(Overlay::ReloadFailure(failure.output.clone()));
    }
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

/// Puts the messages a turn did not send back in the input, before what
/// is typed there now.
pub(crate) fn recall_messages(mut recalled: MessageReader<Recalled>, mut view: ResMut<TuiView>) {
    for recalled in recalled.read() {
        if view.agent != Some(recalled.agent) {
            continue;
        }
        let typed = view.editor.take();
        let text = if typed.trim().is_empty() {
            recalled.text.clone()
        } else {
            format!("{}\n\n{typed}", recalled.text)
        };
        view.editor.set(text);
    }
}
