//! View state, kept apart from the agent core: the focused agent, the input
//! editor and its completion, scrolling, the open overlay and recent
//! notices.

use bevy_ecs::prelude::*;
use rig_core::completion::Reasoning;

use super::complete::Completion;
use super::editor::Editor;
use crate::core::agent::{
    Agent, AgentId, Connection, Conversation, Notice, NoticeLevel, PickKind, PickRequest,
};
use crate::core::models;
use crate::host::reload::ReloadFailed;

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

/// What choosing a picker item sets.
#[derive(Clone, Debug)]
pub(crate) enum PickValue {
    /// A model reference.
    Model(String),
    /// A reasoning setting.
    Effort(Option<Reasoning>),
}

/// A filterable list to choose one item from.
pub(crate) struct Picker {
    /// The agent the choice is for.
    pub(crate) agent: Entity,
    /// The title.
    pub(crate) title: String,
    /// Every item: its label and value.
    items: Vec<(String, PickValue)>,
    /// The filter typed so far.
    pub(crate) filter: String,
    /// The selected position among the visible items.
    pub(crate) selected: usize,
}

impl Picker {
    /// The items whose label holds every word of the filter, ignoring case.
    pub(crate) fn visible(&self) -> Vec<&(String, PickValue)> {
        let filter = self.filter.to_lowercase();
        self.items
            .iter()
            .filter(|(label, _)| {
                let label = label.to_lowercase();
                filter.split_whitespace().all(|word| label.contains(word))
            })
            .collect()
    }

    /// The selected item's value.
    pub(crate) fn chosen(&self) -> Option<PickValue> {
        self.visible()
            .get(self.selected)
            .map(|(_, value)| value.clone())
    }
}

/// Focuses the first agent by id when the focused one is gone.
pub(crate) fn focus_agent(
    mut view: ResMut<TuiView>,
    agents: Query<(Entity, &AgentId), With<Agent>>,
) {
    if view.agent.is_some_and(|agent| agents.contains(agent)) {
        return;
    }
    view.agent = agents
        .iter()
        .min_by(|a, b| a.1.0.cmp(&b.1.0))
        .map(|(entity, _)| entity);
}

/// Opens the picker a command asked for.
pub(crate) fn open_pickers(
    mut requests: MessageReader<PickRequest>,
    agents: Query<&Connection>,
    mut view: ResMut<TuiView>,
    mut notices: MessageWriter<Notice>,
) {
    for request in requests.read() {
        let current = agents
            .get(request.agent)
            .ok()
            .map(|connection| connection.spec);
        let (title, items) = match request.kind {
            PickKind::Model => {
                let items: Vec<(String, PickValue)> = models::available_models()
                    .into_iter()
                    .map(|spec| {
                        let reference = models::reference(spec);
                        let keyless = if spec.provider.requires_credential() {
                            ""
                        } else {
                            "  (no key needed)"
                        };
                        (
                            format!("{reference}  {}{keyless}", spec.display_name),
                            PickValue::Model(reference),
                        )
                    })
                    .collect();
                if items.is_empty() {
                    notices.write(Notice::error(
                        request.agent,
                        "No provider with tool-calling models can be reached: set a key such \
                         as OPENAI_API_KEY.",
                    ));
                    continue;
                }
                ("Model".to_owned(), items)
            }
            PickKind::Effort => {
                let Some(spec) = current else {
                    continue;
                };
                let items = models::effort_options(spec)
                    .into_iter()
                    .map(|option| (option.label(), PickValue::Effort(option.1)))
                    .collect();
                (format!("Reasoning for {}", spec.display_name), items)
            }
        };
        view.overlay = Some(Overlay::Picker(Picker {
            agent: request.agent,
            title,
            items,
            filter: String::new(),
            selected: 0,
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
            .map_or(0, |conversation| conversation.0.len());
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
