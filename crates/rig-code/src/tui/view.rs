//! View state, kept apart from the agent core: the focused agent, the input
//! line, scrolling, the open picker and recent notices.

use bevy_ecs::prelude::*;
use rig_core::completion::Reasoning;

use crate::core::agent::{Agent, AgentId, ModelChoice, Notice, PickKind, PickRequest};
use crate::core::models;

/// Notices kept for display.
const KEPT_NOTICES: usize = 50;

/// The terminal view's state. Never saved and never read by the core.
#[derive(Resource, Default)]
pub struct TuiView {
    /// The agent shown and typed to.
    pub agent: Option<Entity>,
    /// The input line.
    pub input: String,
    /// Lines scrolled up from the bottom of the transcript.
    pub scroll: usize,
    /// The open picker.
    pub picker: Option<Picker>,
    /// Recent notices, oldest first.
    pub notices: Vec<String>,
}

/// What choosing a picker item sets.
#[derive(Clone, Debug)]
pub enum PickValue {
    /// A model reference.
    Model(String),
    /// A reasoning setting.
    Effort(Option<Reasoning>),
}

/// A filterable list to choose one item from.
pub struct Picker {
    /// The agent the choice is for.
    pub agent: Entity,
    /// The title.
    pub title: String,
    /// Every item: its label and value.
    pub items: Vec<(String, PickValue)>,
    /// The filter typed so far.
    pub filter: String,
    /// The selected position among the visible items.
    pub selected: usize,
}

impl Picker {
    /// The items whose label holds every word of the filter, ignoring case.
    pub fn visible(&self) -> Vec<&(String, PickValue)> {
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
    pub fn chosen(&self) -> Option<PickValue> {
        self.visible()
            .get(self.selected)
            .map(|(_, value)| value.clone())
    }
}

/// Focuses the first agent by id when the focused one is gone.
pub fn focus_agent(mut view: ResMut<TuiView>, agents: Query<(Entity, &AgentId), With<Agent>>) {
    if view.agent.is_some_and(|agent| agents.contains(agent)) {
        return;
    }
    view.agent = agents
        .iter()
        .min_by(|a, b| a.1.0.cmp(&b.1.0))
        .map(|(entity, _)| entity);
}

/// Opens the picker a command asked for.
pub fn open_pickers(
    mut requests: MessageReader<PickRequest>,
    agents: Query<&ModelChoice>,
    mut view: ResMut<TuiView>,
) {
    for request in requests.read() {
        let current = agents
            .get(request.agent)
            .ok()
            .and_then(|choice| choice.0.as_deref());
        let (title, items) = match request.kind {
            PickKind::Model => {
                let items: Vec<(String, PickValue)> = models::available_models()
                    .into_iter()
                    .map(|spec| {
                        let reference = models::reference(spec);
                        (
                            format!("{reference}  {}", spec.display_name),
                            PickValue::Model(reference),
                        )
                    })
                    .collect();
                if items.is_empty() {
                    view.notices.push(
                        "No provider has a credential in the environment, such as \
                         OPENAI_API_KEY."
                            .to_owned(),
                    );
                    continue;
                }
                ("Model".to_owned(), items)
            }
            PickKind::Effort => {
                let Some(spec) = current.and_then(models::resolve) else {
                    continue;
                };
                let items = models::effort_options(spec)
                    .into_iter()
                    .map(|(label, effort)| (label, PickValue::Effort(effort)))
                    .collect();
                (format!("Reasoning for {}", spec.display_name), items)
            }
        };
        view.picker = Some(Picker {
            agent: request.agent,
            title,
            items,
            filter: String::new(),
            selected: 0,
        });
    }
}

/// Keeps the latest notices for display.
pub fn collect_notices(mut notices: MessageReader<Notice>, mut view: ResMut<TuiView>) {
    for notice in notices.read() {
        view.notices.push(notice.0.clone());
    }
    let excess = view.notices.len().saturating_sub(KEPT_NOTICES);
    view.notices.drain(..excess);
}
