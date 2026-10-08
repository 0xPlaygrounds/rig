use bevy_ecs::prelude::*;

use crate::agent::{Agent, Choice, Choose, Notice};

/// How many notices the view keeps.
const KEPT_NOTICES: usize = 4;

/// The terminal view's own state, apart from the agents it shows.
#[derive(Resource, Debug)]
pub struct TuiView {
    /// The agent shown and addressed.
    pub agent: Option<Entity>,
    /// The input line.
    pub input: String,
    /// The cursor in the input line, in characters.
    pub cursor: usize,
    /// Transcript rows scrolled up from the bottom.
    pub scroll: usize,
    /// The open picker, if any.
    pub picker: Option<Picker>,
    /// Recent notices, oldest first. Cleared on the next submit.
    pub notices: Vec<String>,
    /// Whether the screen needs drawing.
    pub dirty: bool,
}

impl Default for TuiView {
    fn default() -> Self {
        Self {
            agent: None,
            input: String::new(),
            cursor: 0,
            scroll: 0,
            picker: None,
            notices: Vec::new(),
            dirty: true,
        }
    }
}

/// A filterable list of command lines to pick from.
#[derive(Debug, Clone)]
pub struct Picker {
    /// What is being picked.
    pub title: String,
    /// Every option.
    pub options: Vec<Choice>,
    /// The filter typed so far.
    pub filter: String,
    /// The highlighted row among the filtered options.
    pub selected: usize,
}

impl Picker {
    /// The options whose label contains every word of the filter, ignoring
    /// case.
    pub fn filtered(&self) -> Vec<&Choice> {
        let filter = self.filter.to_lowercase();
        let words: Vec<&str> = filter.split_whitespace().collect();
        self.options
            .iter()
            .filter(|option| {
                let label = option.label.to_lowercase();
                words.iter().all(|word| label.contains(word))
            })
            .collect()
    }
}

/// Take the shown agent's notices and choices into the view, and pick the
/// first agent when none is shown.
pub(super) fn collect_messages(
    mut view: ResMut<TuiView>,
    agents: Query<Entity, With<Agent>>,
    mut notices: MessageReader<Notice>,
    mut choices: MessageReader<Choose>,
) {
    if view.agent.is_none_or(|agent| !agents.contains(agent)) {
        view.agent = agents.iter().next();
    }
    let shown = view.agent;
    for notice in notices.read().filter(|notice| Some(notice.agent) == shown) {
        view.notices.push(notice.text.clone());
        view.dirty = true;
    }
    let excess = view.notices.len().saturating_sub(KEPT_NOTICES);
    view.notices.drain(..excess);
    if let Some(choose) = choices
        .read()
        .filter(|choose| Some(choose.agent) == shown)
        .last()
    {
        view.picker = Some(Picker {
            title: choose.title.clone(),
            options: choose.options.clone(),
            filter: String::new(),
            selected: 0,
        });
        view.dirty = true;
    }
}
