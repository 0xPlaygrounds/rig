//! View state, kept apart from the agent components: the input line, the
//! scroll position, the open picker and the notices on screen.

use bevy::prelude::*;

use crate::core::{Choice, Notice, NoticeLevel, OfferChoices};

/// Notices kept on screen.
const KEPT_NOTICES: usize = 20;

/// What the terminal view shows besides the agent itself.
#[derive(Resource, Default)]
pub(crate) struct TuiView {
    /// The line being typed.
    pub(super) input: String,
    /// Rows scrolled up from the bottom of the conversation.
    pub(super) scroll: usize,
    /// The open picker, if any.
    pub(super) picker: Option<Picker>,
    /// Notices since the last submitted line.
    pub(super) notices: Vec<(NoticeLevel, String)>,
}

/// A filterable list of choices the user picks one of.
pub(super) struct Picker {
    /// The agent the pick is for.
    pub(super) agent: Entity,
    /// The title.
    pub(super) title: String,
    /// The command the pick is sent to.
    pub(super) command: String,
    /// Every choice.
    pub(super) choices: Vec<Choice>,
    /// What was typed to filter the choices.
    pub(super) filter: String,
    /// Index into the filtered choices.
    pub(super) selected: usize,
}

impl Picker {
    /// The choices whose label contains every word of the filter, ignoring
    /// case.
    pub(super) fn visible(&self) -> Vec<&Choice> {
        let filter = self.filter.to_lowercase();
        self.choices
            .iter()
            .filter(|choice| {
                let label = choice.label.to_lowercase();
                filter.split_whitespace().all(|word| label.contains(word))
            })
            .collect()
    }
}

/// Shows notices and opens pickers offered by the core or plugins.
pub(super) fn collect_messages(
    mut view: ResMut<TuiView>,
    mut notices: MessageReader<Notice>,
    mut offers: MessageReader<OfferChoices>,
) {
    for notice in notices.read() {
        view.notices.push((notice.level, notice.text.clone()));
    }
    let excess = view.notices.len().saturating_sub(KEPT_NOTICES);
    if excess > 0 {
        view.notices.drain(..excess);
    }
    if let Some(offer) = offers.read().last() {
        view.picker = Some(Picker {
            agent: offer.agent,
            title: offer.title.clone(),
            command: offer.command.clone(),
            choices: offer.choices.clone(),
            filter: String::new(),
            selected: 0,
        });
    }
}
