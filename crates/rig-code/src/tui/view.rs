//! The view's own state, kept apart from the agents' components.

use bevy_ecs::prelude::*;

use crate::core::agent::Conversation;
use crate::core::registry::{Notice, NoticeLevel, OpenPicker, PickerOption};

/// What the terminal view shows besides the agents' components.
#[derive(Resource, Default)]
pub struct View {
    /// The agent shown: the first agent, until it is gone.
    pub agent: Option<Entity>,
    /// The message being typed.
    pub composer: String,
    /// How many lines the transcript is scrolled up from the bottom.
    pub scroll: usize,
    /// The open picker.
    pub picker: Option<Picker>,
    /// Notices, each placed after the messages that existed when it came.
    pub notices: Vec<ViewNotice>,
}

/// A notice placed in an agent's transcript.
pub struct ViewNotice {
    /// The agent.
    pub agent: Entity,
    /// How many messages the conversation held when the notice came.
    pub after: usize,
    /// How it is shown.
    pub level: NoticeLevel,
    /// The text.
    pub text: String,
}

/// An open picker and what is typed into its filter.
pub struct Picker {
    /// The agent a choice runs the command for.
    pub agent: Entity,
    /// The title.
    pub title: String,
    /// The command a choice runs.
    pub command: String,
    /// Every choice.
    pub options: Vec<PickerOption>,
    /// The filter.
    pub filter: String,
    /// The highlighted row among the matches.
    pub selected: usize,
}

impl Picker {
    /// The choices whose label or detail contains the filter, ignoring case.
    pub fn matches(&self) -> Vec<&PickerOption> {
        let filter = self.filter.to_lowercase();
        self.options
            .iter()
            .filter(|option| {
                option.label.to_lowercase().contains(&filter)
                    || option.detail.to_lowercase().contains(&filter)
            })
            .collect()
    }
}

/// Places a notice in its agent's transcript.
pub(crate) fn on_notice(
    notice: On<Notice>,
    conversations: Query<&Conversation>,
    mut view: ResMut<View>,
) {
    let after = conversations
        .get(notice.entity)
        .map_or(0, |conversation| conversation.0.len());
    view.notices.push(ViewNotice {
        agent: notice.entity,
        after,
        level: notice.level,
        text: notice.text.clone(),
    });
    view.scroll = 0;
}

/// Opens a picker.
pub(crate) fn on_open_picker(open: On<OpenPicker>, mut view: ResMut<View>) {
    view.picker = Some(Picker {
        agent: open.entity,
        title: open.title.clone(),
        command: open.command.clone(),
        options: open.options.clone(),
        filter: String::new(),
        selected: 0,
    });
}
