//! View state: what the terminal shows and edits, kept apart from the
//! agents' components and never saved.

use bevy::prelude::*;

use crate::ecs::{
    ChoiceKind, ChoiceRequested, Notice, NoticeLevel,
    agent::{Agent, Conversation, ModelChoice},
    catalog::{self, Providers},
};

/// The agent the terminal shows and sends to.
#[derive(Resource, Debug, Default)]
pub struct Focus(pub Option<Entity>);

/// The text being typed, and the cursor as a character index.
#[derive(Resource, Debug, Default)]
pub struct Composer {
    /// The text.
    pub text: String,
    /// The cursor, in characters from the start.
    pub cursor: usize,
}

impl Composer {
    /// Insert `text` at the cursor.
    pub fn insert(&mut self, text: &str) {
        let at = self.byte_index();
        self.text.insert_str(at, text);
        self.cursor += text.chars().count();
    }

    /// Delete the character before the cursor.
    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            let at = self.byte_index();
            self.text.remove(at);
        }
    }

    /// Delete the character under the cursor.
    pub fn delete(&mut self) {
        if self.cursor < self.text.chars().count() {
            let at = self.byte_index();
            self.text.remove(at);
        }
    }

    /// Move the cursor by `by` characters, within the text.
    pub fn move_by(&mut self, by: isize) {
        let length = self.text.chars().count();
        self.cursor = self.cursor.saturating_add_signed(by).min(length);
    }

    /// Move the cursor to the end.
    pub fn end(&mut self) {
        self.cursor = self.text.chars().count();
    }

    /// Take the text, leaving the composer empty.
    pub fn take(&mut self) -> String {
        self.cursor = 0;
        std::mem::take(&mut self.text)
    }

    fn byte_index(&self) -> usize {
        self.text
            .char_indices()
            .nth(self.cursor)
            .map_or(self.text.len(), |(index, _)| index)
    }
}

/// How many transcript lines the view is scrolled up from the bottom.
#[derive(Resource, Debug, Default)]
pub struct Scroll(pub usize);

/// An open picker: a filterable list whose choice is submitted as
/// `/<command> <value>`.
#[derive(Debug)]
pub struct PickerState {
    /// The command the choice answers.
    pub command: &'static str,
    /// The title.
    pub title: String,
    /// Each entry: what is shown, and the value submitted.
    pub items: Vec<(String, String)>,
    /// The filter typed so far.
    pub filter: String,
    /// The highlighted entry among the filtered ones.
    pub selected: usize,
}

impl PickerState {
    /// The entries matching every word of the filter, case-insensitively.
    pub fn filtered(&self) -> Vec<&(String, String)> {
        let filter = self.filter.to_lowercase();
        self.items
            .iter()
            .filter(|(label, _)| {
                let label = label.to_lowercase();
                filter.split_whitespace().all(|word| label.contains(word))
            })
            .collect()
    }
}

/// The open picker, if any.
#[derive(Resource, Debug, Default)]
pub struct Picker(pub Option<PickerState>);

/// One notice, placed after the message it followed.
#[derive(Debug)]
pub struct NoticeEntry {
    /// The conversation length when the notice arrived.
    pub after: usize,
    /// How it reads.
    pub level: NoticeLevel,
    /// The text.
    pub text: String,
}

/// The notices of the focused agent.
#[derive(Resource, Debug, Default)]
pub struct NoticeLog(pub Vec<NoticeEntry>);

/// Focus the first agent when nothing valid is focused.
pub(super) fn keep_focus(mut focus: ResMut<Focus>, agents: Query<Entity, With<Agent>>) {
    if focus.0.is_some_and(|agent| agents.contains(agent)) {
        return;
    }
    focus.0 = agents.iter().next();
}

/// Keep the focused agent's notices, and open a picker when a command asks
/// for a choice.
pub(super) fn read_core_messages(
    mut notices: MessageReader<Notice>,
    mut choices: MessageReader<ChoiceRequested>,
    focus: Res<Focus>,
    agents: Query<(&Conversation, &ModelChoice)>,
    providers: Res<Providers>,
    mut log: ResMut<NoticeLog>,
    mut picker: ResMut<Picker>,
    mut scroll: ResMut<Scroll>,
) {
    for notice in notices.read() {
        if focus.0 != Some(notice.agent) {
            continue;
        }
        let after = agents
            .get(notice.agent)
            .map_or(0, |(conversation, _)| conversation.0.len());
        log.0.push(NoticeEntry {
            after,
            level: notice.level,
            text: notice.text.clone(),
        });
        scroll.0 = 0;
    }
    for choice in choices.read() {
        if focus.0 != Some(choice.agent) {
            continue;
        }
        picker.0 = match choice.kind {
            ChoiceKind::Model => Some(PickerState {
                command: "model",
                title: "Model".to_owned(),
                items: providers
                    .models()
                    .map(|spec| {
                        let reference = catalog::reference(spec);
                        let keyless = if spec.provider.requires_credential() {
                            ""
                        } else {
                            "  (no key needed)"
                        };
                        (
                            format!("{}  {reference}{keyless}", spec.display_name),
                            reference,
                        )
                    })
                    .collect(),
                filter: String::new(),
                selected: 0,
            }),
            ChoiceKind::Effort => agents
                .get(choice.agent)
                .ok()
                .and_then(|(_, model)| model.0.as_deref())
                .and_then(catalog::resolve)
                .map(|spec| PickerState {
                    command: "effort",
                    title: format!("Effort for {}", spec.display_name),
                    items: catalog::effort_options(spec)
                        .iter()
                        .map(|option| (option.name().to_owned(), option.name().to_owned()))
                        .collect(),
                    filter: String::new(),
                    selected: 0,
                }),
        };
    }
}
