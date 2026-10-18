//! Rewind in the terminal view: Enter on a message of the user's selected
//! in the transcript (Alt+Up) runs `/fork` through it for the shown agent,
//! whose agent half shows the fork, and puts the message back in the
//! input.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_ecs::agent::{Conversation, Notice};
use rig_harness::commands::RunCommand;
use rig_harness::prelude::typed;
use rig_tui::crossterm::event::KeyCode;
use rig_tui::{
    AppKeyBindingsExt, KeyBinding, KeyPress, KeyPressed, KeyScope, MainInput, MainTranscript,
    TranscriptItem,
};

/// The rewind key, in the view.
pub(crate) fn add(app: &mut App) {
    let enter = KeyBinding::new(KeyPress::key(KeyCode::Enter), "Rewind to this message")
        .in_scope(KeyScope::Transcript);
    app.add_key_binding(enter, rewind);
}

/// Forks the shown agent through the selected message, a user message
/// that asked for a turn and so is at no turn boundary: the fork ends just
/// before it, and its text is back in the input.
fn rewind(
    pressed: On<KeyPressed>,
    mut view: MainTranscript,
    conversations: Query<&Conversation>,
    input: Option<MainInput>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Some(agent) = pressed.agent else {
        return;
    };
    let selected = match &view.selected {
        Some(TranscriptItem::Message(id)) => Some(*id),
        _ => None,
    };
    let conversation = conversations.get(agent).ok();
    let text = selected
        .zip(conversation)
        .and_then(|(id, conversation)| typed(conversation, conversation.index_of(id)?));
    let (Some(id), Some(text)) = (selected, text) else {
        let why = "Select a message of yours with Alt+Up; Enter goes back to before it.";
        notices.write(Notice::info(agent, why));
        return;
    };
    view.selected = None;
    commands.trigger(RunCommand::new(agent, format!("fork {id}")));
    if let Some(mut input) = input {
        input.set(text);
    }
}
