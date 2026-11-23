//! Assistant-turn classification.

use rig_core::message::{AssistantContent, AssistantMessage, Message};

/// The assistant message carrying `choice` with `head`'s origin, stop and
/// provider message, or `None` when the choice is empty.
pub(crate) fn assistant_message(
    head: AssistantMessage,
    choice: Vec<AssistantContent>,
) -> Option<Message> {
    if choice.is_empty() {
        return None;
    }
    Some(Message::Assistant(head.with_content(choice)))
}

/// The assistant message for a generated turn, or `None` for an empty turn
/// ([`AssistantMessage::is_empty_turn`]), which must not enter provider history.
pub(crate) fn assistant_turn(
    head: AssistantMessage,
    choice: Vec<AssistantContent>,
) -> Option<Message> {
    let turn = head.with_content(choice);
    (!turn.is_empty_turn()).then_some(Message::Assistant(turn))
}
