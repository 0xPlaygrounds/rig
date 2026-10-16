//! Invalid-call feedback and assistant-turn classification.

use rig_core::message::{AssistantContent, AssistantMessage, CallId, Message};
use rig_core::tool::feedback::invalid_call_feedback;

/// Build tool-result feedback for an invalid call and skipped-peer notices for
/// other calls, in content order. Returns `None` when there are no tool calls.
pub(crate) fn invalid_tool_retry_user_message(
    assistant_content: &[AssistantContent],
    invalid_tool_call_id: &CallId,
    feedback: &str,
) -> Option<Message> {
    let content = invalid_call_feedback(assistant_content, invalid_tool_call_id, feedback);
    (!content.is_empty()).then_some(Message::User { content })
}

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

/// The text parts of a choice, concatenated, as [`AssistantMessage::text`]
/// reads a turn.
pub(crate) fn choice_text(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|part| match part {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}
