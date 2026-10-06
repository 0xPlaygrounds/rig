//! Request history assembly, invalid-call feedback, and assistant-turn classification.
//!
//! ```
//! use rig_agent::run::transcript::build_history_for_request;
//! use rig_core::message::Message;
//! let history = build_history_for_request(None, &[Message::user("Hello")]);
//! assert_eq!(history.len(), 1);
//! ```

use rig_core::message::{AssistantContent, AssistantMessage, CallId, Message};
pub use rig_core::transcript::{
    TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER, TranscriptError, assistant_text_from_choice,
    is_empty_assistant_turn, tool_result_message, tool_result_output, validate_canonical,
};

/// Combine input history with new messages for building completion requests.
pub fn build_history_for_request(
    chat_history: Option<&[Message]>,
    new_messages: &[Message],
) -> Vec<Message> {
    let input = chat_history.unwrap_or(&[]);
    input.iter().chain(new_messages.iter()).cloned().collect()
}

/// Build the full history for error reporting (input + new messages).
pub fn build_full_history(
    chat_history: Option<&[Message]>,
    new_messages: Vec<Message>,
) -> Vec<Message> {
    let input = chat_history.unwrap_or(&[]);
    input.iter().cloned().chain(new_messages).collect()
}

/// Build tool-result feedback for an invalid call and skipped-peer notices for
/// other calls, in content order. Returns `None` when there are no tool calls.
pub fn invalid_tool_retry_user_message(
    assistant_content: &[AssistantContent],
    invalid_tool_call_id: &CallId,
    feedback: &str,
) -> Option<Message> {
    let content = rig_core::transcript::invalid_call_feedback(
        assistant_content,
        invalid_tool_call_id,
        feedback,
    );
    (!content.is_empty()).then_some(Message::User { content })
}

/// The assistant message carrying `choice` with `head`'s origin, stop and
/// provider message, or `None` when the choice is empty.
pub fn assistant_message(head: AssistantMessage, choice: Vec<AssistantContent>) -> Option<Message> {
    if choice.is_empty() {
        return None;
    }
    Some(Message::Assistant(head.with_content(choice)))
}

/// The assistant message for a generated turn, or `None` for an empty turn
/// ([`is_empty_assistant_turn`]), which must not enter provider history.
pub fn assistant_turn(head: AssistantMessage, choice: Vec<AssistantContent>) -> Option<Message> {
    if is_empty_assistant_turn(&choice) {
        return None;
    }
    assistant_message(head, choice)
}
