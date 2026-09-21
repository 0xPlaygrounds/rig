//! Request history assembly, invalid-call feedback, and assistant-turn classification.
//!
//! ```
//! use rig_agent::run::transcript::build_history_for_request;
//! use rig_core::message::Message;
//! let history = build_history_for_request(None, &[Message::user("Hello")]);
//! assert_eq!(history.len(), 1);
//! ```

use rig_core::message::{AssistantContent, Message, ToolCallId, non_empty};
pub use rig_core::transcript::{
    TranscriptError, tool_result_message, tool_result_output, validate_canonical,
};

/// The result text for a valid call skipped because a sibling call in the
/// same assistant turn was invalid.
pub const TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER: &str =
    "Tool not executed because another tool call in the same assistant turn was invalid.";

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
    invalid_tool_call_id: &ToolCallId,
    feedback: &str,
) -> Option<Message> {
    // Call IDs distinguish peers even when the provider supplies no wire IDs.
    let retry_results = assistant_content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(tool_call) if tool_call.id == *invalid_tool_call_id => {
                Some(tool_result_message(
                    tool_call.id.clone(),
                    tool_call.provider.clone(),
                    tool_call.function.name.clone(),
                    feedback.to_string(),
                ))
            }
            AssistantContent::ToolCall(tool_call) => Some(tool_result_message(
                tool_call.id.clone(),
                tool_call.provider.clone(),
                tool_call.function.name.clone(),
                TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER.to_string(),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();

    Some(Message::User {
        content: non_empty(retry_results)?,
    })
}

/// Return true for zero parts or exactly one empty, unannotated text part.
/// Annotations carry data even without text. The run uses this classification
/// for generated turns, not to filter caller-supplied history.
pub fn is_empty_assistant_turn(choice: &[AssistantContent]) -> bool {
    if choice.is_empty() {
        return true;
    }

    choice.len() == 1
        && matches!(
            choice.first(),
            Some(AssistantContent::Text(text))
                if text.text.is_empty() && text.additional_params.is_none()
        )
}

/// The concatenated text of a choice.
pub fn assistant_text_from_choice(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}
