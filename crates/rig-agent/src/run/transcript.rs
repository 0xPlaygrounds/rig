//! Loop-side transcript helpers: how rig-agent's run threads history for a
//! request, phrases the recovery feedback for an invalid tool call, and
//! classifies an assistant turn. The message-model invariants they build on
//! (`validate_canonical`, the tool-result constructors) are rig-core's and are
//! re-exported here so `crate::run::transcript` is the one path.

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

/// The user message that asks the model to retry after an invalid tool
/// call, naming what was wrong.
pub fn invalid_tool_retry_user_message(
    assistant_content: &[AssistantContent],
    invalid_tool_call_id: &ToolCallId,
    feedback: &str,
) -> Option<Message> {
    // Selecting the invalid call by id is correct by construction:
    // `ToolCallId` is unique and non-empty (minted at the provider boundary
    // when the wire issued none), so id-less wires can no longer collapse
    // every peer onto the first match arm.
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

/// Whether an assistant turn carried nothing the caller should see.
///
/// Two shapes mean the same thing, and both must be recognised:
///
/// - **Zero parts.** A turn that produced no text and no tool call is an
///   empty list — the shape the streaming path produces (its assembler
///   filters empty text deltas out of the canonical order).
/// - **One empty, unannotated text block.** A blocking wire can deliver an
///   assistant message whose only part is an empty text block; it carries
///   nothing, and the agent curates it out of history exactly as it curates
///   a zero-part turn. The annotation guard is load-bearing: an *annotated*
///   empty text block carries data and must not read as empty. Annotation is
///   a plain `is_some()`: [`rig_core::message::AdditionalParams`] is
///   non-empty by construction, so `Some` always carries data, live and
///   restored alike (pinned by
///   `empty_turn_classification_survives_a_serde_round_trip`).
///
/// This runs on turns flowing through the agent loop only. Caller-supplied
/// `chat_history` is never filtered: an empty text block you replay goes to
/// the wire as-is.
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
