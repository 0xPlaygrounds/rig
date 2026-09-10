//! Loop-side transcript helpers: how rig-agent's run threads history for a
//! request, phrases the recovery feedback for an invalid tool call, and
//! classifies an assistant turn. The message-model invariants they build on
//! (`validate_canonical`, the tool-result constructors) are rig-core's and are
//! re-exported here so `crate::run::transcript` is the one path.

use rig_core::message::{AssistantContent, Message, ToolCallId, non_empty};
pub use rig_core::transcript::{
    TranscriptError, tool_result_message, tool_result_output, validate_canonical,
};

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

/// Whether a turn delivered **no answer**: no tool call, and no non-empty text
/// block.
///
/// Deliberately *not* [`is_empty_assistant_turn`], which answers a different
/// question — "does this turn belong in history". They diverge on the shapes
/// that are **worth recording yet answer nothing**, of which there are two:
///
/// 1. a turn carrying only [`AssistantContent::Reasoning`] — the reasoning is
///    real content worth replaying, but it is not an answer;
/// 2. a turn carrying only an **empty text block with `additional_params`** —
///    the annotation (citations, encrypted reasoning references, and other
///    provider metadata some wires require on replay) is worth recording, but
///    the caller still receives no text.
///
/// Metadata-only text therefore does **not** count as an answer. That follows
/// from what the caller actually gets: [`assistant_text_from_choice`]
/// concatenates `text.text` alone, so such a turn yields `""` — the annotation
/// is metadata *about* an answer, never the answer itself.
///
/// Reasoning is not an answer. It is the model's scratch work, it is often not
/// even replayable across turns, and a caller asked a question rather than for
/// the thinking. Treating it as output is how a thinking model that burned its
/// whole budget mid-thought used to report success with an empty string
/// (rig#2322): Gemini counts thinking tokens against `maxOutputTokens`, so a
/// truncated thinking turn *typically* carries reasoning and no text — the
/// common case, not a corner one.
///
/// Tool calls count as delivered: they are an answer in progress, and a
/// truncated tool-call turn must still route to execution. So do images —
/// ten providers emit assistant images, and an image *is* the answer for an
/// image-generation turn.
///
/// The match is **exhaustive on purpose**: no `_` arm. Every content variant
/// must be classified explicitly, so adding one to [`AssistantContent`] breaks
/// this build and forces a decision instead of silently inheriting a default.
/// The first version of this predicate had a `_ => false` catch-all and so
/// classified image-only turns as "no answer" — a truncated image-generation
/// turn would have errored despite delivering an image, which matters because
/// image tokens count against the same output budget.
pub fn turn_delivered_no_answer(choice: &[AssistantContent]) -> bool {
    !choice.iter().any(|content| match content {
        // Real text is an answer; an empty block delivers nothing.
        AssistantContent::Text(text) => !text.text.is_empty(),
        AssistantContent::ToolCall(_) => true,
        AssistantContent::Image(_) => true,
        // The one exclusion: scratch work, not an answer.
        AssistantContent::Reasoning(_) => false,
    })
}

pub fn assistant_text_from_choice(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}
