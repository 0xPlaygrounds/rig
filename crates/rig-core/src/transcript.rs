//! Conversation validation and repair, assistant-turn classification, and
//! constructors for real or synthetic tool results.
//!
//! ```
//! use rig_core::{message::Message, transcript::validate_canonical};
//!
//! validate_canonical(&[Message::user("Hello"), Message::assistant("Hi")])?;
//! # Ok::<(), rig_core::transcript::TranscriptError>(())
//! ```

use std::collections::BTreeSet;

use crate::message::{
    self, AssistantContent, CallId, Message, ToolName, ToolResultContent, UserContent,
};
use crate::tool::ToolResult;

/// Why a history is not a canonical transcript. See [`validate_canonical`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TranscriptError {
    /// Two assistant messages in a row (index of the second).
    #[error("consecutive assistant messages at index {index}")]
    ConsecutiveAssistant {
        /// Index of the offending (second) assistant message.
        index: usize,
    },
    /// An assistant tool call whose result is not in the next message.
    #[error("tool call `{call_id}` at index {index} has no result in the following message")]
    UnansweredToolCall {
        /// Index of the assistant message carrying the call.
        index: usize,
        /// The unanswered call id.
        call_id: CallId,
    },
    /// A tool result that answers no call from the immediately preceding
    /// assistant message.
    #[error(
        "tool result `{call_id}` at index {index} answers no call from the preceding assistant message"
    )]
    OrphanToolResult {
        /// Index of the user message carrying the result.
        index: usize,
        /// The orphan result's call id.
        call_id: CallId,
    },
}

/// Rejects consecutive assistant messages, unanswered tool-call IDs, and results
/// without a pending call. Each pending ID must be answered once in the next user
/// message, before another assistant message or the end of history.
/// System messages reset the consecutive-assistant check but retain pending calls.
/// Duplicate call IDs are treated as one pending ID.
pub fn validate_canonical(messages: &[Message]) -> Result<(), TranscriptError> {
    let mut prev_assistant_calls: Option<BTreeSet<CallId>> = None;
    let mut prev_was_assistant = false;
    for (index, message) in messages.iter().enumerate() {
        match message {
            Message::Assistant(turn) => {
                if prev_was_assistant {
                    return Err(TranscriptError::ConsecutiveAssistant { index });
                }
                if let Some(call_id) = prev_assistant_calls
                    .take()
                    .and_then(|pending| pending.into_iter().next())
                {
                    return Err(TranscriptError::UnansweredToolCall {
                        index: index - 1,
                        call_id,
                    });
                }
                let calls: BTreeSet<CallId> =
                    turn.tool_calls().map(|call| call.id.clone()).collect();
                prev_assistant_calls = (!calls.is_empty()).then_some(calls);
                prev_was_assistant = true;
            }
            Message::User { content } => {
                let mut pending = prev_assistant_calls.take().unwrap_or_default();
                for item in content.iter() {
                    if let UserContent::ToolResult(result) = item {
                        let id = result.call.clone();
                        if !pending.remove(&id) {
                            return Err(TranscriptError::OrphanToolResult { index, call_id: id });
                        }
                    }
                }
                if let Some(call_id) = pending.into_iter().next() {
                    return Err(TranscriptError::UnansweredToolCall {
                        index: index.saturating_sub(1),
                        call_id,
                    });
                }
                prev_was_assistant = false;
            }
            Message::System { .. } => {
                prev_was_assistant = false;
            }
        }
    }
    if let Some(call_id) = prev_assistant_calls.and_then(|pending| pending.into_iter().next()) {
        return Err(TranscriptError::UnansweredToolCall {
            index: messages.len().saturating_sub(1),
            call_id,
        });
    }
    Ok(())
}

/// Shape a tool's result as the tool result the model reads, without
/// reparsing text. Anything but a success is an error result.
pub fn tool_result_output(call: CallId, name: ToolName, result: &ToolResult) -> UserContent {
    UserContent::ToolResult(message::ToolResult {
        call,
        name,
        content: result.output().clone().into_content(),
        is_error: !result.is_success(),
    })
}

/// Constructs the error result of a call that never ran, its text verbatim,
/// such as recovery feedback or a skip reason. JSON-shaped text is not
/// reinterpreted as structured or multimodal output.
pub fn tool_result_message(call: CallId, name: ToolName, message: String) -> UserContent {
    UserContent::ToolResult(message::ToolResult {
        call,
        name,
        content: vec![ToolResultContent::text(message)],
        is_error: true,
    })
}

/// What the model reads when its call to `tool` sent `raw`, arguments that
/// are not a JSON object: the tool never ran, and the model calls it again.
pub fn invalid_arguments_feedback(tool: &str, raw: &str) -> String {
    format!(
        "The arguments for tool `{tool}` are not a JSON object: {raw}. \
         Call the tool again with a JSON object as its arguments."
    )
}

/// The result every other call of a turn gets when one call was retried or
/// skipped: none of the turn's calls ran.
pub const TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER: &str =
    "Tool not executed because another tool call in the same assistant turn was invalid.";

/// The tool results answering a turn with an invalid call, in call order:
/// `feedback` for the call `invalid`, [`TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER`]
/// for every other call. Empty when `content` has no tool calls.
pub fn invalid_call_feedback(
    content: &[AssistantContent],
    invalid: &CallId,
    feedback: &str,
) -> Vec<UserContent> {
    content
        .iter()
        .filter_map(|part| match part {
            AssistantContent::ToolCall(call) => Some(tool_result_message(
                call.id.clone(),
                call.function.name.clone(),
                if &call.id == invalid {
                    feedback
                } else {
                    TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER
                }
                .to_owned(),
            )),
            _ => None,
        })
        .collect()
}

/// Whether a generated assistant turn is empty: no parts, or exactly one
/// empty text part with no provider item. An empty turn must not enter
/// history.
pub fn is_empty_assistant_turn(content: &[AssistantContent]) -> bool {
    match content {
        [] => true,
        [AssistantContent::Text(text)] => text.text.is_empty() && text.native.is_none(),
        _ => false,
    }
}

/// The text parts of an assistant turn, concatenated.
pub fn assistant_text_from_choice(content: &[AssistantContent]) -> String {
    content
        .iter()
        .filter_map(|part| match part {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod validator_tests;
