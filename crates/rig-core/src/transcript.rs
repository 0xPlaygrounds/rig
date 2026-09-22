//! Conversation validation and constructors for real or synthetic tool results.
//!
//! ```
//! use rig_core::{message::Message, transcript::validate_canonical};
//!
//! validate_canonical(&[Message::user("Hello"), Message::assistant("Hi")])?;
//! # Ok::<(), rig_core::transcript::TranscriptError>(())
//! ```

use std::collections::BTreeSet;

use crate::message::{
    AssistantContent, Message, ProviderCallId, ToolCallId, ToolResultContent, UserContent,
};
use crate::tool::ToolOutput;

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
        call_id: ToolCallId,
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
        call_id: ToolCallId,
    },
}

/// Rejects consecutive assistant messages, unanswered tool-call IDs, and results
/// without a pending call. Each pending ID must be answered once in the next user
/// message, before another assistant message or the end of history.
/// System messages reset the consecutive-assistant check but retain pending calls.
/// Duplicate call IDs are treated as one pending ID.
pub fn validate_canonical(messages: &[Message]) -> Result<(), TranscriptError> {
    let mut prev_assistant_calls: Option<BTreeSet<ToolCallId>> = None;
    let mut prev_was_assistant = false;
    for (index, message) in messages.iter().enumerate() {
        match message {
            Message::Assistant { content, .. } => {
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
                let calls: BTreeSet<ToolCallId> = content
                    .iter()
                    .filter_map(|c| match c {
                        AssistantContent::ToolCall(call) => Some(call.id.clone()),
                        _ => None,
                    })
                    .collect();
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

fn tool_result_with(
    call: ToolCallId,
    provider: Option<ProviderCallId>,
    name: String,
    content: Vec<ToolResultContent>,
) -> UserContent {
    // Replay protocols require the executed tool's name separately from its call ID.
    UserContent::tool_result_for(call, provider, name, content)
}

/// Shape a canonical real tool output as a tool result without reparsing text.
pub fn tool_result_output(
    call: ToolCallId,
    provider: Option<ProviderCallId>,
    name: String,
    output: ToolOutput,
) -> UserContent {
    tool_result_with(call, provider, name, output.into_content())
}

/// Constructs a synthetic tool result containing verbatim text, such as recovery
/// feedback or a skip reason. JSON-shaped text is not reinterpreted as structured
/// or multimodal output.
pub fn tool_result_message(
    call: ToolCallId,
    provider: Option<ProviderCallId>,
    name: String,
    message: String,
) -> UserContent {
    tool_result_with(call, provider, name, vec![ToolResultContent::text(message)])
}

#[cfg(test)]
mod validator_tests;
