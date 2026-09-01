//! Canonical-transcript invariants of the message model: a valid conversation
//! answers every assistant tool call in the next user message and carries no
//! orphan results ([`validate_canonical`]), and a tool's output becomes that
//! user-message content through one constructor ([`tool_result_output`]).
//! How an agent loop *uses* these — history threading, recovery feedback,
//! turn classification — lives with the loop (`rig_agent::run::transcript`).

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
        call_id: String,
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
        call_id: String,
    },
}

/// Check that `messages` is a canonical transcript: no two assistant messages
/// in a row, every assistant tool call answered by a tool result in the very
/// next message, and no tool result that answers no call from the preceding
/// assistant message. This is the shape an agent loop produces (rig-agent's
/// run threads history this way and answers calls with
/// [`tool_result_message`]) and the shape it expects back when a driver
/// resumes a run or loads history from memory.
pub fn validate_canonical(messages: &[Message]) -> Result<(), TranscriptError> {
    let mut prev_assistant_calls: Option<BTreeSet<String>> = None;
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
                let calls: BTreeSet<String> = content
                    .iter()
                    .filter_map(|c| match c {
                        AssistantContent::ToolCall(call) => Some(call.id.to_string()),
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
                        let id = result.call.to_string();
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
    // The *executed* tool's name travels as data on the result: several
    // wires require it on replay (Gemini `functionResponse.name`, Ollama
    // tool messages), and an identifier is not a name.
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

/// Shape a **synthetic message** (a hook skip reason, recovery feedback, or a
/// "not executed" notice) as a tool result. Emitted **verbatim as text** and
/// never re-parsed as structured tool output, so a JSON-shaped message is not
/// silently reinterpreted as an image/multimodal result. Used identically by the
/// blocking and streaming drivers so synthetic results match across both.
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
