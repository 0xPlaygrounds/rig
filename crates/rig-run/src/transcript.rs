//! Canonical-transcript helpers: how a turn's messages are threaded so that an
//! assistant tool call is always paired with its results in the next user
//! message, and how the recovery feedback for an invalid call is phrased.

use std::collections::BTreeSet;

use rig_core::message::{
    AssistantContent, Message, ProviderCallId, ToolCallId, ToolResultContent, UserContent,
    non_empty,
};
use rig_core::tool::ToolOutput;

use crate::run::AgentRun;

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
/// assistant message. This is the shape the protocol produces
/// ([`build_full_history`], [`tool_result_message`]) and the shape it expects
/// back when a driver resumes a run or loads history from memory.
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

impl AgentRun {
    /// [`with_history`](AgentRun::with_history), rejecting a history that is
    /// not a canonical transcript. Use this when the history comes from
    /// outside the protocol (a memory backend, a resumed run from another
    /// process); `with_history` stays unchecked for callers that built the
    /// history themselves.
    pub fn with_validated_history(self, history: Vec<Message>) -> Result<Self, TranscriptError> {
        validate_canonical(&history)?;
        Ok(self.with_history(history))
    }
}

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

/// Wrap already-shaped tool-result content for the model (see
/// [`tool_result_output`] / [`tool_result_message`]).
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

#[cfg(test)]
mod validator_tests {
    use super::*;
    use rig_core::message::{ToolCall, ToolFunction, ToolResult};

    fn call(id: &str) -> AssistantContent {
        AssistantContent::ToolCall(ToolCall {
            id: ToolCallId::new_or_mint(id),
            provider: None,
            function: ToolFunction {
                name: "add".into(),
                arguments: serde_json::json!({}),
            },
            additional_params: None,
            signature: None,
        })
    }
    fn result(id: &str) -> UserContent {
        UserContent::ToolResult(ToolResult {
            call: ToolCallId::new_or_mint(id),
            provider: None,
            name: "add".into(),
            content: vec![ToolResultContent::text("3")],
        })
    }
    fn assistant(content: Vec<AssistantContent>) -> Message {
        Message::Assistant { id: None, content }
    }

    #[test]
    fn canonical_transcripts_pass() {
        let history = vec![
            Message::user("hi"),
            assistant(vec![call("c1")]),
            Message::User {
                content: vec![result("c1")],
            },
            assistant(vec![AssistantContent::text("done")]),
            Message::user("thanks"),
        ];
        assert_eq!(validate_canonical(&history), Ok(()));
        assert!(validate_canonical(&[]).is_ok());
    }

    #[test]
    fn consecutive_assistant_is_rejected() {
        let history = vec![
            assistant(vec![AssistantContent::text("a")]),
            assistant(vec![AssistantContent::text("b")]),
        ];
        assert_eq!(
            validate_canonical(&history),
            Err(TranscriptError::ConsecutiveAssistant { index: 1 })
        );
    }

    #[test]
    fn unanswered_and_orphan_results_are_rejected() {
        let unanswered = vec![assistant(vec![call("c1")]), Message::user("no result")];
        assert!(matches!(
            validate_canonical(&unanswered),
            Err(TranscriptError::UnansweredToolCall { .. })
        ));
        let orphan = vec![
            Message::user("hi"),
            Message::User {
                content: vec![result("ghost")],
            },
        ];
        assert!(matches!(
            validate_canonical(&orphan),
            Err(TranscriptError::OrphanToolResult { .. })
        ));
        let trailing = vec![assistant(vec![call("c1")])];
        assert!(matches!(
            validate_canonical(&trailing),
            Err(TranscriptError::UnansweredToolCall { .. })
        ));
    }

    #[test]
    fn with_validated_history_gates_construction() {
        let bad = vec![
            assistant(vec![AssistantContent::text("a")]),
            assistant(vec![AssistantContent::text("b")]),
        ];
        assert!(AgentRun::new("x").with_validated_history(bad).is_err());
        assert!(
            AgentRun::new("x")
                .with_validated_history(vec![Message::user("ok")])
                .is_ok()
        );
    }
}
