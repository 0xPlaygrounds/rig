//! The protocol's outputs: per-call accounting and the final response.

use rig_core::completion::{FinishReason, ResponseIdentity, Usage};
use rig_core::message::{AssistantContent, Message};
use serde::{Deserialize, Serialize};

// No longer `Copy`: the identity fields carry owned strings. No longer `Eq`:
// `raw` is a `serde_json::Value`, which is `PartialEq` but not `Eq` (floats).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionCall {
    /// Zero-based index of the completion request within this agent run.
    pub call_index: usize,
    /// Token usage reported for this completion request.
    ///
    /// Zero-valued usage is [`Usage`]'s documented sentinel for missing
    /// provider usage metrics; rig does not distinguish "reported all zeros"
    /// from "unreported".
    #[serde(default, deserialize_with = "usage_null_as_default")]
    pub usage: Usage,
    /// Provider-assigned assistant message ID for this call, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// Provider-assigned response-scoped ID for this call, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// The provider's transport request id for this call (HTTP response
    /// header, e.g. Anthropic `request-id`) — the id provider support asks
    /// for. `None` means the provider did not report one, never an error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// Why the model stopped generating on this call, when the provider
    /// reported it. `None` means the provider reported no reason.
    ///
    /// Recorded **per call** rather than once per run: a multi-turn run makes N
    /// completion requests, each with its own terminal reason, and collapsing
    /// them to a single run-level value would lose exactly the information that
    /// makes a truncated turn diagnosable — which turn hit the limit. A caller
    /// that wants the run's last reason reads it off the final entry.
    ///
    /// This is the field whose absence hid rig#2322: the provider layer carried
    /// [`FinishReason::Length`] on the stream's terminal record, but the agent
    /// assembler dropped it, so a turn truncated at the output-token limit was
    /// indistinguishable from a turn that simply had nothing to say.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    /// The provider's own response for this call — see
    /// `CompletionResponse::raw` for the exact meaning of the payload. Every
    /// provider seam populates it; `Value::Null` only when the call's response
    /// was built without a provider behind it (a hand-constructed model, a
    /// record persisted before the field, or a hand-driven `AgentRun` that
    /// recorded a streamed call with no terminal record — the runner itself
    /// rejects such a stream as truncated before recording anything).
    ///
    /// Recorded **per call**, like [`Self::finish_reason`]: on a multi-turn
    /// run each entry carries its own attempt's response, never a previous
    /// attempt's, and on a retried turn the recorded call carries the retried
    /// attempt's own.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub raw: serde_json::Value,
}

impl CompletionCall {
    /// Create details for one completion request in an agent run; identity
    /// metadata starts unset and is attached with [`Self::with_identity`].
    pub fn new(call_index: usize, usage: Usage) -> Self {
        Self {
            call_index,
            usage,
            message_id: None,
            response_id: None,
            provider_request_id: None,
            finish_reason: None,
            raw: serde_json::Value::Null,
        }
    }

    /// Attach the provider's own response this call's attempt produced.
    pub fn with_raw(mut self, raw: serde_json::Value) -> Self {
        self.raw = raw;
        self
    }

    /// Attach the response identity metadata this call's attempt reported.
    pub fn with_identity(mut self, identity: ResponseIdentity) -> Self {
        self.message_id = identity.message_id;
        self.response_id = identity.response_id;
        self.provider_request_id = identity.provider_request_id;
        self
    }

    /// Attach the terminal finish reason this call's attempt reported.
    ///
    /// Kept separate from [`Self::with_identity`] because a finish reason is
    /// not identity: [`ResponseIdentity`] answers "which response was this",
    /// while this answers "why did it stop".
    pub fn with_finish_reason(mut self, finish_reason: Option<FinishReason>) -> Self {
        self.finish_reason = finish_reason;
        self
    }

    /// This call's identity metadata as one [`ResponseIdentity`] carrier.
    pub fn identity(&self) -> ResponseIdentity {
        ResponseIdentity {
            message_id: self.message_id.clone(),
            response_id: self.response_id.clone(),
            provider_request_id: self.provider_request_id.clone(),
        }
    }
}

/// Tolerate `null` usage from data serialized before rig dropped the
/// `Option<Usage>` encoding of missing provider usage metrics.
///
/// This tolerance requires a self-describing format such as JSON; data
/// serialized with non-self-describing formats (e.g. bincode) from before the
/// change cannot round-trip.
fn usage_null_as_default<'de, D>(deserializer: D) -> Result<Usage, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<Usage>::deserialize(deserializer)?.unwrap_or_default())
}

/// The result of an agent run, returned by **both** the blocking
/// (`PromptRequest`) and streaming (`StreamingPromptRequest`) surfaces so a
/// call site reads identically whether it used `.prompt()` or `.stream_prompt()`.
///
/// On the streaming surface this is the payload of the terminal
/// `MultiTurnStreamItem::FinalResponse` item.
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptResponse {
    /// Concatenated assistant text for the final turn.
    pub output: String,
    /// Aggregated token usage across the whole run.
    pub usage: Usage,
    /// Successfully completed completion requests made by this agent run.
    ///
    /// `usage` remains the aggregate across the whole run. Use the last
    /// entry's usage to inspect the final completion request's prompt/context
    /// length. Zero-valued entry usage means the provider reported no usage
    /// metrics for that request.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub completion_calls: Vec<CompletionCall>,
    /// Accumulated message history for the run (the run's persisted transcript),
    /// unless memory/history bookkeeping was disabled for the request.
    pub messages: Option<Vec<Message>>,
    /// Structured assistant content for the final turn.
    ///
    /// Where [`output`](Self::output) is the concatenated text, this preserves
    /// the individual content parts (text, reasoning, images, …).
    pub content: Vec<AssistantContent>,
    /// Number of synthetic output-tool calls in the turn that finalized this
    /// response. Kept crate-private because it is runner bookkeeping rather
    /// than provider-facing response content.
    #[serde(skip)]
    output_tool_calls: usize,
}

impl std::fmt::Display for PromptResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.output.fmt(f)
    }
}

impl PromptResponse {
    pub fn new(output: impl Into<String>, usage: Usage) -> Self {
        let output = output.into();
        Self {
            content: vec![AssistantContent::text(output.clone())],
            output,
            usage,
            completion_calls: Vec::new(),
            messages: None,
            output_tool_calls: 0,
        }
    }

    /// An empty run result (empty output, zero usage, no history).
    pub fn empty() -> Self {
        Self::new(String::new(), Usage::new())
    }

    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = Some(messages);
        self
    }

    /// Attach completion call details to this response.
    pub fn with_completion_calls(mut self, completion_calls: Vec<CompletionCall>) -> Self {
        self.completion_calls = completion_calls;
        self
    }

    /// Set the structured assistant content for the final turn.
    pub fn with_content(mut self, content: Vec<AssistantContent>) -> Self {
        self.content = content;
        self
    }

    pub fn with_output_tool_calls(mut self, count: usize) -> Self {
        self.output_tool_calls = count;
        self
    }

    pub fn output_tool_calls(&self) -> usize {
        self.output_tool_calls
    }

    /// The concatenated assistant text for the final turn.
    pub fn output(&self) -> &str {
        &self.output
    }

    /// Aggregated token usage across the whole run.
    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// The run's accumulated message history, if tracked.
    pub fn messages(&self) -> Option<&[Message]> {
        self.messages.as_deref()
    }

    /// The structured assistant content for the final turn.
    pub fn content(&self) -> &[AssistantContent] {
        &self.content
    }

    /// Returns successfully completed completion requests made by this agent run.
    ///
    /// Zero-valued entry usage means the provider reported no usage metrics
    /// for that request.
    pub fn completion_calls(&self) -> &[CompletionCall] {
        &self.completion_calls
    }

    /// Number of completion requests this agent run made.
    pub fn requests(&self) -> usize {
        self.completion_calls.len()
    }
}
