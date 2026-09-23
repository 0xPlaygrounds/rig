//! Per-call accounting, final run responses, and prompting errors.
//!
//! ```
//! use rig_agent::run::response::PromptResponse;
//! let response = PromptResponse::empty();
//! assert!(response.output().is_empty());
//! ```

use rig_core::completion::{FinishReason, ResponseIdentity, Usage};
use rig_core::error::ProviderError;
use rig_core::message::{AssistantContent, Message};
use serde::{Deserialize, Serialize};

/// One completion call of a run: what was asked and what came back.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionCall {
    /// Zero-based index of the completion request within this agent run.
    pub call_index: usize,
    /// Token usage reported for this completion request.
    ///
    /// A counter the provider did not report is `None`; a reported zero is
    /// `Some(0)`. Every counter is `None` when no usage was reported at all.
    pub usage: Usage,
    /// Provider-assigned assistant message ID for this call, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// Provider-assigned response-scoped ID for this call, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Transport request ID for this call, or `None` if the provider reported none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// Why this call stopped generating, or `None` if unreported.
    /// Retained per call so callers can identify which attempt was truncated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    /// This attempt's provider response payload, as defined by
    /// [`rig_core::completion::CompletionResponse::raw`]. A stream without a
    /// terminal response produces no completion-call entry.
    pub raw: serde_json::Value,
}

impl CompletionCall {
    /// Create details for one completion request in an agent run, carrying
    /// the provider's own response `raw` (see [`Self::raw`]); identity
    /// metadata starts unset and is attached with [`Self::with_identity`].
    pub fn new(call_index: usize, usage: Usage, raw: serde_json::Value) -> Self {
        Self {
            call_index,
            usage,
            message_id: None,
            response_id: None,
            provider_request_id: None,
            finish_reason: None,
            raw,
        }
    }

    /// Attach the response identity metadata this call's attempt reported.
    pub fn with_identity(mut self, identity: ResponseIdentity) -> Self {
        self.message_id = identity.message_id;
        self.response_id = identity.response_id;
        self.provider_request_id = identity.provider_request_id;
        self
    }

    /// Attach the terminal finish reason reported by this attempt.
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

/// Final run output and accounting, returned by awaited runs and terminal
/// `MultiTurnStreamItem::FinalResponse` stream items.
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
    /// length. An entry whose counters are all `None` means the provider
    /// reported no usage metrics for that request.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub completion_calls: Vec<CompletionCall>,
    /// The run's transcript: the prompt, every accepted assistant turn, every
    /// committed tool result and the corrective feedback of any retried turn,
    /// excluding the input history the run started from. This is what a
    /// configured conversation memory is asked to persist; whether that
    /// append was acknowledged is [`memory_append`](Self::memory_append).
    /// `None` only for a response built without a run behind it.
    pub messages: Option<Vec<Message>>,
    /// How the run's conversation-memory append settled, when the run had a
    /// memory backend and conversation to append to; `None` when memory was
    /// not configured for the run, bypassed by explicit history, disabled, or
    /// the run was resumed (the driver that persisted it owns the append).
    /// Set by the driver after the append dispatch resolved, so the
    /// protocol's own `Done` response never carries it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_append: Option<MemoryAppend>,
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

/// Outcome of the memory append, independent of the accepted run answer.
/// Acknowledgement provides only the backend's durability guarantee; failure
/// does not prove no write occurred. Dropping an in-flight append yields no
/// response, and this outcome does not guarantee exactly-once persistence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum MemoryAppend {
    /// The backend acknowledged the append of [`PromptResponse::messages`].
    Acknowledged,
    /// The append dispatch failed or was refused, as reported after outcome hooks.
    /// Persistence may still have occurred. The effect log retains the handler's
    /// original answer rather than any hook replacement.
    Failed {
        /// Why the append failed.
        report: rig_core::error::ErrorReport,
    },
}

impl MemoryAppend {
    /// Whether the backend acknowledged the append.
    pub fn is_acknowledged(&self) -> bool {
        matches!(self, Self::Acknowledged)
    }

    /// The failure report, when the append failed.
    pub fn failure(&self) -> Option<&rig_core::error::ErrorReport> {
        match self {
            Self::Acknowledged => None,
            Self::Failed { report } => Some(report),
        }
    }
}

impl std::fmt::Display for PromptResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.output.fmt(f)
    }
}

impl PromptResponse {
    /// A response whose final text is `output`, with the run's `usage`.
    pub fn new(output: impl Into<String>, usage: Usage) -> Self {
        let output = output.into();
        Self {
            content: vec![AssistantContent::text(output.clone())],
            output,
            usage,
            completion_calls: Vec::new(),
            messages: None,
            memory_append: None,
            output_tool_calls: 0,
        }
    }

    /// An empty run result (empty output, zero usage, no history).
    pub fn empty() -> Self {
        Self::new(String::new(), Usage::default())
    }

    /// Attach the run's accumulated message history.
    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = Some(messages);
        self
    }

    /// Attach completion call details to this response.
    pub fn with_completion_calls(mut self, completion_calls: Vec<CompletionCall>) -> Self {
        self.completion_calls = completion_calls;
        self
    }

    /// Record how the run's conversation-memory append settled; the driver
    /// sets this once the append dispatch resolved.
    pub fn with_memory_append(mut self, memory_append: Option<MemoryAppend>) -> Self {
        self.memory_append = memory_append;
        self
    }

    /// Set the structured assistant content for the final turn.
    pub fn with_content(mut self, content: Vec<AssistantContent>) -> Self {
        self.content = content;
        self
    }

    /// Record how many times the output tool was called.
    pub fn with_output_tool_calls(mut self, count: usize) -> Self {
        self.output_tool_calls = count;
        self
    }

    /// How many times the output tool was called.
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

    /// How the run's conversation-memory append settled, when the run had
    /// one (see the [field](Self::memory_append)).
    pub fn memory_append(&self) -> Option<&MemoryAppend> {
        self.memory_append.as_ref()
    }

    /// The structured assistant content for the final turn.
    pub fn content(&self) -> &[AssistantContent] {
        &self.content
    }

    /// Returns successfully completed completion requests made by this agent run.
    ///
    /// An entry whose counters are all `None` means the provider reported no
    /// usage metrics for that request.
    pub fn completion_calls(&self) -> &[CompletionCall] {
        &self.completion_calls
    }

    /// Number of completion requests this agent run made.
    pub fn requests(&self) -> usize {
        self.completion_calls.len()
    }
}

use thiserror::Error;

use rig_core::memory::MemoryError;

/// Errors from classic agent prompting.
#[derive(Debug, Error)]
pub enum PromptError {
    /// A provider completion failed.
    #[error("CompletionError: {0}")]
    CompletionError(#[from] ProviderError),

    /// Structured effect failure from the bus, a handler, a hook, or a stream item.
    #[error("{0}")]
    Report(#[from] rig_core::error::ErrorReport),

    /// Conversation memory failed to load or persist history.
    #[error("MemoryError: {0}")]
    MemoryError(#[from] MemoryError),

    /// The run exhausted its total model-call budget.
    #[error("MaxTurnsError: reached max turns limit: {max_turns}")]
    MaxTurnsError {
        /// Configured total model-call budget.
        max_turns: usize,
        /// Canonical history available when the budget was exhausted.
        chat_history: Vec<Message>,
        /// Prompt for the call that could not be dispatched.
        prompt: Message,
    },

    /// A prompting loop was cancelled.
    #[error("PromptCancelled: {reason}")]
    PromptCancelled {
        /// Canonical history available at cancellation.
        chat_history: Vec<Message>,
        /// Human-readable cancellation reason.
        reason: String,
    },

    /// The model attempted to call a tool unavailable for the current turn.
    #[error(
        "UnknownToolCall: model attempted to call unknown or disallowed tool `{tool_name}`. Available tools: {available_tools:?}. Allowed tools for this turn: {allowed_tools:?}"
    )]
    UnknownToolCall {
        /// Tool name emitted by the model.
        tool_name: String,
        /// Tools registered on the runtime.
        available_tools: Vec<String>,
        /// Exact immutable set allowed for this turn.
        allowed_tools: Vec<String>,
        /// Canonical history available at failure.
        chat_history: Vec<Message>,
    },
}

/// Forward provider response accessors through wrapped errors and optional reports.
macro_rules! forward_provider_response_helpers {
    ($err:ident, $variant:ident, $inner:literal $(, report = $report:ident)?) => {
        impl $err {
            #[doc = concat!("Returns the provider response body exposed by a wrapped ", $inner, ".")]
            pub fn provider_response_body(&self) -> Option<&str> {
                match self {
                    Self::$variant(error) => error.provider_response_body(),
                    $(Self::$report(report) => report.provider_response_body(),)?
                    _ => None,
                }
            }

            #[doc = concat!("Parses the provider response body of a wrapped ", $inner, " as JSON when present.")]
            pub fn provider_response_json(
                &self,
            ) -> Result<Option<serde_json::Value>, serde_json::Error> {
                match self {
                    Self::$variant(error) => error.provider_response_json(),
                    $(Self::$report(report) => report.provider_response_json(),)?
                    _ => Ok(None),
                }
            }

            #[doc = concat!("Returns the provider transport request id exposed by a wrapped ", $inner, ", or carried by a wire report.")]
            pub fn provider_request_id(&self) -> Option<&str> {
                match self {
                    Self::$variant(error) => error.provider_request_id(),
                    $(Self::$report(report) => report.request_id.as_deref(),)?
                    _ => None,
                }
            }

            #[doc = concat!("Returns the HTTP status exposed by a wrapped ", $inner, ", or carried by a wire report.")]
            pub fn provider_response_status(&self) -> Option<http::StatusCode> {
                match self {
                    Self::$variant(error) => error.provider_response_status(),
                    $(Self::$report(report) => report
                        .http_status
                        .and_then(|status| http::StatusCode::from_u16(status).ok()),)?
                    _ => None,
                }
            }

            #[doc = concat!("Returns the response headers exposed by a wrapped ", $inner, " or report.")]
            pub fn provider_response_headers(&self) -> Option<&http::HeaderMap> {
                match self {
                    Self::$variant(error) => error.provider_response_headers(),
                    $(Self::$report(report) => report.provider_response_headers(),)?
                    _ => None,
                }
            }
        }
    };
}

forward_provider_response_helpers!(
    PromptError,
    CompletionError,
    "completion error",
    report = Report
);

impl PromptError {
    /// Build a [`PromptError::PromptCancelled`] from the history available at
    /// cancellation and a reason.
    pub fn prompt_cancelled(
        chat_history: impl IntoIterator<Item = Message>,
        reason: impl Into<String>,
    ) -> Self {
        Self::PromptCancelled {
            chat_history: chat_history.into_iter().collect(),
            reason: reason.into(),
        }
    }
}
