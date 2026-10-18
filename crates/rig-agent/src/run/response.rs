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
            response_id: None,
            provider_request_id: None,
            finish_reason: None,
            raw,
        }
    }

    /// Attach the response identity metadata this call's attempt reported.
    pub fn with_identity(mut self, identity: ResponseIdentity) -> Self {
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
            response_id: self.response_id.clone(),
            provider_request_id: self.provider_request_id.clone(),
        }
    }
}

/// Final run output and accounting, returned by awaited runs and terminal
/// `MultiTurnStreamItem::FinalResponse` stream items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptResponse {
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
    /// excluding the input history the run started from: what a caller keeping
    /// the conversation appends to it. Empty only for a response built without
    /// a run behind it.
    #[serde(default)]
    pub messages: Vec<Message>,
    /// Structured assistant content for the final turn: its text,
    /// reasoning, images, and other parts. In output-tool mode the model's
    /// prose and tool calls are replaced by one text part holding the
    /// serialized structured output. [`output`](Self::output) is derived
    /// from it.
    pub content: Vec<AssistantContent>,
    /// Number of synthetic output-tool calls in the turn that finalized this
    /// response. Kept crate-private because it is runner bookkeeping rather
    /// than provider-facing response content.
    #[serde(skip)]
    output_tool_calls: usize,
}

impl std::fmt::Display for PromptResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.output().fmt(f)
    }
}

impl PromptResponse {
    /// A response whose final turn is the single text part `output`, with the
    /// run's `usage`.
    pub fn new(output: impl Into<String>, usage: Usage) -> Self {
        Self::from_content(vec![AssistantContent::text(output)], usage)
    }

    /// A response whose final turn is `content`.
    pub(crate) fn from_content(content: Vec<AssistantContent>, usage: Usage) -> Self {
        Self {
            content,
            usage,
            completion_calls: Vec::new(),
            messages: Vec::new(),
            output_tool_calls: 0,
        }
    }

    /// An empty run result (empty output, zero usage, no history).
    pub fn empty() -> Self {
        Self::new(String::new(), Usage::default())
    }

    /// Attach the run's transcript.
    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    /// Attach completion call details to this response.
    pub fn with_completion_calls(mut self, completion_calls: Vec<CompletionCall>) -> Self {
        self.completion_calls = completion_calls;
        self
    }

    /// Record how many times the output tool was called.
    pub(crate) fn with_output_tool_calls(mut self, count: usize) -> Self {
        self.output_tool_calls = count;
        self
    }

    /// How many times the output tool was called.
    pub fn output_tool_calls(&self) -> usize {
        self.output_tool_calls
    }

    /// The concatenated text parts of [`content`](Self::content). In
    /// output-tool mode this is the serialized structured output.
    pub fn output(&self) -> String {
        super::transcript::choice_text(&self.content)
    }

    /// Aggregated token usage across the whole run.
    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// The run's transcript (see the [field](Self::messages)).
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// The structured assistant content for the final turn (see the
    /// [field](Self::content)).
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

/// Replace calls and prose in a finished output-tool turn with final output text,
/// retaining reasoning and images. Call only after finalization, when remaining
/// calls are output-tool calls. Persisted history retains prose independently;
/// returns `None` if no call is present.
pub(crate) fn finalize_output_tool_choice(
    choice: &[AssistantContent],
    output: &str,
) -> Option<Vec<AssistantContent>> {
    let finalized_via_output_tool = choice
        .iter()
        .any(|item| matches!(item, AssistantContent::ToolCall(_)));
    if !finalized_via_output_tool {
        return None;
    }
    let mut items: Vec<AssistantContent> = choice
        .iter()
        .filter(|item| {
            !matches!(
                item,
                AssistantContent::ToolCall(_) | AssistantContent::Text(_)
            )
        })
        .cloned()
        .collect();
    items.push(AssistantContent::text(output.to_string()));
    Some(items)
}

use thiserror::Error;

/// Why an agent run failed, whether it was awaited or streamed.
///
/// Variants that wrap another error display it unchanged and forward its
/// source.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PromptError {
    /// A provider completion failed.
    #[error(transparent)]
    Provider(ProviderError),

    /// Structured effect failure from a handler, a hook, or a stream item. A
    /// relayed provider failure arrives as its report.
    #[error(transparent)]
    Report(#[from] rig_core::error::ErrorReport),

    /// The run's conversation could not be loaded from its memory store,
    /// before any model call, or the finished run could not be saved there
    /// ([`AgentBuilder::memory`](crate::AgentBuilder::memory)). The run did
    /// not succeed. A failed save may already have set aside a torn last
    /// line the log ended in (to `recovered/` in the store) and stored the
    /// blobs of the run's images; its own append stored nothing or, cut
    /// short, a cut last line, which the next save sets aside. A remote
    /// store that applied the append before failing to answer may hold all
    /// of it.
    #[error("the conversation memory failed: {error}")]
    Memory {
        /// Why the store refused or failed, such as
        /// [`SessionError::Changed`](rig_memory::SessionError::Changed) when
        /// another run saved to the log after this one loaded it.
        #[source]
        error: rig_memory::SessionError,
        /// The finished run whose save failed; `None` when the load did.
        response: Option<Box<PromptResponse>>,
    },

    /// The run exhausted its total model-call budget.
    #[error("reached the max turns limit of {max_turns}")]
    MaxTurns {
        /// Configured total model-call budget.
        max_turns: usize,
        /// Canonical history available when the budget was exhausted.
        chat_history: Vec<Message>,
        /// Prompt for the call that could not be dispatched.
        prompt: Message,
    },

    /// The run was cancelled.
    #[error("the run was cancelled: {reason}")]
    Cancelled {
        /// The run's history at cancellation, every tool call answered
        /// (closed by [`repair`](rig_core::conversation::pairing::repair)).
        chat_history: Vec<Message>,
        /// Human-readable cancellation reason.
        reason: String,
    },

    /// The model attempted to call a tool unavailable for the current turn.
    #[error(
        "model attempted to call unknown or disallowed tool `{tool_name}`. Available tools: {available_tools:?}. Allowed tools for this turn: {allowed_tools:?}"
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

/// A relayed failure is the report the origin sent.
impl From<ProviderError> for PromptError {
    fn from(error: ProviderError) -> Self {
        match error {
            ProviderError::Relayed(report) => Self::Report(*report),
            error => Self::Provider(error),
        }
    }
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

pub(crate) use forward_provider_response_helpers;

forward_provider_response_helpers!(PromptError, Provider, "completion error", report = Report);

impl PromptError {
    /// Build a [`PromptError::Cancelled`]. A driver gets the history from
    /// [`AgentRun::cancel_error`](super::AgentRun::cancel_error), or closes
    /// its own with [`repair`](rig_core::conversation::pairing::repair).
    pub fn cancelled(chat_history: Vec<Message>, reason: impl Into<String>) -> Self {
        Self::Cancelled {
            chat_history,
            reason: reason.into(),
        }
    }
}
