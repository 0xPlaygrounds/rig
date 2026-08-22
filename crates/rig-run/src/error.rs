//! Errors of the run protocol.

use thiserror::Error;

use rig_core::{completion::CompletionError, memory::MemoryError, message::Message};

/// Errors from classic agent prompting.
#[derive(Debug, Error)]
pub enum PromptError {
    /// A provider completion failed.
    #[error("CompletionError: {0}")]
    CompletionError(#[from] CompletionError),

    /// Conversation memory failed to load or persist history.
    #[error("MemoryError: {0}")]
    MemoryError(#[from] MemoryError),

    /// The run exhausted its total model-call budget.
    #[error("MaxTurnsError: reached max turns limit: {max_turns}")]
    MaxTurnsError {
        /// Configured total model-call budget.
        max_turns: usize,
        /// Canonical history available when the budget was exhausted.
        chat_history: Box<Vec<Message>>,
        /// Prompt for the call that could not be dispatched.
        prompt: Box<Message>,
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
        chat_history: Box<Vec<Message>>,
    },
}

/// Forwards the `provider_response_*` accessor trio through the variant that
/// wraps an error which itself exposes them.
macro_rules! forward_provider_response_helpers {
    ($err:ident, $variant:ident, $inner:literal) => {
        impl $err {
            #[doc = concat!("Returns the provider response body exposed by a wrapped ", $inner, ".")]
            pub fn provider_response_body(&self) -> Option<&str> {
                match self {
                    Self::$variant(error) => error.provider_response_body(),
                    _ => None,
                }
            }

            #[doc = concat!("Parses the provider response body of a wrapped ", $inner, " as JSON when present.")]
            pub fn provider_response_json(
                &self,
            ) -> Result<Option<serde_json::Value>, serde_json::Error> {
                match self {
                    Self::$variant(error) => error.provider_response_json(),
                    _ => Ok(None),
                }
            }

            #[doc = concat!("Returns the provider transport request id exposed by a wrapped ", $inner, " (rig#2314).")]
            pub fn provider_request_id(&self) -> Option<&str> {
                match self {
                    Self::$variant(error) => error.provider_request_id(),
                    _ => None,
                }
            }

            #[doc = concat!("Returns the HTTP status exposed by a wrapped ", $inner, ".")]
            pub fn provider_response_status(&self) -> Option<http::StatusCode> {
                match self {
                    Self::$variant(error) => error.provider_response_status(),
                    _ => None,
                }
            }

            #[doc = concat!("Returns the response headers exposed by a wrapped ", $inner, " — e.g. `Retry-After` on a 429 (rig#2210).")]
            pub fn provider_response_headers(&self) -> Option<&http::HeaderMap> {
                match self {
                    Self::$variant(error) => error.provider_response_headers(),
                    _ => None,
                }
            }
        }
    };
}

forward_provider_response_helpers!(PromptError, CompletionError, "completion error");

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
