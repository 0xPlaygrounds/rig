//! Classic prompting traits and errors layered over portable completion types.

pub use rig_core::completion::*;

use serde::de::DeserializeOwned;
use thiserror::Error;

use rig_core::{
    memory::MemoryError,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

/// Errors from classic agent prompting.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PromptError {
    /// A portable model call failed.
    #[error("CompletionError: {0}")]
    CompletionError(#[from] CompletionError),
    /// Conversation memory failed to load or append history.
    #[error("MemoryError: {0}")]
    MemoryError(#[from] MemoryError),
    /// The run exhausted its total model-call budget.
    #[error("MaxTurnsError: reached max turns limit: {max_turns}")]
    MaxTurnsError {
        /// Configured total model-call budget.
        max_turns: usize,
        /// Diagnostic history accumulated before exhaustion.
        chat_history: Box<Vec<Message>>,
        /// Prompt that could not be completed.
        prompt: Box<Message>,
    },
    /// The prompting loop was cancelled or stopped.
    #[error("PromptCancelled: {reason}")]
    PromptCancelled {
        /// Diagnostic history at cancellation.
        chat_history: Vec<Message>,
        /// Stable cancellation reason.
        reason: String,
    },
    /// The model called a tool outside the current advertised snapshot.
    #[error(
        "UnknownToolCall: model attempted to call unknown or disallowed tool `{tool_name}`. Available tools: {available_tools:?}. Allowed tools for this turn: {allowed_tools:?}"
    )]
    UnknownToolCall {
        /// Rejected provider-facing tool name.
        tool_name: String,
        /// Names registered in the runtime.
        available_tools: Vec<String>,
        /// Names advertised for the current turn.
        allowed_tools: Vec<String>,
        /// Diagnostic history at rejection.
        chat_history: Box<Vec<Message>>,
    },
}

impl PromptError {
    /// Return a wrapped provider response body when present.
    pub fn provider_response_body(&self) -> Option<&str> {
        match self {
            Self::CompletionError(error) => error.provider_response_body(),
            _ => None,
        }
    }

    /// Parse a wrapped provider response body as JSON.
    pub fn provider_response_json(&self) -> Result<Option<serde_json::Value>, serde_json::Error> {
        match self {
            Self::CompletionError(error) => error.provider_response_json(),
            _ => Ok(None),
        }
    }

    /// Return a wrapped provider HTTP status when present.
    pub fn provider_response_status(&self) -> Option<http::StatusCode> {
        match self {
            Self::CompletionError(error) => error.provider_response_status(),
            _ => None,
        }
    }

    pub(crate) fn prompt_cancelled(
        chat_history: impl IntoIterator<Item = Message>,
        reason: impl Into<String>,
    ) -> Self {
        Self::PromptCancelled {
            chat_history: chat_history.into_iter().collect(),
            reason: reason.into(),
        }
    }
}

/// Errors from typed classic-runtime output recovery.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StructuredOutputError {
    /// Prompt execution failed.
    #[error("PromptError: {0}")]
    PromptError(#[from] Box<PromptError>),
    /// Accepted output could not be decoded into the target type.
    #[error("DeserializationError: {0}")]
    DeserializationError(#[from] serde_json::Error),
    /// The model returned no accepted content.
    #[error("EmptyResponse: model returned no content")]
    EmptyResponse,
}

impl StructuredOutputError {
    /// Return a wrapped provider response body when present.
    pub fn provider_response_body(&self) -> Option<&str> {
        match self {
            Self::PromptError(error) => error.provider_response_body(),
            _ => None,
        }
    }

    /// Parse a wrapped provider response body as JSON.
    pub fn provider_response_json(&self) -> Result<Option<serde_json::Value>, serde_json::Error> {
        match self {
            Self::PromptError(error) => error.provider_response_json(),
            _ => Ok(None),
        }
    }

    /// Return a wrapped provider HTTP status when present.
    pub fn provider_response_status(&self) -> Option<http::StatusCode> {
        match self {
            Self::PromptError(error) => error.provider_response_status(),
            _ => None,
        }
    }
}

/// High-level classic one-shot prompt interface.
pub trait Prompt: WasmCompatSend + WasmCompatSync {
    /// Send one prompt and return accepted assistant text.
    fn prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> impl std::future::IntoFuture<Output = Result<String, PromptError>, IntoFuture: WasmCompatSend>;
}

/// High-level classic chat interface with caller-managed history.
pub trait Chat: WasmCompatSend + WasmCompatSync {
    /// Send one prompt, committing accepted messages to `chat_history`.
    fn chat(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: &mut Vec<Message>,
    ) -> impl std::future::Future<Output = Result<String, PromptError>> + WasmCompatSend;
}

/// High-level typed prompt interface for structured output.
pub trait TypedPrompt: WasmCompatSend + WasmCompatSync {
    /// Concrete runtime request returned for target `T`.
    type TypedRequest<T>: std::future::IntoFuture<Output = Result<T, StructuredOutputError>>
    where
        T: schemars::JsonSchema + DeserializeOwned + WasmCompatSend + 'static;

    /// Send a prompt and decode accepted output into `T`.
    fn prompt_typed<T>(&self, prompt: impl Into<Message> + WasmCompatSend) -> Self::TypedRequest<T>
    where
        T: schemars::JsonSchema + DeserializeOwned + WasmCompatSend;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::ProviderResponseError;

    #[test]
    fn provider_response_helpers_forward_through_runtime_errors() {
        let body = r#"{"error":{"message":"bad input"}}"#;
        let error = StructuredOutputError::PromptError(Box::new(PromptError::CompletionError(
            CompletionError::ProviderResponse(ProviderResponseError {
                status: Some(http::StatusCode::BAD_REQUEST),
                body: body.to_string(),
            }),
        )));

        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::BAD_REQUEST)
        );
    }
}
