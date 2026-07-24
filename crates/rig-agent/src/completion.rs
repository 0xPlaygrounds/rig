//! High-level prompting traits and runtime errors for the classic agent runtime.

use serde::de::DeserializeOwned;
use thiserror::Error;

use rig_core::{
    memory::MemoryError,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

pub use rig_core::completion::*;

/// Errors from classic agent prompting.
#[derive(Debug, Error)]
#[non_exhaustive]
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

impl PromptError {
    /// Returns the provider response body exposed by a wrapped completion error.
    pub fn provider_response_body(&self) -> Option<&str> {
        match self {
            Self::CompletionError(error) => error.provider_response_body(),
            _ => None,
        }
    }

    /// Parses a wrapped provider response body as JSON when present.
    pub fn provider_response_json(&self) -> Result<Option<serde_json::Value>, serde_json::Error> {
        match self {
            Self::CompletionError(error) => error.provider_response_json(),
            _ => Ok(None),
        }
    }

    /// Returns the HTTP status exposed by a wrapped completion error.
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

/// Errors returned by typed structured prompting.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StructuredOutputError {
    /// The underlying classic run failed.
    #[error("PromptError: {0}")]
    PromptError(#[from] Box<PromptError>),
    /// The accepted response could not be deserialized.
    #[error("DeserializationError: {0}")]
    DeserializationError(#[from] serde_json::Error),
    /// The model returned no accepted content.
    #[error("EmptyResponse: model returned no content")]
    EmptyResponse,
}

impl StructuredOutputError {
    /// Returns the provider response body exposed through the wrapped prompt error.
    pub fn provider_response_body(&self) -> Option<&str> {
        match self {
            Self::PromptError(error) => error.provider_response_body(),
            _ => None,
        }
    }

    /// Parses the wrapped provider response body as JSON when present.
    pub fn provider_response_json(&self) -> Result<Option<serde_json::Value>, serde_json::Error> {
        match self {
            Self::PromptError(error) => error.provider_response_json(),
            _ => Ok(None),
        }
    }

    /// Returns the provider HTTP status exposed through the wrapped prompt error.
    pub fn provider_response_status(&self) -> Option<http::StatusCode> {
        match self {
            Self::PromptError(error) => error.provider_response_status(),
            _ => None,
        }
    }
}

/// High-level one-shot prompting for the classic runtime.
pub trait Prompt: WasmCompatSend + WasmCompatSync {
    /// Send a prompt and return accepted assistant text after runtime orchestration.
    fn prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> impl std::future::IntoFuture<Output = Result<String, PromptError>, IntoFuture: WasmCompatSend>;
}

/// High-level prompting with caller-owned canonical chat history.
pub trait Chat: WasmCompatSend + WasmCompatSync {
    /// Execute one turn and append only committed messages to `chat_history`.
    fn chat(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: &mut Vec<Message>,
    ) -> impl std::future::Future<Output = Result<String, PromptError>> + WasmCompatSend;
}

/// High-level typed structured prompting for the classic runtime.
pub trait TypedPrompt: WasmCompatSend + WasmCompatSync {
    /// Request type returned for one target output type.
    type TypedRequest<T>: std::future::IntoFuture<Output = Result<T, StructuredOutputError>>
    where
        T: schemars::JsonSchema + DeserializeOwned + WasmCompatSend + 'static;

    /// Send a prompt and deserialize the accepted structured response as `T`.
    fn prompt_typed<T>(&self, prompt: impl Into<Message> + WasmCompatSend) -> Self::TypedRequest<T>
    where
        T: schemars::JsonSchema + DeserializeOwned + WasmCompatSend;
}

#[cfg(test)]
mod provider_response_tests {
    use rig_core::{ProviderResponseError, http_client};

    use super::*;

    #[test]
    fn prompt_error_forwards_provider_response_to_completion_error() {
        let body = r#"{"error":{"message":"boom"}}"#;
        let inner =
            CompletionError::from_http_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let error = PromptError::CompletionError(inner);

        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE),
        );
        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(
            error
                .provider_response_json()
                .expect("valid json")
                .expect("present json")["error"]["message"],
            "boom",
        );
    }

    #[test]
    fn prompt_error_provider_response_helpers_forward_http_status_and_body() {
        let body = r#"{"error":{"message":"unauthorized"}}"#;
        let error = PromptError::CompletionError(CompletionError::HttpError(
            http_client::Error::InvalidStatusCodeWithMessage(
                http::StatusCode::UNAUTHORIZED,
                body.to_string(),
            ),
        ));

        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::UNAUTHORIZED)
        );
        assert_eq!(
            error.provider_response_json().expect("valid JSON body"),
            Some(serde_json::json!({
                "error": { "message": "unauthorized" }
            }))
        );
    }

    #[test]
    fn prompt_error_provider_response_helpers_forward_wrapped_completion_error() {
        let body = r#"{"error":{"code":"invalid_request","message":"bad input"}}"#;
        let error = PromptError::CompletionError(CompletionError::ProviderResponse(
            ProviderResponseError {
                status: None,
                body: body.to_string(),
            },
        ));

        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(
            error.provider_response_json().expect("valid JSON body"),
            Some(serde_json::json!({
                "error": {
                    "code": "invalid_request",
                    "message": "bad input"
                }
            }))
        );
    }

    #[test]
    fn prompt_error_provider_response_helpers_return_none_for_unrelated_variant() {
        let error = PromptError::PromptCancelled {
            chat_history: vec![Message::user("hi")],
            reason: "cancelled".to_string(),
        };

        assert_eq!(error.provider_response_body(), None);
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(
            error
                .provider_response_json()
                .expect("no body is not an error"),
            None
        );
    }

    #[test]
    fn structured_output_error_provider_response_helpers_forward_prompt_error() {
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
