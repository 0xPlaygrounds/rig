//! High-level prompting contracts owned by the classic runtime.

pub use rig_core::completion::*;

use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};
use serde::de::DeserializeOwned;
use thiserror::Error;

/// Errors from agent prompting.
///
/// When the failure wraps [`CompletionError`], [`Self::provider_response_body`],
/// [`Self::provider_response_json`], and [`Self::provider_response_status`] forward
/// to the inner completion error's helpers.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PromptError {
    /// Something went wrong with the completion
    #[error("CompletionError: {0}")]
    CompletionError(#[from] CompletionError),

    /// Conversation memory failed to load history.
    #[error("MemoryError: {0}")]
    MemoryError(#[from] crate::memory::MemoryError),

    /// The run exhausted its total model-call budget. The budget includes the
    /// initial call and every retry or continuation; increase `.max_turns()` if
    /// the intended interaction requires more calls.
    #[error("MaxTurnsError: reached max turns limit: {max_turns}")]
    MaxTurnsError {
        max_turns: usize,
        chat_history: Box<Vec<Message>>,
        prompt: Box<Message>,
    },

    /// A prompting loop was cancelled.
    #[error("PromptCancelled: {reason}")]
    PromptCancelled {
        chat_history: Vec<Message>,
        reason: String,
    },

    /// The model emitted a structured tool call for a tool Rig did not allow
    /// for the current turn.
    #[error(
        "UnknownToolCall: model attempted to call unknown or disallowed tool `{tool_name}`. Available tools: {available_tools:?}. Allowed tools for this turn: {allowed_tools:?}"
    )]
    UnknownToolCall {
        tool_name: String,
        available_tools: Vec<String>,
        allowed_tools: Vec<String>,
        chat_history: Box<Vec<Message>>,
    },
}

impl PromptError {
    /// Returns the provider response body when this wraps a completion error that exposes one.
    pub fn provider_response_body(&self) -> Option<&str> {
        match self {
            Self::CompletionError(error) => error.provider_response_body(),
            _ => None,
        }
    }

    /// Parses the provider response body as JSON when available through a wrapped completion error.
    ///
    /// Returns:
    /// - `Ok(Some(value))` when a body is present and valid JSON.
    /// - `Ok(None)` when no provider response body is available.
    /// - `Err(error)` when a body is present but isn't valid JSON.
    pub fn provider_response_json(&self) -> Result<Option<serde_json::Value>, serde_json::Error> {
        match self {
            Self::CompletionError(error) => error.provider_response_json(),
            _ => Ok(None),
        }
    }

    /// Returns the HTTP status when this wraps a completion error that preserves
    /// one, including from non-success HTTP responses and 2xx error envelopes.
    pub fn provider_response_status(&self) -> Option<http::StatusCode> {
        match self {
            Self::CompletionError(error) => error.provider_response_status(),
            _ => None,
        }
    }

    #[doc(hidden)]
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

/// Errors that can occur when using typed structured output via [`TypedPrompt::prompt_typed`].
///
/// When the failure wraps [`PromptError`] that in turn wraps a [`CompletionError`]
/// exposing a provider response, [`Self::provider_response_body`],
/// [`Self::provider_response_json`], and [`Self::provider_response_status`] forward
/// through the chain.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StructuredOutputError {
    /// An error occurred during the prompt execution.
    #[error("PromptError: {0}")]
    PromptError(#[from] Box<PromptError>),

    /// Failed to deserialize the model's response into the target type.
    #[error("DeserializationError: {0}")]
    DeserializationError(#[from] serde_json::Error),

    /// The model returned an empty response.
    #[error("EmptyResponse: model returned no content")]
    EmptyResponse,
}

impl StructuredOutputError {
    /// Returns the provider response body when this wraps a prompt error that exposes one.
    pub fn provider_response_body(&self) -> Option<&str> {
        match self {
            Self::PromptError(error) => error.provider_response_body(),
            _ => None,
        }
    }

    /// Parses the provider response body as JSON when available through a wrapped prompt error.
    ///
    /// Returns:
    /// - `Ok(Some(value))` when a body is present and valid JSON.
    /// - `Ok(None)` when no provider response body is available.
    /// - `Err(error)` when a body is present but isn't valid JSON.
    pub fn provider_response_json(&self) -> Result<Option<serde_json::Value>, serde_json::Error> {
        match self {
            Self::PromptError(error) => error.provider_response_json(),
            _ => Ok(None),
        }
    }

    /// Returns the HTTP status when this wraps a prompt error that preserves one.
    pub fn provider_response_status(&self) -> Option<http::StatusCode> {
        match self {
            Self::PromptError(error) => error.provider_response_status(),
            _ => None,
        }
    }
}

// ================================================================
// Implementations
// ================================================================
/// Trait defining a high-level LLM simple prompt interface (i.e.: prompt in, response out).
pub trait Prompt: WasmCompatSend + WasmCompatSync {
    /// Send a simple prompt to the underlying completion model.
    ///
    /// If the completion model's response is a message, then it is returned as a string.
    ///
    /// If the completion model's response is a tool call, then the tool is called and
    /// the result is returned as a string.
    ///
    /// If the tool does not exist, or the tool call fails, then an error is returned.
    fn prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> impl std::future::IntoFuture<Output = Result<String, PromptError>, IntoFuture: WasmCompatSend>;
}

/// Trait defining a high-level LLM chat interface (i.e.: prompt and chat history in, response out).
pub trait Chat: WasmCompatSend + WasmCompatSync {
    /// Send a prompt with optional chat history to the underlying completion model.
    ///
    /// If the completion model's response is a message, then it is returned as a string.
    ///
    /// If the completion model's response is a tool call, then the tool is called and the result
    /// is returned as a string.
    ///
    /// If the tool does not exist, or the tool call fails, then an error is returned.
    ///
    /// The prompt and any assistant or tool messages produced during the turn
    /// are appended to `chat_history`. Callers should pass the current
    /// conversation history and should not push the user prompt themselves
    /// before calling this method.
    fn chat(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: &mut Vec<Message>,
    ) -> impl std::future::Future<Output = Result<String, PromptError>> + WasmCompatSend;
}

/// Trait defining a high-level typed prompt interface for structured output.
///
/// This trait provides an ergonomic way to get typed responses from an LLM by automatically
/// generating a JSON schema from the target type and deserializing the response.
///
/// # Example
/// ```rust,ignore
/// use rig_core::prelude::*;
/// use schemars::JsonSchema;
/// use serde::Deserialize;
///
/// #[derive(Debug, Deserialize, JsonSchema)]
/// struct WeatherForecast {
///     city: String,
///     temperature_f: f64,
///     conditions: String,
/// }
///
/// let agent = client.agent("gpt-4o").build();
/// let forecast: WeatherForecast = agent
///     .prompt_typed("What's the weather in NYC?")
///     .await?;
/// ```
pub trait TypedPrompt: WasmCompatSend + WasmCompatSync {
    /// The type of the typed prompt request returned by `prompt_typed`.
    type TypedRequest<T>: std::future::IntoFuture<Output = Result<T, StructuredOutputError>>
    where
        T: schemars::JsonSchema + DeserializeOwned + WasmCompatSend + 'static;

    /// Send a prompt and receive a typed structured response.
    ///
    /// The JSON schema for `T` is automatically generated and sent to the provider.
    /// Providers that support native structured outputs will constrain the model's
    /// response to match this schema.
    ///
    /// # Type Parameters
    /// * `T` - The target type to deserialize the response into. Must implement
    ///   `JsonSchema` (for schema generation), `DeserializeOwned` (for deserialization),
    ///   and `WasmCompatSend` (for async compatibility).
    ///
    /// # Example
    /// ```rust,ignore
    /// // Type can be inferred
    /// let forecast: WeatherForecast = agent.prompt_typed("What's the weather?").await?;
    ///
    /// // Or specified explicitly with turbofish
    /// let forecast = agent.prompt_typed::<WeatherForecast>("What's the weather?").await?;
    /// ```
    fn prompt_typed<T>(&self, prompt: impl Into<Message> + WasmCompatSend) -> Self::TypedRequest<T>
    where
        T: schemars::JsonSchema + DeserializeOwned + WasmCompatSend;
}
