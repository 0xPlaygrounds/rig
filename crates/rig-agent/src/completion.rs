//! Runtime errors for the classic agent runtime, plus the portable completion contracts.
//!
//! ```
//! use rig_agent::completion::StructuredOutputError;
//! let error = StructuredOutputError::EmptyResponse;
//! assert!(error.provider_response_body().is_none());
//! ```

use thiserror::Error;

pub use rig_core::completion::*;

pub use crate::run::response::PromptError;

crate::run::response::forward_provider_response_helpers!(
    StructuredOutputError,
    Prompt,
    "prompt error"
);

/// Why a typed run produced no value.
#[derive(Debug, Error)]
pub enum StructuredOutputError {
    /// The underlying run failed.
    #[error(transparent)]
    Prompt(#[from] PromptError),
    /// The accepted output did not deserialize into the requested type.
    #[error("structured output does not match the requested type: {error}")]
    Deserialization {
        /// The model's output that was parsed.
        output: String,
        /// Why it did not parse.
        error: serde_json::Error,
    },
    /// The model returned no accepted content.
    #[error("the model returned no structured output")]
    EmptyResponse,
}

#[cfg(test)]
mod provider_response_tests;
