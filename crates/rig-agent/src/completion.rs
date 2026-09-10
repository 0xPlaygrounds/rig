//! Runtime errors for the classic agent runtime, plus the portable completion contracts.

use thiserror::Error;

pub use rig_core::completion::*;

pub use crate::run::response::PromptError;

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

forward_provider_response_helpers!(StructuredOutputError, PromptError, "prompt error");

/// Errors returned by typed structured prompting.
#[derive(Debug, Error)]
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

#[cfg(test)]
mod provider_response_tests;
