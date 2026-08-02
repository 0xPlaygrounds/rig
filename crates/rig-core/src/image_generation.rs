//! Everything related to core image generation abstractions in Rig.
//! Rig allows calling a number of different providers (that support image generation)
//! through each provider's `functions::generate_image` free function over these types.
use crate::{http_client, provider_response};
use serde_json::Value;
use thiserror::Error;

/// Errors returned by image generation models.
///
/// Inspect provider failures with [`Self::provider_response_body`],
/// [`Self::provider_response_json`], and [`Self::provider_response_status`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ImageGenerationError {
    /// Http error (e.g.: connection error, timeout, etc.)
    #[error("HttpError: {0}")]
    HttpError(#[from] http_client::Error),

    /// Json error (e.g.: serialization, deserialization)
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Error building the image generation request
    #[error("RequestError: {0}")]
    RequestError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

    /// Error parsing the image generation response
    #[error("ResponseError: {0}")]
    ResponseError(String),

    /// Error returned by the image generation model provider
    #[error("ProviderError: {0}")]
    ProviderError(String),

    /// Raw error response preserved from the image generation model provider
    #[error("ProviderResponseError: {0}")]
    ProviderResponse(provider_response::ProviderResponseError),
}

impl From<crate::json_utils::RequestOverlayError> for ImageGenerationError {
    fn from(error: crate::json_utils::RequestOverlayError) -> Self {
        Self::RequestError(Box::new(error))
    }
}

crate::provider_response::impl_provider_response_helpers!(ImageGenerationError);

/// A unified response for a model image generation, returning both the image and the raw response.
#[derive(Debug)]
pub struct ImageGenerationResponse<T> {
    pub image: Vec<u8>,
    pub response: T,
}

/// An image generation request.
#[non_exhaustive]
pub struct ImageGenerationRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub additional_params: Option<Value>,
}

impl ImageGenerationRequest {
    /// Creates a request from the prompt, defaulting to a 256x256 image.
    ///
    /// Refine with the `with_*` methods, then execute it with the provider's
    /// `functions::generate_image`.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            width: 256,
            height: 256,
            additional_params: None,
        }
    }

    /// Sets the prompt for the image generation request.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    /// Sets the width of the generated image.
    pub fn with_width(mut self, width: u32) -> Self {
        self.width = width;
        self
    }

    /// Sets the height of the generated image.
    pub fn with_height(mut self, height: u32) -> Self {
        self.height = height;
        self
    }

    /// Sets additional parameters for the image generation request.
    pub fn with_additional_params(mut self, params: Value) -> Self {
        self.additional_params = Some(params);
        self
    }
}

#[cfg(test)]
mod provider_response_tests {
    use super::*;
    use http::StatusCode;

    #[test]
    fn image_generation_error_provider_response_helpers_with_preserved_json_body() {
        let body = r#"{"error":{"message":"content policy"}}"#;
        let error =
            ImageGenerationError::ProviderResponse(provider_response::ProviderResponseError {
                status: None,
                body: body.to_string(),
            });

        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(
            error.provider_response_json().expect("valid JSON"),
            Some(serde_json::json!({ "error": { "message": "content policy" } }))
        );
    }

    #[test]
    fn image_generation_error_provider_response_helpers_with_http_non_success() {
        let body = r#"{"error":{"message":"bad request"}}"#;
        let error =
            ImageGenerationError::HttpError(http_client::Error::InvalidStatusCodeWithMessage(
                StatusCode::BAD_REQUEST,
                body.to_string(),
            ));

        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(
            error.provider_response_status(),
            Some(StatusCode::BAD_REQUEST)
        );
        assert_eq!(
            error.provider_response_json().expect("valid JSON"),
            Some(serde_json::json!({ "error": { "message": "bad request" } }))
        );
    }

    #[test]
    fn image_generation_error_provider_error_is_not_a_provider_response() {
        let error = ImageGenerationError::ProviderError("internal diagnostic".to_string());

        assert_eq!(error.provider_response_body(), None);
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(error.provider_response_json().expect("no body"), None);
    }

    #[test]
    fn image_generation_error_provider_response_helpers_with_unrelated_variant() {
        let error = ImageGenerationError::ResponseError("parse failed".to_string());

        assert_eq!(error.provider_response_body(), None);
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(error.provider_response_json().expect("no body"), None);
    }
}
