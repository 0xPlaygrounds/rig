//! Everything related to core image generation abstractions in Rig.
//! Rig allows calling a number of different providers (that support image generation) using the [ImageGenerationModel] trait.
use crate::completion::{ResponseIdentity, Usage};
use crate::markers::{Missing, Provided};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

crate::provider_response::provider_error_enum!(
    ImageGenerationError, "image generation" {
        /// Error building the image generation request
        #[error("RequestError: {0}")]
        RequestError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),
    }
);

/// The normalized image generation response: the image plus the metadata
/// every provider can report, attributed to the provider that produced it.
///
/// This type is concrete — it carries no provider type parameter — so the
/// provider does not leak into the request builder or into any caller holding
/// a model. The provider's own payload stays reachable through a model's
/// inherent `raw_image_generation` method, which performs the same request and
/// returns the provider's native type, and through [`Self::raw`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationResponse {
    /// The generated image, decoded to bytes.
    pub image: Vec<u8>,
    /// Usage as the provider reported it. Zero-valued when the provider
    /// reported none — the same sentinel [`Usage`] documents for completions.
    #[serde(default)]
    pub usage: Usage,
    /// Stable descriptor name of the provider that produced this response,
    /// for example `"openai"`. Always populated.
    pub provider: String,
    /// Provider-reported model identifier, when the wire response named one.
    #[serde(default)]
    pub model: Option<String>,
    /// Provider-assigned response-scoped identifier, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// The provider's transport-level request identifier, taken from the HTTP
    /// response headers — the id provider support asks for. `None` means the
    /// provider reported none; that is a documented outcome, never an error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// The provider's own response for this call: the value the model's
    /// inherent `raw_image_generation` would have returned, serialized.
    /// Providers whose endpoint answers with the image bytes directly (no
    /// JSON envelope) have nothing to serialize here and leave it `Null`;
    /// their `raw_image_generation` returns the bytes.
    /// `Value::Null` otherwise means the value was built without a provider
    /// behind it (a test double), never that the provider sent nothing.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub raw: serde_json::Value,
}

impl ImageGenerationResponse {
    /// Create a response from its required parts; optional metadata starts
    /// unset and is filled in with the `with_*` helpers.
    pub fn new(image: Vec<u8>, provider: impl Into<String>) -> Self {
        Self {
            image,
            usage: Usage::new(),
            provider: provider.into(),
            model: None,
            response_id: None,
            provider_request_id: None,
            raw: serde_json::Value::Null,
        }
    }

    /// This response's identity metadata as one [`ResponseIdentity`] carrier.
    /// `message_id` is always `None`: nothing here is replayed as an
    /// assistant message.
    pub fn identity(&self) -> ResponseIdentity {
        ResponseIdentity {
            message_id: None,
            response_id: self.response_id.clone(),
            provider_request_id: self.provider_request_id.clone(),
        }
    }
}

crate::provider_response::modality_response_metadata_setters!(ImageGenerationResponse);

/// Convert a provider's own image generation payload into the normalized
/// [`ImageGenerationResponse`].
///
/// The provider descriptor name is an *input*, never something the conversion
/// knows — several providers share one wire shape, and a hardcoded name would
/// mislabel every provider but one. A trait rather than `TryFrom<(&str, T)>`
/// so that out-of-tree provider extensions can implement it on their own
/// response type without tripping the orphan rule.
pub trait NormalizeImageGenerationResponse {
    /// Normalize this payload, attributing it to `provider`.
    fn normalize(self, provider: &str) -> Result<ImageGenerationResponse, ImageGenerationError>;
}

/// Trait defining an image generation model.
///
/// The trait describes only what a model *does*: it has no associated types.
/// Construction lives on the capability client trait, and `Clone` is required
/// only by [`ImageGenerationModel::image_generation_request`], which hands the builder its own
/// copy. The trait is implemented for `Arc<M>` by forwarding.
pub trait ImageGenerationModel: WasmCompatSend + WasmCompatSync {
    fn image_generation(
        &self,
        request: ImageGenerationRequest,
    ) -> impl std::future::Future<Output = Result<ImageGenerationResponse, ImageGenerationError>>
    + WasmCompatSend;

    fn image_generation_request(&self) -> ImageGenerationRequestBuilder<Self, Missing>
    where
        Self: Sized + Clone,
    {
        ImageGenerationRequestBuilder::new(self.clone())
    }
}

impl<M> ImageGenerationModel for Arc<M>
where
    M: ImageGenerationModel,
{
    fn image_generation(
        &self,
        request: ImageGenerationRequest,
    ) -> impl std::future::Future<Output = Result<ImageGenerationResponse, ImageGenerationError>>
    + WasmCompatSend {
        (**self).image_generation(request)
    }
}

pub struct ImageGenerationRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub additional_params: Option<Value>,
}

/// A builder for `ImageGenerationRequest`.
/// Can be sent to a model provider.
pub struct ImageGenerationRequestBuilder<M, P = Missing> {
    model: M,
    prompt: P,
    width: u32,
    height: u32,
    additional_params: Option<Value>,
}

impl<M> ImageGenerationRequestBuilder<M, Missing>
where
    M: ImageGenerationModel,
{
    pub fn new(model: M) -> Self {
        Self {
            model,
            prompt: Missing,
            height: 256,
            width: 256,
            additional_params: None,
        }
    }
}

impl<M, P> ImageGenerationRequestBuilder<M, P>
where
    M: ImageGenerationModel,
{
    /// Sets the prompt for the image generation request
    pub fn prompt(self, prompt: &str) -> ImageGenerationRequestBuilder<M, Provided<String>> {
        ImageGenerationRequestBuilder {
            model: self.model,
            prompt: Provided(prompt.to_string()),
            width: self.width,
            height: self.height,
            additional_params: self.additional_params,
        }
    }

    /// The width of the generated image
    pub fn width(mut self, width: u32) -> Self {
        self.width = width;
        self
    }

    /// The height of the generated image
    pub fn height(mut self, height: u32) -> Self {
        self.height = height;
        self
    }

    /// Adds additional parameters to the image generation request.
    pub fn additional_params(mut self, params: Value) -> Self {
        self.additional_params = Some(params);
        self
    }
}

impl<M> ImageGenerationRequestBuilder<M, Provided<String>>
where
    M: ImageGenerationModel,
{
    pub fn build(self) -> ImageGenerationRequest {
        self.into_parts().1
    }

    fn into_parts(self) -> (M, ImageGenerationRequest) {
        let Self {
            model,
            prompt,
            width,
            height,
            additional_params,
        } = self;
        (
            model,
            ImageGenerationRequest {
                prompt: prompt.0,
                width,
                height,
                additional_params,
            },
        )
    }

    pub async fn send(self) -> Result<ImageGenerationResponse, ImageGenerationError> {
        let (model, request) = self.into_parts();
        model.image_generation(request).await
    }
}

#[cfg(test)]
mod provider_response_tests {
    use super::*;
    use crate::{http_client, provider_response};
    use http::StatusCode;

    #[test]
    fn image_generation_error_provider_response_helpers_with_preserved_json_body() {
        let body = r#"{"error":{"message":"content policy"}}"#;
        let error = ImageGenerationError::ProviderResponse(
            provider_response::ProviderResponseError::without_status(body.to_string()),
        );

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
