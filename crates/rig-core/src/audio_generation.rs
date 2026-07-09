//! Everything related to audio generation (ie, Text To Speech).
//! Rig abstracts over a number of different providers using the [AudioGenerationModel] trait.
use crate::markers::{Missing, Provided};
use crate::{
    http_client, provider_response,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use serde_json::Value;
use thiserror::Error;

/// Errors returned by audio generation models.
///
/// Inspect provider failures with [`Self::provider_response_body`],
/// [`Self::provider_response_json`], and [`Self::provider_response_status`].
///
/// HTTP audio failures preserve the provider's status and body: a non-success
/// response surfaces as [`Self::HttpError`], and a provider error envelope
/// returned with a 2xx status surfaces as [`Self::ProviderResponse`] (for
/// example the Hyperbolic audio path). Both are read by the helpers.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AudioGenerationError {
    /// Http error (e.g.: connection error, timeout, etc.)
    #[error("HttpError: {0}")]
    HttpError(#[from] http_client::Error),

    /// Json error (e.g.: serialization, deserialization)
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Error building the audio generation request
    #[error("RequestError: {0}")]
    RequestError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

    /// Error parsing the audio generation response
    #[error("ResponseError: {0}")]
    ResponseError(String),

    /// Error returned by the audio generation model provider
    #[error("ProviderError: {0}")]
    ProviderError(String),

    /// Raw error response preserved from the audio generation model provider
    #[error("ProviderResponseError: {0}")]
    ProviderResponse(provider_response::ProviderResponseError),
}

crate::provider_response::impl_provider_response_helpers!(AudioGenerationError);

pub struct AudioGenerationResponse<T> {
    pub audio: Vec<u8>,
    pub response: T,
}

pub trait AudioGenerationModel: Sized + Clone + WasmCompatSend + WasmCompatSync {
    type Response: Send + Sync;

    type Client;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self;

    fn audio_generation(
        &self,
        request: AudioGenerationRequest,
    ) -> impl std::future::Future<
        Output = Result<AudioGenerationResponse<Self::Response>, AudioGenerationError>,
    > + Send;

    fn audio_generation_request(&self) -> AudioGenerationRequestBuilder<Self, Missing, Missing> {
        AudioGenerationRequestBuilder::new(self.clone())
    }
}
#[non_exhaustive]
pub struct AudioGenerationRequest {
    pub text: String,
    pub voice: String,
    pub speed: f32,
    pub additional_params: Option<Value>,
}

#[non_exhaustive]
pub struct AudioGenerationRequestBuilder<M, T = Missing, V = Missing>
where
    M: AudioGenerationModel,
{
    model: M,
    text: T,
    voice: V,
    speed: f32,
    additional_params: Option<Value>,
}

impl<M> AudioGenerationRequestBuilder<M, Missing, Missing>
where
    M: AudioGenerationModel,
{
    pub fn new(model: M) -> Self {
        Self {
            model,
            text: Missing,
            voice: Missing,
            speed: 1.0,
            additional_params: None,
        }
    }
}

impl<M, T, V> AudioGenerationRequestBuilder<M, T, V>
where
    M: AudioGenerationModel,
{
    /// Sets the text for the audio generation request
    pub fn text(self, text: &str) -> AudioGenerationRequestBuilder<M, Provided<String>, V> {
        AudioGenerationRequestBuilder {
            model: self.model,
            text: Provided(text.to_string()),
            voice: self.voice,
            speed: self.speed,
            additional_params: self.additional_params,
        }
    }

    /// The voice of the generated audio
    pub fn voice(self, voice: &str) -> AudioGenerationRequestBuilder<M, T, Provided<String>> {
        AudioGenerationRequestBuilder {
            model: self.model,
            text: self.text,
            voice: Provided(voice.to_string()),
            speed: self.speed,
            additional_params: self.additional_params,
        }
    }

    /// The speed of the generated audio
    pub fn speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    /// Adds additional parameters to the audio generation request.
    pub fn additional_params(mut self, params: Value) -> Self {
        self.additional_params = Some(params);
        self
    }
}

impl<M> AudioGenerationRequestBuilder<M, Provided<String>, Provided<String>>
where
    M: AudioGenerationModel,
{
    pub fn build(self) -> AudioGenerationRequest {
        AudioGenerationRequest {
            text: self.text.0,
            voice: self.voice.0,
            speed: self.speed,
            additional_params: self.additional_params,
        }
    }

    pub async fn send(self) -> Result<AudioGenerationResponse<M::Response>, AudioGenerationError> {
        let model = self.model.clone();

        model.audio_generation(self.build()).await
    }
}

#[cfg(test)]
mod provider_response_tests {
    use super::*;
    use http::StatusCode;

    #[test]
    fn audio_generation_error_provider_response_helpers_with_preserved_json_body() {
        let body = r#"{"error":{"message":"invalid voice"}}"#;
        let error =
            AudioGenerationError::ProviderResponse(provider_response::ProviderResponseError {
                status: None,
                body: body.to_string(),
            });

        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(
            error.provider_response_json().expect("valid JSON"),
            Some(serde_json::json!({ "error": { "message": "invalid voice" } }))
        );
    }

    #[test]
    fn audio_generation_error_provider_response_helpers_with_http_non_success() {
        let body = r#"{"error":{"message":"bad request"}}"#;
        let error =
            AudioGenerationError::HttpError(http_client::Error::InvalidStatusCodeWithMessage(
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
    fn audio_generation_error_provider_error_is_not_a_provider_response() {
        let error = AudioGenerationError::ProviderError("internal diagnostic".to_string());

        assert_eq!(error.provider_response_body(), None);
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(error.provider_response_json().expect("no body"), None);
    }

    #[test]
    fn audio_generation_error_provider_response_helpers_with_unrelated_variant() {
        let error = AudioGenerationError::ResponseError("parse failed".to_string());

        assert_eq!(error.provider_response_body(), None);
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(error.provider_response_json().expect("no body"), None);
    }
}
