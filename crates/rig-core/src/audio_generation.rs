//! Everything related to audio generation (ie, Text To Speech).
//! Rig abstracts over a number of different providers using the [AudioGenerationModel] trait.
use crate::completion::{ResponseIdentity, Usage};
use crate::markers::{Missing, Provided};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

crate::provider_response::provider_error_enum!(
    ///
    /// HTTP audio failures preserve the provider's status and body: a non-success
    /// response surfaces as [`Self::HttpError`], and a provider error envelope
    /// returned with a 2xx status surfaces as [`Self::ProviderResponse`] (for
    /// example the Hyperbolic audio path). Both are read by the helpers.
    AudioGenerationError, "audio generation" {
        #[cfg(not(target_family = "wasm"))]
        /// Error building the audio generation request
        #[error("RequestError: {0}")]
        RequestError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

        #[cfg(target_family = "wasm")]
        /// Error building the audio generation request
        #[error("RequestError: {0}")]
        RequestError(#[from] Box<dyn std::error::Error + 'static>),
    }
);

/// The normalized audio generation response: the audio plus the metadata
/// every provider can report, attributed to the provider that produced it.
///
/// This type is concrete — it carries no provider type parameter — so the
/// provider does not leak into the request builder or into any caller holding
/// a model. The provider's own payload stays reachable through a model's
/// inherent `raw_audio_generation` method, which performs the same request and
/// returns the provider's native type, and through [`Self::raw`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioGenerationResponse {
    /// The generated audio bytes.
    pub audio: Vec<u8>,
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
    /// inherent `raw_audio_generation` would have returned, serialized.
    /// Most text-to-speech endpoints answer with the audio bytes directly and
    /// no JSON envelope; those providers leave this `Null` and
    /// `raw_audio_generation` returns the bytes.
    /// `Value::Null` otherwise means the value was built without a provider
    /// behind it (a test double), never that the provider sent nothing.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub raw: serde_json::Value,
}

impl AudioGenerationResponse {
    /// Create a response from its required parts; optional metadata starts
    /// unset and is filled in with the `with_*` helpers.
    pub fn new(audio: Vec<u8>, provider: impl Into<String>) -> Self {
        Self {
            audio,
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

crate::provider_response::modality_response_metadata_setters!(AudioGenerationResponse);

/// Convert a provider's own audio generation payload into the normalized
/// [`AudioGenerationResponse`].
///
/// The provider descriptor name is an *input*, never something the conversion
/// knows — several providers share one wire shape, and a hardcoded name would
/// mislabel every provider but one. A trait rather than `TryFrom<(&str, T)>`
/// so that out-of-tree provider extensions can implement it on their own
/// response type without tripping the orphan rule.
pub trait NormalizeAudioGenerationResponse {
    /// Normalize this payload, attributing it to `provider`.
    fn normalize(self, provider: &str) -> Result<AudioGenerationResponse, AudioGenerationError>;
}

/// Trait defining an audio generation (text-to-speech) model.
///
/// The trait describes only what a model *does*: it has no associated types.
/// Construction lives on the capability client trait, and `Clone` is required
/// only by [`AudioGenerationModel::audio_generation_request`], which hands the builder its own
/// copy. The trait is implemented for `Arc<M>` by forwarding.
pub trait AudioGenerationModel: WasmCompatSend + WasmCompatSync {
    fn audio_generation(
        &self,
        request: AudioGenerationRequest,
    ) -> impl std::future::Future<Output = Result<AudioGenerationResponse, AudioGenerationError>>
    + WasmCompatSend;

    fn audio_generation_request(&self) -> AudioGenerationRequestBuilder<Self, Missing, Missing>
    where
        Self: Sized + Clone,
    {
        AudioGenerationRequestBuilder::new(self.clone())
    }
}

impl<M> AudioGenerationModel for Arc<M>
where
    M: AudioGenerationModel,
{
    fn audio_generation(
        &self,
        request: AudioGenerationRequest,
    ) -> impl std::future::Future<Output = Result<AudioGenerationResponse, AudioGenerationError>>
    + WasmCompatSend {
        (**self).audio_generation(request)
    }
}

pub struct AudioGenerationRequest {
    pub text: String,
    pub voice: String,
    pub speed: f32,
    pub additional_params: Option<Value>,
}

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
        self.into_parts().1
    }

    fn into_parts(self) -> (M, AudioGenerationRequest) {
        let Self {
            model,
            text,
            voice,
            speed,
            additional_params,
        } = self;
        (
            model,
            AudioGenerationRequest {
                text: text.0,
                voice: voice.0,
                speed,
                additional_params,
            },
        )
    }

    pub async fn send(self) -> Result<AudioGenerationResponse, AudioGenerationError> {
        let (model, request) = self.into_parts();
        model.audio_generation(request).await
    }
}

#[cfg(test)]
mod provider_response_tests {
    use super::*;
    use crate::{http_client, provider_response};
    use http::StatusCode;

    #[test]
    fn audio_generation_error_provider_response_helpers_with_preserved_json_body() {
        let body = r#"{"error":{"message":"invalid voice"}}"#;
        let error = AudioGenerationError::ProviderResponse(
            provider_response::ProviderResponseError::without_status(body.to_string()),
        );

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
