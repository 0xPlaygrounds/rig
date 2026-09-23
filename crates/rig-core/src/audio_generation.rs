//! Text-to-speech requests and normalized audio responses.
//!
//! ```no_run
//! use rig_core::audio_generation::{AudioGenerationModel, AudioGenerationRequestBuilder};
//!
//! # async fn example(model: impl AudioGenerationModel, voice: &str) -> Result<(), Box<dyn std::error::Error>> {
//! let response = AudioGenerationRequestBuilder::new(model, "Hello", voice)
//!     .send().await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```
use crate::completion::{ResponseIdentity, Usage};
use crate::error::ProviderError;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

/// Generated audio and normalized provider metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioGenerationResponse {
    /// The generated audio bytes.
    pub audio: Vec<u8>,
    /// Usage as the provider reported it; every counter is `None` when the
    /// provider reported none (see [`Usage`]).
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
    /// Transport request ID from HTTP headers, or `None` when unreported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// Provider response metadata. May be null for byte-only responses or
    /// responses constructed without metadata; audio bytes remain in [`Self::audio`].
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub raw: serde_json::Value,
}

impl AudioGenerationResponse {
    /// Create a response from its required parts; optional metadata starts
    /// unset and is filled in with the `with_*` helpers.
    pub fn new(audio: Vec<u8>, provider: impl Into<String>) -> Self {
        Self {
            audio,
            usage: Usage::default(),
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

/// Normalizes provider audio payloads, attributing the response to the supplied
/// provider name.
pub trait NormalizeAudioGenerationResponse {
    /// Normalize this payload, attributing it to `provider`.
    fn normalize(self, provider: &str) -> Result<AudioGenerationResponse, ProviderError>;
}

/// Generates speech from text. Only [`Self::audio_generation_request`] requires
/// cloning; `Arc<M>` forwards generation calls.
pub trait AudioGenerationModel: WasmCompatSend + WasmCompatSync {
    fn audio_generation(
        &self,
        request: AudioGenerationRequest,
    ) -> impl std::future::Future<Output = Result<AudioGenerationResponse, ProviderError>> + WasmCompatSend;

    /// Creates a request builder to speak `text` in `voice`.
    fn audio_generation_request(
        &self,
        text: impl Into<String>,
        voice: impl Into<String>,
    ) -> AudioGenerationRequestBuilder<Self>
    where
        Self: Sized + Clone,
    {
        AudioGenerationRequestBuilder::new(self.clone(), text, voice)
    }
}

impl<M> AudioGenerationModel for Arc<M>
where
    M: AudioGenerationModel,
{
    fn audio_generation(
        &self,
        request: AudioGenerationRequest,
    ) -> impl std::future::Future<Output = Result<AudioGenerationResponse, ProviderError>> + WasmCompatSend
    {
        (**self).audio_generation(request)
    }
}

pub struct AudioGenerationRequest {
    pub text: String,
    pub voice: String,
    pub speed: f32,
    pub additional_params: Option<Value>,
}

/// Builds a speech request for a text and voice. The speed defaults to 1.0.
pub struct AudioGenerationRequestBuilder<M> {
    model: M,
    request: AudioGenerationRequest,
}

impl<M> AudioGenerationRequestBuilder<M> {
    /// A request to speak `text` in `voice`.
    pub fn new(model: M, text: impl Into<String>, voice: impl Into<String>) -> Self {
        Self {
            model,
            request: AudioGenerationRequest {
                text: text.into(),
                voice: voice.into(),
                speed: 1.0,
                additional_params: None,
            },
        }
    }

    /// The speed of the generated audio.
    pub fn speed(mut self, speed: f32) -> Self {
        self.request.speed = speed;
        self
    }

    /// Merges provider-specific parameters over earlier ones, key by key for
    /// JSON objects; `None` clears existing parameters.
    pub fn additional_params(mut self, params: impl Into<Option<Value>>) -> Self {
        self.request.additional_params =
            crate::json_utils::merge_params(self.request.additional_params.take(), params.into());
        self
    }

    /// Builds the audio generation request.
    pub fn build(self) -> AudioGenerationRequest {
        self.request
    }
}

impl<M: AudioGenerationModel> AudioGenerationRequestBuilder<M> {
    /// Sends the request to the model and returns the generated audio.
    pub async fn send(self) -> Result<AudioGenerationResponse, ProviderError> {
        self.model.audio_generation(self.request).await
    }
}

#[cfg(test)]
mod builder_tests;
#[cfg(test)]
mod provider_response_tests;
