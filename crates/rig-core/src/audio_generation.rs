//! Text-to-speech requests and normalized audio responses.
//!
//! ```no_run
//! use rig_core::audio_generation::{AudioGenerationModel, AudioGenerationRequestBuilder};
//!
//! # async fn example(model: impl AudioGenerationModel, voice: &str) -> Result<(), Box<dyn std::error::Error>> {
//! let response = AudioGenerationRequestBuilder::new(model)
//!     .text("Hello").voice(voice).send().await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```
use crate::completion::{ResponseIdentity, Usage};
use crate::markers::{Missing, Provided};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

crate::provider_response::provider_error_enum!(
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
    fn normalize(self, provider: &str) -> Result<AudioGenerationResponse, AudioGenerationError>;
}

/// Generates speech from text. Only [`Self::audio_generation_request`] requires
/// cloning; `Arc<M>` forwards generation calls.
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

    /// Replaces provider-specific request parameters.
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
mod provider_response_tests;
