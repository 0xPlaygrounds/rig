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
use crate::error::ProviderError;
use crate::response::Response;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use serde_json::Value;
use std::sync::Arc;

/// The generated audio bytes and the metadata the provider reported.
pub type AudioGenerationResponse = Response<Vec<u8>>;

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
