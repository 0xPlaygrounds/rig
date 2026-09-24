//! Audio transcription requests, normalized responses, and model interfaces.
//!
//! ```no_run
//! use rig_core::transcription::{TranscriptionModel, TranscriptionRequestBuilder};
//!
//! # async fn example(model: impl TranscriptionModel) -> Result<(), Box<dyn std::error::Error>> {
//! let response = TranscriptionRequestBuilder::from_file(model, "audio.wav")?
//!     .send().await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```
use crate::error::ProviderError;
use crate::json_utils;
use crate::response::Response;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use std::io;
use std::sync::Arc;
use std::{fs, path::Path};

/// A transcript and the metadata the provider reported. The usage holds
/// token counts only, never an audio duration.
pub type TranscriptionResponse = Response<String>;

/// Transcribes audio into normalized responses. Only
/// [`Self::transcription_request`] requires cloning; `Arc<M>` forwards operations.
pub trait TranscriptionModel: WasmCompatSend + WasmCompatSync {
    /// Transcribes the supplied audio request or returns a provider or transport error.
    fn transcription(
        &self,
        request: TranscriptionRequest,
    ) -> impl std::future::Future<Output = Result<TranscriptionResponse, ProviderError>> + WasmCompatSend;

    /// Creates a request builder over `data`.
    fn transcription_request(&self, data: Vec<u8>) -> TranscriptionRequestBuilder<Self>
    where
        Self: Sized + Clone,
    {
        TranscriptionRequestBuilder::new(self.clone(), data)
    }
}

impl<M> TranscriptionModel for Arc<M>
where
    M: TranscriptionModel,
{
    fn transcription(
        &self,
        request: TranscriptionRequest,
    ) -> impl std::future::Future<Output = Result<TranscriptionResponse, ProviderError>> + WasmCompatSend
    {
        (**self).transcription(request)
    }
}

/// Struct representing a general transcription request that can be sent to a transcription model provider.
pub struct TranscriptionRequest {
    /// The file data to be sent to the transcription model provider
    pub data: Vec<u8>,
    /// The file name to be used in the request
    pub filename: String,
    /// The language used in the response from the transcription model provider
    pub language: Option<String>,
    /// The prompt to be sent to the transcription model provider
    pub prompt: Option<String>,
    /// The temperature sent to the transcription model provider
    pub temperature: Option<f64>,
    /// Additional parameters to be sent to the transcription model provider
    pub additional_params: Option<serde_json::Value>,
}

/// The filename a request carries until the caller or its file names it.
const DEFAULT_FILENAME: &str = "file";

/// Builds a transcription request over the supplied audio. The model is moved
/// into the builder and consumed by [`Self::send`]. The audio is not validated
/// for format or nonemptiness.
pub struct TranscriptionRequestBuilder<M> {
    model: M,
    request: TranscriptionRequest,
}

impl<M> TranscriptionRequestBuilder<M> {
    /// A request over `data`, named `"file"` until [`Self::filename`] names it.
    pub fn new(model: M, data: Vec<u8>) -> Self {
        Self {
            model,
            request: TranscriptionRequest {
                data,
                filename: DEFAULT_FILENAME.to_owned(),
                language: None,
                prompt: None,
                temperature: None,
                additional_params: None,
            },
        }
    }

    /// A request over the file at `path`, named after its base name. Reads the
    /// file synchronously and returns I/O errors unchanged.
    pub fn from_file(model: M, path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let filename = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned());
        Ok(Self::new(model, fs::read(path)?).filename(filename))
    }

    /// Names the audio file; `None` restores the default name.
    pub fn filename(mut self, filename: impl Into<Option<String>>) -> Self {
        self.request.filename = filename
            .into()
            .unwrap_or_else(|| DEFAULT_FILENAME.to_owned());
        self
    }

    /// Sets the output language.
    pub fn language(mut self, language: String) -> Self {
        self.request.language = Some(language);
        self
    }

    /// Sets the prompt sent with the audio.
    pub fn prompt(mut self, prompt: String) -> Self {
        self.request.prompt = Some(prompt);
        self
    }

    /// Sets the sampling temperature.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.request.temperature = Some(temperature);
        self
    }

    /// Merges provider-specific parameters over earlier ones, key by key for
    /// JSON objects; `None` clears existing parameters.
    pub fn additional_params(mut self, params: impl Into<Option<serde_json::Value>>) -> Self {
        self.request.additional_params =
            json_utils::merge_params(self.request.additional_params.take(), params.into());
        self
    }

    /// Builds the transcription request.
    pub fn build(self) -> TranscriptionRequest {
        self.request
    }
}

impl<M: TranscriptionModel> TranscriptionRequestBuilder<M> {
    /// Sends the request to the model and returns its transcription.
    pub async fn send(self) -> Result<TranscriptionResponse, ProviderError> {
        self.model.transcription(self.request).await
    }
}

#[cfg(test)]
mod builder_tests;
#[cfg(test)]
mod provider_response_tests;
