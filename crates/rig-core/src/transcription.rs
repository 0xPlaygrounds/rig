//! Audio transcription requests, normalized responses, and model interfaces.
//!
//! ```no_run
//! use rig_core::transcription::{TranscriptionModel, TranscriptionRequestBuilder};
//!
//! # async fn example(model: impl TranscriptionModel) -> Result<(), Box<dyn std::error::Error>> {
//! let response = TranscriptionRequestBuilder::new(model)
//!     .load_file("audio.wav")?
//!     .send().await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```
use crate::completion::{ResponseIdentity, Usage};
use crate::error::ProviderError;
use crate::json_utils;
use crate::markers::{Missing, Provided};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use serde::{Deserialize, Serialize};
use std::io;
use std::sync::Arc;
use std::{fs, path::Path};

/// Transcript and normalized provider metadata, with provider-specific data
/// available through [`Self::raw`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    /// The transcribed text.
    pub text: String,
    /// Provider-reported token usage. Unreported counters remain `None`;
    /// this field does not contain audio duration.
    #[serde(default)]
    pub usage: Usage,
    /// Stable descriptor name of the provider that produced this response,
    /// for example `"openai"`. Always populated.
    pub provider: String,
    /// Provider-reported model identifier, when the wire response named one.
    /// This is the model the provider says answered, not the model requested.
    #[serde(default)]
    pub model: Option<String>,
    /// Provider-assigned response-scoped identifier, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Transport request ID from HTTP headers, or `None` when unreported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// Provider response document. Defaults to null until populated.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub raw: serde_json::Value,
}

impl TranscriptionResponse {
    /// Create a response from its required parts; optional metadata starts
    /// unset and is filled in with the `with_*` helpers.
    pub fn new(text: impl Into<String>, provider: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            usage: Usage::default(),
            provider: provider.into(),
            model: None,
            response_id: None,
            provider_request_id: None,
            raw: serde_json::Value::Null,
        }
    }

    /// This response's identity metadata as one [`ResponseIdentity`] carrier.
    /// Transcriptions are never replayed as assistant messages, so
    /// `message_id` is always `None`.
    pub fn identity(&self) -> ResponseIdentity {
        ResponseIdentity {
            message_id: None,
            response_id: self.response_id.clone(),
            provider_request_id: self.provider_request_id.clone(),
        }
    }
}

crate::provider_response::modality_response_metadata_setters!(TranscriptionResponse);

/// Converts provider payloads into normalized transcription responses.
/// Implementations must attribute the response to the supplied provider name.
pub trait NormalizeTranscriptionResponse {
    /// Normalize this payload, attributing it to `provider`.
    fn normalize(self, provider: &str) -> Result<TranscriptionResponse, ProviderError>;
}

/// Transcribes audio into normalized responses. Only
/// [`Self::transcription_request`] requires cloning; `Arc<M>` forwards operations.
pub trait TranscriptionModel: WasmCompatSend + WasmCompatSync {
    /// Transcribes the supplied audio request or returns a provider or transport error.
    fn transcription(
        &self,
        request: TranscriptionRequest,
    ) -> impl std::future::Future<Output = Result<TranscriptionResponse, ProviderError>> + WasmCompatSend;

    /// Creates a request builder without audio data.
    fn transcription_request(&self) -> TranscriptionRequestBuilder<Self, Missing>
    where
        Self: Sized + Clone,
    {
        TranscriptionRequestBuilder::new(self.clone())
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

/// Builds a transcription request after audio data is supplied.
/// The model is moved into the builder and consumed when sending; building
/// without sending drops it. Data presence is tracked by type, not validated
/// for audio format or nonemptiness.
pub struct TranscriptionRequestBuilder<M, D> {
    model: M,
    data: D,
    filename: Option<String>,
    language: Option<String>,
    prompt: Option<String>,
    temperature: Option<f64>,
    additional_params: Option<serde_json::Value>,
}

impl<M> TranscriptionRequestBuilder<M, Missing>
where
    M: TranscriptionModel,
{
    pub fn new(model: M) -> Self {
        TranscriptionRequestBuilder {
            model,
            data: Missing,
            filename: None,
            language: None,
            prompt: None,
            temperature: None,
            additional_params: None,
        }
    }
}

impl<M, D> TranscriptionRequestBuilder<M, D>
where
    M: TranscriptionModel,
{
    pub fn filename(mut self, filename: Option<String>) -> Self {
        self.filename = filename;
        self
    }

    /// Supplies audio bytes and enables building or sending the request.
    pub fn data(self, data: Vec<u8>) -> TranscriptionRequestBuilder<M, Provided<Vec<u8>>> {
        TranscriptionRequestBuilder {
            model: self.model,
            data: Provided(data),
            filename: self.filename,
            language: self.language,
            prompt: self.prompt,
            temperature: self.temperature,
            additional_params: self.additional_params,
        }
    }

    /// Reads a file synchronously, returning I/O errors unchanged. Uses its base
    /// filename when available and enables building or sending the request.
    pub fn load_file<P>(
        self,
        path: P,
    ) -> io::Result<TranscriptionRequestBuilder<M, Provided<Vec<u8>>>>
    where
        P: AsRef<Path>,
    {
        let path = path.as_ref();
        let data = fs::read(path)?;

        let filename = path.file_name().map(|n| n.to_string_lossy().into_owned());

        Ok(TranscriptionRequestBuilder {
            model: self.model,
            data: Provided(data),
            filename: filename.or(self.filename),
            language: self.language,
            prompt: self.prompt,
            temperature: self.temperature,
            additional_params: self.additional_params,
        })
    }

    /// Sets the output language for the transcription request
    pub fn language(mut self, language: String) -> Self {
        self.language = Some(language);
        self
    }

    /// Sets the prompt to be sent in the transcription request
    pub fn prompt(mut self, prompt: String) -> Self {
        self.prompt = Some(prompt);
        self
    }

    /// Set the temperature to be sent in the transcription request
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Merges provider-specific parameters with existing parameters.
    pub fn additional_params(mut self, additional_params: serde_json::Value) -> Self {
        match self.additional_params {
            Some(params) => {
                self.additional_params = Some(json_utils::merge(params, additional_params));
            }
            None => {
                self.additional_params = Some(additional_params);
            }
        }
        self
    }

    /// Replaces provider-specific parameters, or clears them with `None`.
    pub fn additional_params_opt(mut self, additional_params: Option<serde_json::Value>) -> Self {
        self.additional_params = additional_params;
        self
    }
}

/// Request construction and dispatch after audio data has been supplied.
impl<M> TranscriptionRequestBuilder<M, Provided<Vec<u8>>>
where
    M: TranscriptionModel,
{
    /// Builds the transcription request
    pub fn build(self) -> TranscriptionRequest {
        self.into_parts().1
    }

    fn into_parts(self) -> (M, TranscriptionRequest) {
        let Self {
            model,
            data,
            filename,
            language,
            prompt,
            temperature,
            additional_params,
        } = self;
        (
            model,
            TranscriptionRequest {
                data: data.0,
                filename: filename.unwrap_or_else(|| "file".to_string()),
                language,
                prompt,
                temperature,
                additional_params,
            },
        )
    }

    /// Sends the transcription request to the transcription model provider and returns the transcription response
    pub async fn send(self) -> Result<TranscriptionResponse, ProviderError> {
        let (model, request) = self.into_parts();
        model.transcription(request).await
    }
}

#[cfg(test)]
mod provider_response_tests;
