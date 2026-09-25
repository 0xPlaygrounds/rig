//! Audio transcription requests, normalized responses, and model interfaces.
//!
//! ```no_run
//! use rig_core::driver::{Model, Transport};
//! use rig_core::operation::Transcription;
//! use rig_core::transcription::TranscriptionRequestBuilder;
//! use rig_core::wire::Wire;
//!
//! # async fn example<W, T>(model: Model<W, T>) -> Result<(), Box<dyn std::error::Error>>
//! # where W: Wire<Op = Transcription>, T: Transport<W> {
//! let request = TranscriptionRequestBuilder::from_file("audio.wav")?.build();
//! let response = model.call(request).await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```
use crate::completion::{ResponseIdentity, Usage};
use crate::error::ProviderError;
use crate::json_utils;
use serde::{Deserialize, Serialize};
use std::io;
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

/// Builds a transcription request over the supplied audio. The audio is not
/// validated for format or nonemptiness.
pub struct TranscriptionRequestBuilder {
    request: TranscriptionRequest,
}

impl TranscriptionRequestBuilder {
    /// A request over `data`, named `"file"` until [`Self::filename`] names it.
    pub fn new(data: Vec<u8>) -> Self {
        Self {
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
    pub fn from_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let filename = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned());
        Ok(Self::new(fs::read(path)?).filename(filename))
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

#[cfg(test)]
mod builder_tests;
#[cfg(test)]
mod provider_response_tests;
