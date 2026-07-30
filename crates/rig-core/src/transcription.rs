//! This module provides functionality for working with audio transcription models.
//! It provides the structs and enums used to build audio transcription requests and
//! to represent transcription responses and errors. Providers expose the actual call
//! as a `functions::transcribe` free function over these types.
use crate::{http_client, json_utils, provider_response};
use std::io;
use std::{fs, path::Path};
use thiserror::Error;

// Errors
/// Errors returned by transcription models.
///
/// Inspect provider failures with [`Self::provider_response_body`],
/// [`Self::provider_response_json`], and [`Self::provider_response_status`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TranscriptionError {
    /// Http error (e.g.: connection error, timeout, etc.)
    #[error("HttpError: {0}")]
    HttpError(#[from] http_client::Error),

    /// Json error (e.g.: serialization, deserialization)
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    #[cfg(not(target_family = "wasm"))]
    /// Error building the transcription request
    #[error("RequestError: {0}")]
    RequestError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

    #[cfg(target_family = "wasm")]
    /// Error building the transcription request
    #[error("RequestError: {0}")]
    RequestError(#[from] Box<dyn std::error::Error + 'static>),

    /// Error parsing the transcription response
    #[error("ResponseError: {0}")]
    ResponseError(String),

    /// Error returned by the transcription model provider
    #[error("ProviderError: {0}")]
    ProviderError(String),

    /// Raw error response preserved from the transcription model provider
    #[error("ProviderResponseError: {0}")]
    ProviderResponse(provider_response::ProviderResponseError),
}

crate::provider_response::impl_provider_response_helpers!(TranscriptionError);

/// General transcription response struct that contains the transcription text
/// and the raw response.
pub struct TranscriptionResponse<T> {
    pub text: String,
    pub response: T,
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

impl TranscriptionRequest {
    /// Creates a request from the audio bytes, with the default filename `file`.
    ///
    /// Refine with the `with_*` methods, then execute it with the provider's
    /// `functions::transcribe`:
    ///
    /// ```no_run
    /// use rig_core::{
    ///     http_runtime::HttpRuntime,
    ///     providers::openai,
    ///     transcription::TranscriptionRequest,
    /// };
    ///
    /// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let cfg = openai::functions::Config::from_env(openai::WHISPER_1)?;
    /// let rt = HttpRuntime::new();
    ///
    /// let request = TranscriptionRequest::new(vec![0; 16])
    ///     .with_filename("audio.mp3")
    ///     .with_temperature(0.5);
    ///
    /// let response = openai::functions::transcribe(&cfg, &rt, request).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            filename: "file".to_string(),
            language: None,
            prompt: None,
            temperature: None,
            additional_params: None,
        }
    }

    /// Reads `path` into a request, taking the filename from the path.
    ///
    /// Falls back to the default filename `file` when the path has no final component.
    pub fn from_file<P>(path: P) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        let path = path.as_ref();
        let data = fs::read(path)?;

        let mut request = Self::new(data);
        if let Some(filename) = path.file_name().map(|n| n.to_string_lossy().into_owned()) {
            request.filename = filename;
        }

        Ok(request)
    }

    /// Sets the file name to be used in the request.
    pub fn with_filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = filename.into();
        self
    }

    /// Sets the output language for the transcription request.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Sets the prompt to be sent in the transcription request.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Sets the temperature to be sent in the transcription request.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Merges additional parameters into the transcription request.
    ///
    /// Existing parameters are merged with (not replaced by) `additional_params`; use
    /// [`Self::with_additional_params_opt`] to replace them outright.
    pub fn with_additional_params(mut self, additional_params: serde_json::Value) -> Self {
        self.additional_params = Some(match self.additional_params {
            Some(params) => json_utils::merge(params, additional_params),
            None => additional_params,
        });
        self
    }

    /// Replaces the additional parameters for the transcription request.
    pub fn with_additional_params_opt(
        mut self,
        additional_params: Option<serde_json::Value>,
    ) -> Self {
        self.additional_params = additional_params;
        self
    }
}

#[cfg(test)]
mod provider_response_tests {
    use super::*;
    use http::StatusCode;

    #[test]
    fn transcription_error_provider_response_helpers_with_preserved_json_body() {
        let body = r#"{"error":{"message":"rate limited"}}"#;
        let error =
            TranscriptionError::ProviderResponse(provider_response::ProviderResponseError {
                status: None,
                body: body.to_string(),
            });

        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(
            error.provider_response_json().expect("valid JSON"),
            Some(serde_json::json!({ "error": { "message": "rate limited" } }))
        );
    }

    #[test]
    fn transcription_error_provider_response_helpers_with_http_non_success() {
        let body = r#"{"error":{"message":"bad request"}}"#;
        let error =
            TranscriptionError::HttpError(http_client::Error::InvalidStatusCodeWithMessage(
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
    fn transcription_error_provider_response_helpers_with_preserved_plain_text_body() {
        let error =
            TranscriptionError::ProviderResponse(provider_response::ProviderResponseError {
                status: None,
                body: "not json".to_string(),
            });

        assert_eq!(error.provider_response_body(), Some("not json"));
        assert!(error.provider_response_json().is_err());
    }

    #[test]
    fn transcription_error_provider_error_is_not_a_provider_response() {
        let error = TranscriptionError::ProviderError("internal diagnostic".to_string());

        assert_eq!(error.provider_response_body(), None);
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(error.provider_response_json().expect("no body"), None);
    }

    #[test]
    fn transcription_error_provider_response_helpers_with_unrelated_variant() {
        let error = TranscriptionError::ResponseError("parse failed".to_string());

        assert_eq!(error.provider_response_body(), None);
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(error.provider_response_json().expect("no body"), None);
    }
}
