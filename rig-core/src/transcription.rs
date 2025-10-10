//! This module provides functionality for working with audio transcription models.
//! It provides traits, structs, and enums for generating audio transcription requests,
//! handling transcription responses, and defining transcription models.

use crate::client::transcription::TranscriptionModelHandle;
use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};
use crate::{http_client, json_utils};
use std::sync::Arc;
use std::{fs, path::Path};
use thiserror::Error;

// Errors
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
}

/// Trait defining a low-level LLM transcription interface
pub trait Transcription<M>
where
    M: TranscriptionModel,
{
    /// Generates a transcription request builder for the given `file`.
    /// This function is meant to be called by the user to further customize the
    /// request at transcription time before sending it.
    ///
    /// ❗IMPORTANT: The type that implements this trait might have already
    /// populated fields in the builder (the exact fields depend on the type).
    /// For fields that have already been set by the model, calling the corresponding
    /// method on the builder will overwrite the value set by the model.
    fn transcription(
        &self,
        filename: &str,
        data: &[u8],
    ) -> impl std::future::Future<
        Output = Result<TranscriptionRequestBuilder<M>, TranscriptionError>,
    > + WasmCompatSend;
}

/// General transcription response struct that contains the transcription text
/// and the raw response.
pub struct TranscriptionResponse<T> {
    pub text: String,
    pub response: T,
}

/// Trait defining a transcription model that can be used to generate transcription requests.
/// This trait is meant to be implemented by the user to define a custom transcription model,
/// either from a third-party provider (e.g: OpenAI) or a local model.
pub trait TranscriptionModel: Clone + WasmCompatSend + WasmCompatSync {
    /// The raw response type returned by the underlying model.
    type Response: WasmCompatSend + WasmCompatSync;

    /// Generates a completion response for the given transcription model
    fn transcription(
        &self,
        request: TranscriptionRequest,
    ) -> impl std::future::Future<
        Output = Result<TranscriptionResponse<Self::Response>, TranscriptionError>,
    > + WasmCompatSend;

    /// Generates a transcription request builder for the given `file`
    fn transcription_request(&self) -> TranscriptionRequestBuilder<Self> {
        TranscriptionRequestBuilder::new(self.clone())
    }
}

pub trait TranscriptionModelDyn: WasmCompatSend + WasmCompatSync {
    fn transcription(
        &self,
        request: TranscriptionRequest,
    ) -> WasmBoxedFuture<'_, Result<TranscriptionResponse<()>, TranscriptionError>>;

    fn transcription_request(&self) -> TranscriptionRequestBuilder<TranscriptionModelHandle<'_>>;
}

impl<T> TranscriptionModelDyn for T
where
    T: TranscriptionModel,
{
    fn transcription(
        &self,
        request: TranscriptionRequest,
    ) -> WasmBoxedFuture<'_, Result<TranscriptionResponse<()>, TranscriptionError>> {
        Box::pin(async move {
            let resp = self.transcription(request).await?;

            Ok(TranscriptionResponse {
                text: resp.text,
                response: (),
            })
        })
    }

    fn transcription_request(&self) -> TranscriptionRequestBuilder<TranscriptionModelHandle<'_>> {
        TranscriptionRequestBuilder::new(TranscriptionModelHandle {
            inner: Arc::new(self.clone()),
        })
    }
}

/// Struct representing a general transcription request that can be sent to a transcription model provider.
pub struct TranscriptionRequest {
    /// The file data to be sent to the transcription model provider
    pub data: Vec<u8>,
    /// The file name to be used in the request
    pub filename: String,
    /// The language used in the response from the transcription model provider
    pub language: String,
    /// The prompt to be sent to the transcription model provider
    pub prompt: Option<String>,
    /// The temperature sent to the transcription model provider
    pub temperature: Option<f64>,
    /// Additional parameters to be sent to the transcription model provider
    pub additional_params: Option<serde_json::Value>,
}

/// Builder struct for a transcription request
///
/// Example usage:
/// ```rust
/// use rig::{
///     providers::openai::{Client, self},
///     transcription::TranscriptionRequestBuilder,
/// };
///
/// let openai = Client::new("your-openai-api-key");
/// let model = openai.transcription_model(openai::WHISPER_1).build();
///
/// // Create the completion request and execute it separately
/// let request = TranscriptionRequestBuilder::new(model, "~/audio.mp3".to_string())
///     .temperature(0.5)
///     .build();
///
/// let response = model.transcription(request)
///     .await
///     .expect("Failed to get transcription response");
/// ```
///
/// Alternatively, you can execute the transcription request directly from the builder:
/// ```rust
/// use rig::{
///     providers::openai::{Client, self},
///     transcription::TranscriptionRequestBuilder,
/// };
///
/// let openai = Client::new("your-openai-api-key");
/// let model = openai.transcription_model(openai::WHISPER_1).build();
///
/// // Create the completion request and execute it directly
/// let response = TranscriptionRequestBuilder::new(model, "~/audio.mp3".to_string())
///     .temperature(0.5)
///     .send()
///     .await
///     .expect("Failed to get transcription response");
/// ```
///
/// Note: It is usually unnecessary to create a completion request builder directly.
/// Instead, use the [TranscriptionModel::transcription_request] method.
pub struct TranscriptionRequestBuilder<M>
where
    M: TranscriptionModel,
{
    model: M,
    data: Vec<u8>,
    filename: Option<String>,
    language: String,
    prompt: Option<String>,
    temperature: Option<f64>,
    additional_params: Option<serde_json::Value>,
}

impl<M> TranscriptionRequestBuilder<M>
where
    M: TranscriptionModel,
{
    pub fn new(model: M) -> Self {
        TranscriptionRequestBuilder {
            model,
            data: vec![],
            filename: None,
            language: "en".to_string(),
            prompt: None,
            temperature: None,
            additional_params: None,
        }
    }

    pub fn filename(mut self, filename: Option<String>) -> Self {
        self.filename = filename;
        self
    }

    /// Sets the data for the request
    pub fn data(mut self, data: Vec<u8>) -> Self {
        self.data = data;
        self
    }

    /// Load the specified file into data
    pub fn load_file<P>(self, path: P) -> Self
    where
        P: AsRef<Path>,
    {
        let path = path.as_ref();
        let data = fs::read(path).expect("Failed to load audio file, file did not exist");

        self.filename(Some(
            path.file_name()
                .expect("Path was not a file")
                .to_str()
                .expect("Failed to convert filename to ascii")
                .to_string(),
        ))
        .data(data)
    }

    /// Sets the output language for the transcription request
    pub fn language(mut self, language: String) -> Self {
        self.language = language;
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

    /// Adds additional parameters to the transcription request.
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

    /// Sets the additional parameters for the transcription request.
    pub fn additional_params_opt(mut self, additional_params: Option<serde_json::Value>) -> Self {
        self.additional_params = additional_params;
        self
    }

    /// Builds the transcription request
    /// Panics if data is empty.
    pub fn build(self) -> TranscriptionRequest {
        if self.data.is_empty() {
            panic!("Data cannot be empty!")
        }

        TranscriptionRequest {
            data: self.data,
            filename: self.filename.unwrap_or("file".to_string()),
            language: self.language,
            prompt: self.prompt,
            temperature: self.temperature,
            additional_params: self.additional_params,
        }
    }

    /// Sends the transcription request to the transcription model provider and returns the transcription response
    pub async fn send(self) -> Result<TranscriptionResponse<M::Response>, TranscriptionError> {
        let model = self.model.clone();

        model.transcription(self.build()).await
    }
}
