//! This module provides functionality for working with audio transcription models.
//! It provides traits, structs, and enums for generating audio transcription requests,
//! handling transcription responses, and defining transcription models.
use crate::completion::{ResponseIdentity, Usage};
use crate::json_utils;
use crate::markers::{Missing, Provided};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use serde::{Deserialize, Serialize};
use std::io;
use std::sync::Arc;
use std::{fs, path::Path};

crate::provider_response::provider_error_enum!(
    TranscriptionError, "transcription" {
        #[cfg(not(target_family = "wasm"))]
        /// Error building the transcription request
        #[error("RequestError: {0}")]
        RequestError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

        #[cfg(target_family = "wasm")]
        /// Error building the transcription request
        #[error("RequestError: {0}")]
        RequestError(#[from] Box<dyn std::error::Error + 'static>),
    }
);

/// The normalized transcription response: the transcript plus the metadata
/// every provider can report, attributed to the provider that produced it.
///
/// This type is concrete — it carries no provider type parameter — so the
/// provider does not leak into [`TranscriptionRequestBuilder`] or into any
/// caller holding a [`TranscriptionModel`]. The provider's own payload stays
/// reachable two ways: a model's inherent `raw_transcription` method performs
/// the same request and returns the provider's native type, and
/// [`TranscriptionResponse::raw`] carries that value serialized.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    /// The transcribed text.
    pub text: String,
    /// Token or duration usage as the provider reported it. Zero-valued when
    /// the provider reported none — the same sentinel [`Usage`] documents for
    /// completions. Duration-billed endpoints report no token counts.
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
    /// The provider's transport-level request identifier, taken from the HTTP
    /// response headers (OpenAI `x-request-id`, Mistral
    /// `mistral-correlation-id`) — the id provider support asks for. `None`
    /// means the provider reported none; that is a documented outcome, never
    /// an error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// The provider's own response for this call: the value the model's
    /// inherent `raw_transcription` would have returned, serialized. Every
    /// provider seam populates it. `Value::Null` means the value was built
    /// without a provider behind it ([`TranscriptionResponse::new`] in a test
    /// double), never that the provider sent nothing.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub raw: serde_json::Value,
}

impl TranscriptionResponse {
    /// Create a response from its required parts; optional metadata starts
    /// unset and is filled in with the `with_*` helpers.
    pub fn new(text: impl Into<String>, provider: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            usage: Usage::new(),
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

/// Convert a provider's own transcription payload into the normalized
/// [`TranscriptionResponse`].
///
/// The provider descriptor name is an *input*, never something the conversion
/// knows: the OpenAI transcription wire shape is shared by several providers,
/// and a conversion that hardcoded a name would mislabel every provider but
/// one. This is a trait rather than `TryFrom<(&str, T)>` so that a provider
/// extension outside `rig-core` can implement it on its own response type —
/// a tuple is not a local type, and the orphan rule would reject the `TryFrom`
/// form anywhere but here.
pub trait NormalizeTranscriptionResponse {
    /// Normalize this payload, attributing it to `provider`.
    fn normalize(self, provider: &str) -> Result<TranscriptionResponse, TranscriptionError>;
}

/// Trait defining a transcription model that can be used to generate transcription requests.
/// This trait is meant to be implemented by the user to define a custom transcription model,
/// either from a third-party provider (e.g: OpenAI) or a local model.
///
/// The trait describes only what a model *does*: it has no associated types.
/// Construction lives on [`crate::client::transcription::TranscriptionClient`], and
/// `Clone` is required only by [`TranscriptionModel::transcription_request`],
/// which needs to hand the builder its own copy. A model behind an `Arc` is a
/// model: the trait is implemented for `Arc<M>` by forwarding.
pub trait TranscriptionModel: WasmCompatSend + WasmCompatSync {
    /// Generates a completion response for the given transcription model
    fn transcription(
        &self,
        request: TranscriptionRequest,
    ) -> impl std::future::Future<Output = Result<TranscriptionResponse, TranscriptionError>>
    + WasmCompatSend;

    /// Generates a transcription request builder for the given `file`
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
    ) -> impl std::future::Future<Output = Result<TranscriptionResponse, TranscriptionError>>
    + WasmCompatSend {
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

/// Builder struct for a transcription request
///
/// Example usage:
/// ```ignore
/// use rig_core::{
///     prelude::TranscriptionClient,
///     providers::openai::{Client, self},
///     transcription::{TranscriptionModel, TranscriptionRequestBuilder},
/// };
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let openai = Client::new("your-openai-api-key")?;
/// let model = openai.transcription_model(openai::WHISPER_1);
///
/// // Create the transcription request and execute it separately.
/// let request = TranscriptionRequestBuilder::new(model.clone())
///     .data(vec![0; 16])
///     .filename(Some("audio.mp3".to_string()))
///     .temperature(0.5)
///     .build();
///
/// let response = model.transcription(request).await?;
/// # Ok(())
/// # }
/// ```
///
/// Alternatively, you can execute the transcription request directly from the builder:
/// ```ignore
/// use rig_core::{
///     prelude::TranscriptionClient,
///     providers::openai::{Client, self},
///     transcription::TranscriptionRequestBuilder,
/// };
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let openai = Client::new("your-openai-api-key")?;
/// let model = openai.transcription_model(openai::WHISPER_1);
///
/// // Create the transcription request and execute it directly.
/// let response = TranscriptionRequestBuilder::new(model)
///     .data(vec![0; 16])
///     .filename(Some("audio.mp3".to_string()))
///     .temperature(0.5)
///     .send()
///     .await?;
/// # Ok(())
/// # }
/// ```
///
/// Note: It is usually unnecessary to create a completion request builder directly.
/// Instead, use the [TranscriptionModel::transcription_request] method.
pub struct TranscriptionRequestBuilder<M, D> {
    model: M,
    data: D, // starts Missing, becomes Provided<Vec<u8>> after data is set or load_file is called
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

    /// Sets the data for the request and transitions the builder to the next state where data is provided.
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

    /// Load the specified file into data and transitions the builder to the next state where data is provided.
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
}

/// The build and send methods are only available when data is provided, ensuring that the request cannot be sent without the required data.
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
    pub async fn send(self) -> Result<TranscriptionResponse, TranscriptionError> {
        let (model, request) = self.into_parts();
        model.transcription(request).await
    }
}

#[cfg(test)]
mod provider_response_tests;
