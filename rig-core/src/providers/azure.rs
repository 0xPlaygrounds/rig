//! Azure OpenAI API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::azure;
//!
//! let client = azure::Client::new("YOUR_API_KEY", "YOUR_API_VERSION", "YOUR_ENDPOINT");
//!
//! let gpt4o = client.completion_model(azure::GPT_4O);
//! ```

use std::fmt::Debug;

use super::openai::{TranscriptionResponse, send_compatible_streaming_request};
#[cfg(feature = "image")]
use crate::client::Nothing;
use crate::client::{
    self, ApiKey, Capabilities, Capable, DebugExt, Provider, ProviderBuilder, ProviderClient,
};
use crate::completion::GetTokenUsage;
use crate::http_client::multipart::Part;
use crate::http_client::{self, HttpClientExt, MultipartForm, bearer_auth_header};
use crate::streaming::StreamingCompletionResponse;
use crate::transcription::TranscriptionError;
use crate::{
    completion::{self, CompletionError, CompletionRequest},
    embeddings::{self, EmbeddingError},
    json_utils,
    providers::openai,
    telemetry::SpanCombinator,
    transcription::{self},
};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::json;
// ================================================================
// Main Azure OpenAI Client
// ================================================================

const DEFAULT_API_VERSION: &str = "2024-10-21";

#[derive(Debug, Clone)]
pub struct AzureExt {
    endpoint: String,
    api_version: String,
}

impl DebugExt for AzureExt {
    fn fields(&self) -> impl Iterator<Item = (&'static str, &dyn std::fmt::Debug)> {
        [
            ("endpoint", (&self.endpoint as &dyn Debug)),
            ("api_version", (&self.api_version as &dyn Debug)),
        ]
        .into_iter()
    }
}

// TODO: @FayCarsons - this should be a type-safe builder,
// but that would require extending the `ProviderBuilder`
// to have some notion of complete vs incomplete states in a
// given extension builder
#[derive(Debug, Clone)]
pub struct AzureExtBuilder {
    endpoint: Option<String>,
    api_version: String,
}

impl Default for AzureExtBuilder {
    fn default() -> Self {
        Self {
            endpoint: None,
            api_version: DEFAULT_API_VERSION.into(),
        }
    }
}

pub type Client<H = reqwest::Client> = client::Client<AzureExt, H>;
pub type ClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<AzureExtBuilder, AzureOpenAIAuth, H>;

impl Provider for AzureExt {
    type Builder = AzureExtBuilder;

    /// Verifying Azure auth without consuming tokens is not supported
    const VERIFY_PATH: &'static str = "";

    fn build<H>(
        builder: &client::ClientBuilder<
            Self::Builder,
            <Self::Builder as ProviderBuilder>::ApiKey,
            H,
        >,
    ) -> http_client::Result<Self> {
        let AzureExtBuilder {
            endpoint,
            api_version,
            ..
        } = builder.ext().clone();

        match endpoint {
            Some(endpoint) => Ok(Self {
                endpoint,
                api_version,
            }),
            None => Err(http_client::Error::Instance(
                "Azure client must be provided an endpoint prior to building".into(),
            )),
        }
    }
}

impl<H> Capabilities<H> for AzureExt {
    type Completion = Capable<CompletionModel<H>>;
    type Embeddings = Capable<EmbeddingModel<H>>;
    type Transcription = Capable<TranscriptionModel<H>>;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Capable<AudioGenerationModel<H>>;
}

impl ProviderBuilder for AzureExtBuilder {
    type Output = AzureExt;
    type ApiKey = AzureOpenAIAuth;

    const BASE_URL: &'static str = "";

    fn finish<H>(
        &self,
        mut builder: client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<client::ClientBuilder<Self, Self::ApiKey, H>> {
        use AzureOpenAIAuth::*;

        let auth = builder.get_api_key().clone();

        match auth {
            Token(token) => bearer_auth_header(builder.headers_mut(), token.as_str())?,
            ApiKey(key) => {
                let k = http::HeaderName::from_static("api-key");
                let v = http::HeaderValue::from_str(key.as_str())?;

                builder.headers_mut().insert(k, v);
            }
        }

        Ok(builder)
    }
}

impl<H> ClientBuilder<H> {
    /// API version to use (e.g., "2024-10-21" for GA, "2024-10-01-preview" for preview)
    pub fn api_version(mut self, api_version: &str) -> Self {
        self.ext_mut().api_version = api_version.into();

        self
    }
}

impl<H> client::ClientBuilder<AzureExtBuilder, AzureOpenAIAuth, H> {
    /// Azure OpenAI endpoint URL, for example: https://{your-resource-name}.openai.azure.com
    pub fn azure_endpoint(self, endpoint: String) -> ClientBuilder<H> {
        self.over_ext(|AzureExtBuilder { api_version, .. }| AzureExtBuilder {
            endpoint: Some(endpoint),
            api_version,
        })
    }
}

#[derive(Clone)]
pub enum AzureOpenAIAuth {
    ApiKey(String),
    Token(String),
}

impl ApiKey for AzureOpenAIAuth {}

impl std::fmt::Debug for AzureOpenAIAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiKey(_) => write!(f, "API key <REDACTED>"),
            Self::Token(_) => write!(f, "Token <REDACTED>"),
        }
    }
}

impl<S> From<S> for AzureOpenAIAuth
where
    S: Into<String>,
{
    fn from(token: S) -> Self {
        AzureOpenAIAuth::Token(token.into())
    }
}

impl<T> Client<T>
where
    T: HttpClientExt,
{
    fn endpoint(&self) -> &str {
        &self.ext().endpoint
    }

    fn api_version(&self) -> &str {
        &self.ext().api_version
    }

    fn post_embedding(&self, deployment_id: &str) -> http_client::Result<http_client::Builder> {
        let url = format!(
            "{}/openai/deployments/{}/embeddings?api-version={}",
            self.endpoint(),
            deployment_id.trim_start_matches('/'),
            self.api_version()
        );

        self.post(&url)
    }

    #[cfg(feature = "audio")]
    fn post_audio_generation(
        &self,
        deployment_id: &str,
    ) -> http_client::Result<http_client::Builder> {
        let url = format!(
            "{}/openai/deployments/{}/audio/speech?api-version={}",
            self.endpoint(),
            deployment_id.trim_start_matches('/'),
            self.api_version()
        );

        self.post(url)
    }

    fn post_chat_completion(
        &self,
        deployment_id: &str,
    ) -> http_client::Result<http_client::Builder> {
        let url = format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.endpoint(),
            deployment_id.trim_start_matches('/'),
            self.api_version()
        );

        self.post(&url)
    }

    fn post_transcription(&self, deployment_id: &str) -> http_client::Result<http_client::Builder> {
        let url = format!(
            "{}/openai/deployments/{}/audio/translations?api-version={}",
            self.endpoint(),
            deployment_id.trim_start_matches('/'),
            self.api_version()
        );

        self.post(&url)
    }

    #[cfg(feature = "image")]
    fn post_image_generation(
        &self,
        deployment_id: &str,
    ) -> http_client::Result<http_client::Builder> {
        let url = format!(
            "{}/openai/deployments/{}/images/generations?api-version={}",
            self.endpoint(),
            deployment_id.trim_start_matches('/'),
            self.api_version()
        );

        self.post(&url)
    }
}

pub struct AzureOpenAIClientParams {
    api_key: String,
    version: String,
    header: String,
}

impl ProviderClient for Client {
    type Input = AzureOpenAIClientParams;

    /// Create a new Azure OpenAI client from the `AZURE_API_KEY` or `AZURE_TOKEN`, `AZURE_API_VERSION`, and `AZURE_ENDPOINT` environment variables.
    fn from_env() -> Self {
        let auth = if let Ok(api_key) = std::env::var("AZURE_API_KEY") {
            AzureOpenAIAuth::ApiKey(api_key)
        } else if let Ok(token) = std::env::var("AZURE_TOKEN") {
            AzureOpenAIAuth::Token(token)
        } else {
            panic!("Neither AZURE_API_KEY nor AZURE_TOKEN is set");
        };

        let api_version = std::env::var("AZURE_API_VERSION").expect("AZURE_API_VERSION not set");
        let azure_endpoint = std::env::var("AZURE_ENDPOINT").expect("AZURE_ENDPOINT not set");

        Self::builder()
            .api_key(auth)
            .azure_endpoint(azure_endpoint)
            .api_version(&api_version)
            .build()
            .unwrap()
    }

    fn from_val(
        AzureOpenAIClientParams {
            api_key,
            version,
            header,
        }: Self::Input,
    ) -> Self {
        let auth = AzureOpenAIAuth::ApiKey(api_key.to_string());

        Self::builder()
            .api_key(auth)
            .azure_endpoint(header)
            .api_version(&version)
            .build()
            .unwrap()
    }
}

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

// ================================================================
// Azure OpenAI Embedding API
// ================================================================

/// `text-embedding-3-large` embedding model
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
/// `text-embedding-3-small` embedding model
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// `text-embedding-ada-002` embedding model
pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

fn model_dimensions_from_identifier(identifier: &str) -> Option<usize> {
    match identifier {
        TEXT_EMBEDDING_3_LARGE => Some(3_072),
        TEXT_EMBEDDING_3_SMALL | TEXT_EMBEDDING_ADA_002 => Some(1_536),
        _ => None,
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

impl From<ApiErrorResponse> for EmbeddingError {
    fn from(err: ApiErrorResponse) -> Self {
        EmbeddingError::ProviderError(err.message)
    }
}

impl From<ApiResponse<EmbeddingResponse>> for Result<EmbeddingResponse, EmbeddingError> {
    fn from(value: ApiResponse<EmbeddingResponse>) -> Self {
        match value {
            ApiResponse::Ok(response) => Ok(response),
            ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

impl GetTokenUsage for Usage {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();

        usage.input_tokens = self.prompt_tokens as u64;
        usage.total_tokens = self.total_tokens as u64;
        usage.output_tokens = usage.total_tokens - usage.input_tokens;

        Some(usage)
    }
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {} Total tokens: {}",
            self.prompt_tokens, self.total_tokens
        )
    }
}

#[derive(Clone)]
pub struct EmbeddingModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
    ndims: usize,
}

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
    T: HttpClientExt + Default + Clone + 'static,
{
    const MAX_DOCUMENTS: usize = 1024;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self {
        Self::new(client.clone(), model, dims)
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents = documents.into_iter().collect::<Vec<_>>();

        let body = serde_json::to_vec(&json!({
            "input": documents,
        }))?;

        let req = self
            .client
            .post_embedding(self.model.as_str())?
            .body(body)
            .map_err(|e| EmbeddingError::HttpError(e.into()))?;

        let response = self.client.send(req).await?;

        if response.status().is_success() {
            let body: Vec<u8> = response.into_body().await?;
            let body: ApiResponse<EmbeddingResponse> = serde_json::from_slice(&body)?;

            match body {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "Azure embedding token usage: {}",
                        response.usage
                    );

                    if response.data.len() != documents.len() {
                        return Err(EmbeddingError::ResponseError(
                            "Response data length does not match input length".into(),
                        ));
                    }

                    Ok(response
                        .data
                        .into_iter()
                        .zip(documents.into_iter())
                        .map(|(embedding, document)| embeddings::Embedding {
                            document,
                            vec: embedding.embedding,
                        })
                        .collect())
                }
                ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
            }
        } else {
            let text = http_client::text(response).await?;
            Err(EmbeddingError::ProviderError(text))
        }
    }
}

impl<T> EmbeddingModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>, ndims: Option<usize>) -> Self {
        let model = model.into();
        let ndims = ndims
            .or(model_dimensions_from_identifier(&model))
            .unwrap_or_default();

        Self {
            client,
            model,
            ndims,
        }
    }

    pub fn with_model(client: Client<T>, model: &str, ndims: Option<usize>) -> Self {
        let ndims = ndims.unwrap_or_default();

        Self {
            client,
            model: model.into(),
            ndims,
        }
    }
}

// ================================================================
// Azure OpenAI Completion API
// ================================================================

/// `o1` completion model
pub const O1: &str = "o1";
/// `o1-preview` completion model
pub const O1_PREVIEW: &str = "o1-preview";
/// `o1-mini` completion model
pub const O1_MINI: &str = "o1-mini";
/// `gpt-4o` completion model
pub const GPT_4O: &str = "gpt-4o";
/// `gpt-4o-mini` completion model
pub const GPT_4O_MINI: &str = "gpt-4o-mini";
/// `gpt-4o-realtime-preview` completion model
pub const GPT_4O_REALTIME_PREVIEW: &str = "gpt-4o-realtime-preview";
/// `gpt-4-turbo` completion model
pub const GPT_4_TURBO: &str = "gpt-4";
/// `gpt-4` completion model
pub const GPT_4: &str = "gpt-4";
/// `gpt-4-32k` completion model
pub const GPT_4_32K: &str = "gpt-4-32k";
/// `gpt-4-32k` completion model
pub const GPT_4_32K_0613: &str = "gpt-4-32k";
/// `gpt-3.5-turbo` completion model
pub const GPT_35_TURBO: &str = "gpt-3.5-turbo";
/// `gpt-3.5-turbo-instruct` completion model
pub const GPT_35_TURBO_INSTRUCT: &str = "gpt-3.5-turbo-instruct";
/// `gpt-3.5-turbo-16k` completion model
pub const GPT_35_TURBO_16K: &str = "gpt-3.5-turbo-16k";

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct AzureOpenAICompletionRequest {
    model: String,
    pub messages: Vec<openai::Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<openai::ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<crate::providers::openrouter::ToolChoice>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

impl TryFrom<(&str, CompletionRequest)> for AzureOpenAICompletionRequest {
    type Error = CompletionError;

    fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        //FIXME: Must fix!
        if req.tool_choice.is_some() {
            tracing::warn!(
                "Tool choice is currently not supported in Azure OpenAI. This should be fixed by Rig 0.25."
            );
        }

        let mut full_history: Vec<openai::Message> = match &req.preamble {
            Some(preamble) => vec![openai::Message::system(preamble)],
            None => vec![],
        };

        if let Some(docs) = req.normalized_documents() {
            let docs: Vec<openai::Message> = docs.try_into()?;
            full_history.extend(docs);
        }

        let chat_history: Vec<openai::Message> = req
            .chat_history
            .clone()
            .into_iter()
            .map(|message| message.try_into())
            .collect::<Result<Vec<Vec<openai::Message>>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        full_history.extend(chat_history);

        let tool_choice = req
            .tool_choice
            .clone()
            .map(crate::providers::openrouter::ToolChoice::try_from)
            .transpose()?;

        Ok(Self {
            model: model.to_string(),
            messages: full_history,
            temperature: req.temperature,
            tools: req
                .tools
                .clone()
                .into_iter()
                .map(openai::ToolDefinition::from)
                .collect::<Vec<_>>(),
            tool_choice,
            additional_params: req.additional_params,
        })
    }
}

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    client: Client<T>,
    /// Name of the model (e.g.: gpt-4o-mini)
    pub model: String,
}

impl<T> CompletionModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type Response = openai::CompletionResponse;
    type StreamingResponse = openai::StreamingCompletionResponse;
    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model.into())
    }

    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<openai::CompletionResponse>, CompletionError> {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "azure.openai",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = &completion_request.preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let request =
            AzureOpenAICompletionRequest::try_from((self.model.as_ref(), completion_request))?;

        if enabled!(Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "Azure OpenAI completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post_chat_completion(&self.model)?
            .body(body)
            .map_err(http_client::Error::from)?;

        async move {
            let response = self.client.send::<_, Bytes>(req).await.unwrap();

            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if status.is_success() {
                match serde_json::from_slice::<ApiResponse<openai::CompletionResponse>>(
                    &response_body,
                )? {
                    ApiResponse::Ok(response) => {
                        let span = tracing::Span::current();
                        span.record_response_metadata(&response);
                        span.record_token_usage(&response.usage);
                        if enabled!(Level::TRACE) {
                            tracing::trace!(target: "rig::completions",
                                "Azure OpenAI completion response: {}",
                                serde_json::to_string_pretty(&response)?
                            );
                        }
                        response.try_into()
                    }
                    ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
                }
            } else {
                Err(CompletionError::ProviderError(
                    String::from_utf8_lossy(&response_body).to_string(),
                ))
            }
        }
        .instrument(span)
        .await
    }

    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let preamble = completion_request.preamble.clone();
        let mut request =
            AzureOpenAICompletionRequest::try_from((self.model.as_ref(), completion_request))?;

        let params = json_utils::merge(
            request.additional_params.unwrap_or(serde_json::json!({})),
            serde_json::json!({"stream": true, "stream_options": {"include_usage": true} }),
        );

        request.additional_params = Some(params);

        if enabled!(Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "Azure OpenAI completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post_chat_completion(&self.model)?
            .body(body)
            .map_err(http_client::Error::from)?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "azure.openai",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = &preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        tracing_futures::Instrument::instrument(
            send_compatible_streaming_request(self.client.clone(), req),
            span,
        )
        .await
    }
}

// ================================================================
// Azure OpenAI Transcription API
// ================================================================

#[derive(Clone)]
pub struct TranscriptionModel<T = reqwest::Client> {
    client: Client<T>,
    /// Name of the model (e.g.: gpt-3.5-turbo-1106)
    pub model: String,
}

impl<T> TranscriptionModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl<T> transcription::TranscriptionModel for TranscriptionModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    type Response = TranscriptionResponse;
    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn transcription(
        &self,
        request: transcription::TranscriptionRequest,
    ) -> Result<
        transcription::TranscriptionResponse<Self::Response>,
        transcription::TranscriptionError,
    > {
        let data = request.data;

        let mut body =
            MultipartForm::new().part(Part::bytes("file", data).filename(request.filename.clone()));

        if let Some(prompt) = request.prompt {
            body = body.text("prompt", prompt.clone());
        }

        if let Some(ref temperature) = request.temperature {
            body = body.text("temperature", temperature.to_string());
        }

        if let Some(ref additional_params) = request.additional_params {
            for (key, value) in additional_params
                .as_object()
                .expect("Additional Parameters to OpenAI Transcription should be a map")
            {
                body = body.text(key.to_owned(), value.to_string());
            }
        }

        let req = self
            .client
            .post_transcription(&self.model)?
            .body(body)
            .map_err(|e| TranscriptionError::HttpError(e.into()))?;

        let response = self.client.send_multipart::<Bytes>(req).await?;
        let status = response.status();
        let response_body = response.into_body().into_future().await?.to_vec();

        if status.is_success() {
            match serde_json::from_slice::<ApiResponse<TranscriptionResponse>>(&response_body)? {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(api_error_response) => Err(TranscriptionError::ProviderError(
                    api_error_response.message,
                )),
            }
        } else {
            Err(TranscriptionError::ProviderError(
                String::from_utf8_lossy(&response_body).to_string(),
            ))
        }
    }
}

// ================================================================
// Azure OpenAI Image Generation API
// ================================================================
#[cfg(feature = "image")]
pub use image_generation::*;
use tracing::{Instrument, Level, enabled, info_span};
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
mod image_generation {
    use crate::http_client::HttpClientExt;
    use crate::image_generation;
    use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
    use crate::providers::azure::{ApiResponse, Client};
    use crate::providers::openai::ImageGenerationResponse;
    use bytes::Bytes;
    use serde_json::json;

    #[derive(Clone)]
    pub struct ImageGenerationModel<T = reqwest::Client> {
        client: Client<T>,
        pub model: String,
    }

    impl<T> image_generation::ImageGenerationModel for ImageGenerationModel<T>
    where
        T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
    {
        type Response = ImageGenerationResponse;

        type Client = Client<T>;

        fn make(client: &Self::Client, model: impl Into<String>) -> Self {
            Self {
                client: client.clone(),
                model: model.into(),
            }
        }

        async fn image_generation(
            &self,
            generation_request: ImageGenerationRequest,
        ) -> Result<image_generation::ImageGenerationResponse<Self::Response>, ImageGenerationError>
        {
            let request = json!({
                "model": self.model,
                "prompt": generation_request.prompt,
                "size": format!("{}x{}", generation_request.width, generation_request.height),
                "response_format": "b64_json"
            });

            let body = serde_json::to_vec(&request)?;

            let req = self
                .client
                .post_image_generation(&self.model)?
                .body(body)
                .map_err(|e| ImageGenerationError::HttpError(e.into()))?;

            let response = self.client.send::<_, Bytes>(req).await?;
            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if !status.is_success() {
                return Err(ImageGenerationError::ProviderError(format!(
                    "{status}: {}",
                    String::from_utf8_lossy(&response_body)
                )));
            }

            match serde_json::from_slice::<ApiResponse<ImageGenerationResponse>>(&response_body)? {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(err) => Err(ImageGenerationError::ProviderError(err.message)),
            }
        }
    }
}
// ================================================================
// Azure OpenAI Audio Generation API
// ================================================================

#[cfg(feature = "audio")]
pub use audio_generation::*;

#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
mod audio_generation {
    use super::Client;
    use crate::audio_generation::{
        self, AudioGenerationError, AudioGenerationRequest, AudioGenerationResponse,
    };
    use crate::http_client::HttpClientExt;
    use bytes::Bytes;
    use serde_json::json;

    #[derive(Clone)]
    pub struct AudioGenerationModel<T = reqwest::Client> {
        client: Client<T>,
        model: String,
    }

    impl<T> AudioGenerationModel<T> {
        pub fn new(client: Client<T>, deployment_name: impl Into<String>) -> Self {
            Self {
                client,
                model: deployment_name.into(),
            }
        }
    }

    impl<T> audio_generation::AudioGenerationModel for AudioGenerationModel<T>
    where
        T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
    {
        type Response = Bytes;
        type Client = Client<T>;

        fn make(client: &Self::Client, model: impl Into<String>) -> Self {
            Self::new(client.clone(), model)
        }

        async fn audio_generation(
            &self,
            request: AudioGenerationRequest,
        ) -> Result<AudioGenerationResponse<Self::Response>, AudioGenerationError> {
            let request = json!({
                "model": self.model,
                "input": request.text,
                "voice": request.voice,
                "speed": request.speed,
            });

            let body = serde_json::to_vec(&request)?;

            let req = self
                .client
                .post_audio_generation("/audio/speech")?
                .header("Content-Type", "application/json")
                .body(body)
                .map_err(|e| AudioGenerationError::HttpError(e.into()))?;

            let response = self.client.send::<_, Bytes>(req).await?;
            let status = response.status();
            let response_body = response.into_body().into_future().await?;

            if !status.is_success() {
                return Err(AudioGenerationError::ProviderError(format!(
                    "{status}: {}",
                    String::from_utf8_lossy(&response_body)
                )));
            }

            Ok(AudioGenerationResponse {
                audio: response_body.to_vec(),
                response: response_body,
            })
        }
    }
}

#[cfg(test)]
mod azure_tests {
    use super::*;

    use crate::OneOrMany;
    use crate::client::{completion::CompletionClient, embeddings::EmbeddingsClient};
    use crate::completion::CompletionModel;
    use crate::embeddings::EmbeddingModel;

    #[tokio::test]
    #[ignore]
    async fn test_azure_embedding() {
        let _ = tracing_subscriber::fmt::try_init();

        let client = Client::<reqwest::Client>::from_env();
        let model = client.embedding_model(TEXT_EMBEDDING_3_SMALL);
        let embeddings = model
            .embed_texts(vec!["Hello, world!".to_string()])
            .await
            .unwrap();

        tracing::info!("Azure embedding: {:?}", embeddings);
    }

    #[tokio::test]
    #[ignore]
    async fn test_azure_completion() {
        let _ = tracing_subscriber::fmt::try_init();

        let client = Client::<reqwest::Client>::from_env();
        let model = client.completion_model(GPT_4O_MINI);
        let completion = model
            .completion(CompletionRequest {
                preamble: Some("You are a helpful assistant.".to_string()),
                chat_history: OneOrMany::one("Hello!".into()),
                documents: vec![],
                max_tokens: Some(100),
                temperature: Some(0.0),
                tools: vec![],
                tool_choice: None,
                additional_params: None,
            })
            .await
            .unwrap();

        tracing::info!("Azure completion: {:?}", completion);
    }
}
