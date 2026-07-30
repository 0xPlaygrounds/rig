//! Azure OpenAI API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig_core::providers::azure;
//! use rig_core::client::CompletionClient;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = azure::Client::builder()
//!     .api_key("test")
//!     .azure_endpoint("test".to_string()) // add your endpoint here!
//!     .build()?;
//!
//! let gpt4o = client.completion_model(azure::GPT_4O);
//! # Ok(())
//! # }
//! ```
//!
//! ## Authentication
//! The authentication type used for the `azure` module is [`AzureOpenAIAuth`].
//!
//! By default, using a type that implements `Into<String>` as the input for the client builder will turn the type into a bearer auth token.
//! If you want to use an API key, you need to use the type specifically.

use std::fmt::Debug;

use super::openai::TranscriptionResponse;
use crate::client::{
    self, ApiKey, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
    ProviderClient,
};
use crate::completion::CompletionError;
use crate::http_client::{self, HttpClientExt, bearer_auth_header};
use crate::transcription::TranscriptionError;
use crate::{
    embeddings::{self, EmbeddingError},
    providers::openai,
    transcription::{self},
};
use bytes::Bytes;
use serde::Deserialize;
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

impl AzureExt {
    /// Resource endpoint this client targets, e.g. `https://my-resource.openai.azure.com`.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// API version query parameter this client sends.
    pub fn api_version(&self) -> &str {
        &self.api_version
    }
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
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<AzureExtBuilder, AzureOpenAIAuth, H>;

impl Provider for AzureExt {
    type Builder = AzureExtBuilder;

    /// Verifying Azure auth without consuming tokens is not supported
    const VERIFY_PATH: &'static str = "";
}

impl<H> Capabilities<H> for AzureExt {
    type Completion = Capable<CompletionModel<H>>;
    type Embeddings = Capable<EmbeddingModel<H>>;
    type Transcription = Capable<TranscriptionModel<H>>;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Capable<AudioGenerationModel<H>>;
    type Rerank = Nothing;
}

impl ProviderBuilder for AzureExtBuilder {
    type Extension<H>
        = AzureExt
    where
        H: HttpClientExt;
    type ApiKey = AzureOpenAIAuth;

    const BASE_URL: &'static str = "";

    fn build<H>(
        builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        let AzureExtBuilder {
            endpoint,
            api_version,
            ..
        } = builder.ext().clone();

        match endpoint {
            Some(endpoint) => Ok(AzureExt {
                endpoint,
                api_version,
            }),
            None => Err(http_client::Error::Instance(
                "Azure client must be provided an endpoint prior to building".into(),
            )),
        }
    }

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

/// The authentication type for Azure OpenAI. Can either be an API key or a token.
/// String types will automatically be coerced to a bearer auth token by default.
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
    type Error = crate::client::ProviderClientError;

    /// Create a new Azure OpenAI client from the `AZURE_API_KEY` or `AZURE_TOKEN`, `AZURE_API_VERSION`, and `AZURE_ENDPOINT` environment variables.
    fn from_env() -> Result<Self, Self::Error> {
        let auth = if let Some(api_key) = crate::client::optional_env_var("AZURE_API_KEY")? {
            AzureOpenAIAuth::ApiKey(api_key)
        } else if let Some(token) = crate::client::optional_env_var("AZURE_TOKEN")? {
            AzureOpenAIAuth::Token(token)
        } else {
            return Err(crate::client::ProviderClientError::InvalidConfiguration(
                "either `AZURE_API_KEY` or `AZURE_TOKEN` must be set",
            ));
        };

        let api_version = crate::client::required_env_var("AZURE_API_VERSION")?;
        let azure_endpoint = crate::client::required_env_var("AZURE_ENDPOINT")?;

        Self::builder()
            .api_key(auth)
            .azure_endpoint(azure_endpoint)
            .api_version(&api_version)
            .build()
            .map_err(Into::into)
    }

    fn from_val(
        AzureOpenAIClientParams {
            api_key,
            version,
            header,
        }: Self::Input,
    ) -> Result<Self, Self::Error> {
        let auth = AzureOpenAIAuth::ApiKey(api_key.to_string());

        Self::builder()
            .api_key(auth)
            .azure_endpoint(header)
            .api_version(&version)
            .build()
            .map_err(Into::into)
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

impl From<Usage> for crate::completion::Usage {
    fn from(value: Usage) -> crate::completion::Usage {
        let mut usage = crate::completion::Usage::new();

        usage.input_tokens = value.prompt_tokens as u64;
        usage.total_tokens = value.total_tokens as u64;
        usage.output_tokens = usage.total_tokens - usage.input_tokens;

        usage
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

/// Build the serialized Azure embeddings request body (`input` +
/// optional `dimensions`). Pure; shared by the trait path and
/// [`functions::embed`].
pub(crate) fn build_embedding_body(
    texts: &[String],
    dimensions: Option<usize>,
) -> Result<Vec<u8>, EmbeddingError> {
    let mut body = json!({
        "input": texts,
    });
    let body_object = body.as_object_mut().ok_or_else(|| {
        EmbeddingError::ResponseError("embedding request body must be a JSON object".into())
    })?;
    if let Some(dimensions) = dimensions {
        body_object.insert("dimensions".to_owned(), json!(dimensions));
    }
    Ok(serde_json::to_vec(&body)?)
}

/// Parse an Azure embeddings response into the normalized
/// [`embeddings::EmbeddingResponse`], zipping vectors back onto
/// `documents`. Pure; shared by the trait path and [`functions::embed`].
pub(crate) fn parse_embedding_response(
    status: http::StatusCode,
    body: &str,
    documents: Vec<String>,
) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
    if !status.is_success() {
        return Err(EmbeddingError::from_http_response(status, body.to_string()));
    }
    let parsed: ApiResponse<EmbeddingResponse> = serde_json::from_str(body)?;
    match parsed {
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

            let usage = response.usage.clone().into();
            let embeddings = response
                .data
                .into_iter()
                .zip(documents)
                .map(|(embedding, document)| embeddings::Embedding {
                    document,
                    vec: embedding.embedding,
                })
                .collect();
            Ok(embeddings::EmbeddingResponse { embeddings, usage })
        }
        ApiResponse::Err(err) => {
            tracing::warn!(message = %err.message, "provider returned an error response");
            Err(EmbeddingError::from_http_response(status, body.to_string()))
        }
    }
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

        let dimensions =
            (self.ndims > 0 && self.model.as_str() != TEXT_EMBEDDING_ADA_002).then_some(self.ndims);
        let body = build_embedding_body(&documents, dimensions)?;

        let req = self
            .client
            .post_embedding(self.model.as_str())?
            .body(body)
            .map_err(|e| EmbeddingError::HttpError(e.into()))?;

        let response = self.client.send(req).await?;

        let status = response.status();
        let body = if status.is_success() {
            let response_body: Vec<u8> = response.into_body().await?;
            String::from_utf8_lossy(&response_body).into_owned()
        } else {
            http_client::text(response).await?
        };
        parse_embedding_response(status, &body, documents).map(|response| response.embeddings)
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

/// Azure OpenAI completion model, driven by the shared OpenAI Chat Completions
/// path. The deployment-scoped URL (including `api-version`) is produced by
/// [`completion_path`](crate::providers::openai::completion::OpenAICompatibleProvider::completion_path)
/// on [`AzureExt`], pinned to the deployment this model handle was created
/// with (a per-request `model` override changes only the request body, as
/// before the migration).
pub type CompletionModel<H = reqwest::Client> =
    openai::completion::GenericCompletionModel<AzureExt, H>;

impl openai::completion::OpenAICompatibleProvider for AzureExt {
    const DESCRIPTOR: crate::providers::descriptor::ProviderDescriptor = functions::DESCRIPTOR;
    const STREAM_DIALECT: crate::providers::descriptor::ChatCompletionsDialect =
        functions::STREAM_DIALECT;

    type Response = openai::CompletionResponse;

    // Azure routes the deployment (model) through the URL path and versions
    // the API via a query parameter; the client base URL is blank so this
    // absolute URL passes through `build_uri` untouched.
    fn completion_path(&self, model: &str) -> String {
        functions::completion_path(&self.endpoint, &self.api_version, model)
    }

    fn build_body(
        &self,
        model: &str,
        request: &crate::completion::CompletionRequest,
        options: openai::completion::CompletionModelOptions,
        stream: bool,
    ) -> Result<Vec<u8>, CompletionError> {
        functions::build_body(model, request, options, stream)
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
        let body = functions::build_transcription_form(request)?;

        let req = self
            .client
            .post_transcription(&self.model)?
            .body(body)
            .map_err(|e| TranscriptionError::HttpError(e.into()))?;

        let response = self.client.send_multipart::<Bytes>(req).await?;
        let status = response.status();
        let response_body = response.into_body().into_future().await?.to_vec();
        crate::providers::openai::functions::parse_transcription_response(status, &response_body)
    }
}

// ================================================================
// Azure OpenAI Image Generation API
// ================================================================
#[cfg(feature = "image")]
pub use image_generation::*;
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
mod image_generation {
    use crate::http_client::HttpClientExt;
    use crate::image_generation;
    use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
    use crate::providers::azure::Client;
    use crate::providers::openai::ImageGenerationResponse;
    use bytes::Bytes;

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
            let body = crate::providers::azure::functions::build_image_generation_body(
                &self.model,
                &generation_request,
            )?;

            let req = self
                .client
                .post_image_generation(&self.model)?
                .body(body)
                .map_err(|e| ImageGenerationError::HttpError(e.into()))?;

            let response = self.client.send::<_, Bytes>(req).await?;
            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();
            crate::providers::openai::functions::parse_image_generation_response(
                status,
                &response_body,
            )
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
            let body = crate::providers::openai::functions::build_audio_generation_body(
                &self.model,
                &request,
            )?;

            let req = self
                .client
                .post_audio_generation("/audio/speech")?
                .header("Content-Type", "application/json")
                .body(body)
                .map_err(|e| AudioGenerationError::HttpError(e.into()))?;

            let response = self.client.send::<_, Bytes>(req).await?;
            let status = response.status();
            let response_body = response.into_body().into_future().await?;
            crate::providers::openai::functions::parse_audio_generation_response(
                status,
                response_body.to_vec(),
            )
        }
    }
}

#[cfg(test)]
mod azure_tests {
    use super::*;
    use crate::completion::{CompletionError, CompletionRequest};

    use crate::OneOrMany;
    use crate::client::{completion::CompletionClient, embeddings::EmbeddingsClient};
    use crate::completion::CompletionModel;
    use crate::embeddings::EmbeddingModel;

    #[cfg(any(feature = "image", feature = "audio"))]
    fn test_client(
        http_client: crate::test_utils::RecordingHttpClient,
    ) -> Client<crate::test_utils::RecordingHttpClient> {
        Client::builder()
            .api_key("test-key")
            .azure_endpoint("https://example.openai.azure.com".to_string())
            .http_client(http_client)
            .build()
            .expect("build client")
    }

    #[cfg(feature = "image")]
    #[tokio::test]
    async fn image_generation_non_success_response_preserves_status_and_body() {
        use crate::image_generation::{
            ImageGenerationError, ImageGenerationModel as ImageGenerationModelTrait,
            ImageGenerationRequest,
        };
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"invalid image request"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
        let model = ImageGenerationModel::make(&test_client(http_client), "dall-e-3");

        let error = model
            .image_generation(ImageGenerationRequest {
                prompt: "draw a cat".to_string(),
                width: 256,
                height: 256,
                additional_params: None,
            })
            .await
            .expect_err("image generation should fail with non-success status");

        assert!(matches!(error, ImageGenerationError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::BAD_REQUEST)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[cfg(feature = "audio")]
    #[tokio::test]
    async fn audio_generation_non_success_response_preserves_status_and_body() {
        use crate::audio_generation::{
            AudioGenerationError, AudioGenerationModel as _, AudioGenerationRequest,
        };
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"invalid voice"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::UNPROCESSABLE_ENTITY, body);
        let model = AudioGenerationModel::new(test_client(http_client), "tts-1");

        let error = match model
            .audio_generation(AudioGenerationRequest {
                text: "hello".to_string(),
                voice: "alloy".to_string(),
                speed: 1.0,
                additional_params: None,
            })
            .await
        {
            Err(error) => error,
            Ok(_) => panic!("audio generation should fail with non-success status"),
        };

        assert!(matches!(error, AudioGenerationError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::UNPROCESSABLE_ENTITY)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn transcription_http_non_success_preserves_status_and_body() {
        use crate::test_utils::RecordingHttpClient;
        use crate::transcription::{
            TranscriptionError, TranscriptionModel as _, TranscriptionRequest,
        };

        let body = r#"{"error":{"message":"bad audio","type":"invalid_request_error"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
        let client = Client::builder()
            .api_key("test-key")
            .azure_endpoint("https://example.openai.azure.com".to_string())
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = TranscriptionModel::new(client, "whisper");

        let error = match model
            .transcription(TranscriptionRequest::new(vec![0u8; 16]))
            .await
        {
            Err(error) => error,
            Ok(_) => panic!("transcription should fail with non-success status"),
        };

        assert!(matches!(error, TranscriptionError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::BAD_REQUEST)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn embedding_http_non_success_preserves_status_and_body() {
        use crate::embeddings::EmbeddingModel as _;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"bad embedding","type":"invalid_request_error"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
        let client = Client::builder()
            .api_key("test-key")
            .azure_endpoint("https://example.openai.azure.com".to_string())
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = super::EmbeddingModel::new(client, TEXT_EMBEDDING_3_SMALL, None);

        let error = match model.embed_texts(vec!["Hello, world!".to_string()]).await {
            Err(error) => error,
            Ok(_) => panic!("embedding should fail with non-success status"),
        };

        assert!(matches!(error, EmbeddingError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::BAD_REQUEST)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn completion_pins_deployment_url_under_model_override() {
        use crate::completion::CompletionModel as _;
        use crate::test_utils::RecordingHttpClient;

        // The error response keeps the test independent of response parsing;
        // only the captured request matters here.
        let http_client = RecordingHttpClient::with_error_response(
            http::StatusCode::BAD_REQUEST,
            r#"{"error":{"message":"x"}}"#,
        );
        let client = Client::builder()
            .api_key("test-key")
            .azure_endpoint("https://example.openai.azure.com".to_string())
            .http_client(http_client.clone())
            .build()
            .expect("build client");
        let model = super::CompletionModel::new(client, GPT_4O_MINI);

        let _ = model
            .completion(CompletionRequest {
                model: Some("other-deployment".to_string()),
                preamble: None,
                chat_history: OneOrMany::one("Hello!".into()),
                documents: vec![],
                max_tokens: None,
                temperature: None,
                tools: vec![],
                tool_choice: None,
                additional_params: None,
                output_schema: None,
                record_telemetry_content: false,
            })
            .await;

        let requests = http_client.requests();
        let request = requests.first().expect("request should be captured");
        // The deployment URL stays pinned to the configured model; the
        // override only changes the body.
        assert!(
            request
                .uri
                .contains("/openai/deployments/gpt-4o-mini/chat/completions"),
            "unexpected uri: {}",
            request.uri
        );
        let body: serde_json::Value =
            serde_json::from_slice(&request.body).expect("captured body should be JSON");
        assert_eq!(body["model"], "other-deployment");
    }

    #[tokio::test]
    async fn completion_http_non_success_preserves_status_and_body() {
        use crate::completion::CompletionModel as _;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"bad completion","type":"invalid_request_error"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
        let client = Client::builder()
            .api_key("test-key")
            .azure_endpoint("https://example.openai.azure.com".to_string())
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = super::CompletionModel::new(client, GPT_4O_MINI);

        let error = match model
            .completion(CompletionRequest {
                model: None,
                preamble: Some("You are a helpful assistant.".to_string()),
                chat_history: OneOrMany::one("Hello!".into()),
                documents: vec![],
                max_tokens: Some(100),
                temperature: Some(0.0),
                tools: vec![],
                tool_choice: None,
                additional_params: None,
                output_schema: None,
                record_telemetry_content: false,
            })
            .await
        {
            Err(error) => error,
            Ok(_) => panic!("completion should fail with non-success status"),
        };

        assert!(matches!(error, CompletionError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::BAD_REQUEST)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    #[ignore]
    async fn test_azure_embedding() -> anyhow::Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let client = Client::from_env()?;
        let model = client.embedding_model(TEXT_EMBEDDING_3_SMALL);
        let embeddings = model.embed_texts(vec!["Hello, world!".to_string()]).await?;

        tracing::info!("Azure embedding: {:?}", embeddings);
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_azure_embedding_dimensions() -> anyhow::Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let ndims = 256;
        let client = Client::from_env()?;
        let model = client.embedding_model_with_ndims(TEXT_EMBEDDING_3_SMALL, ndims);
        let embedding = model.embed_text("Hello, world!").await?;

        anyhow::ensure!(
            embedding.vec.len() == ndims,
            "expected embedding dimensions {ndims}, got {}",
            embedding.vec.len()
        );

        tracing::info!("Azure dimensions embedding: {:?}", embedding);
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_azure_completion() -> anyhow::Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let client = Client::from_env()?;
        let model = client.completion_model(GPT_4O_MINI);
        let completion = model
            .completion(CompletionRequest {
                model: None,
                preamble: Some("You are a helpful assistant.".to_string()),
                chat_history: OneOrMany::one("Hello!".into()),
                documents: vec![],
                max_tokens: Some(100),
                temperature: Some(0.0),
                tools: vec![],
                tool_choice: None,
                additional_params: None,
                output_schema: None,
                record_telemetry_content: false,
            })
            .await?;

        tracing::info!("Azure completion: {:?}", completion);
        Ok(())
    }

    #[tokio::test]
    async fn test_client_initialization() {
        let _client = crate::providers::azure::Client::builder()
            .api_key("test")
            .azure_endpoint("test".to_string()) // add your endpoint here!
            .build()
            .expect("Client::builder() failed");
    }
}

pub mod functions {
    //! Azure OpenAI chat completions as config + pure functions.
    //!
    //! The data-oriented face of the Azure OpenAI provider, mirroring
    //! [`crate::providers::openai::functions`]. Azure does not fit the
    //! simple `base_url + path` shape: the deployment (model) is routed
    //! through the URL and the API is versioned via a query parameter, so
    //! [`Config`] carries `endpoint` + `api_version` instead of a
    //! `base_url`, and [`build_request`] uses the same absolute URL
    //! [`AzureExt::completion_path`](super::AzureExt) produces.
    //!
    //! Authentication mirrors the client's `AzureOpenAIAuth::ApiKey` arm:
    //! the resolved key is sent in the `api-key` header. The Entra
    //! bearer-token arm (`AzureOpenAIAuth::Token`) is not modeled here; use
    //! `extra_headers` with an `authorization` header for token auth.

    use serde::{Deserialize, Serialize};

    use crate::completion::{self, CompletionError, CompletionRequest};
    use crate::http_runtime::HttpRuntime;
    use crate::providers::descriptor::ChatCompletionsDialect;
    use crate::providers::descriptor::{ApiKeyLocation, ProviderDescriptor};
    use crate::providers::openai::completion::CompletionModelOptions;
    use crate::providers::openai::functions as openai_functions;

    /// Default Azure OpenAI API version (GA).
    pub const DEFAULT_API_VERSION: &str = "2024-10-21";

    /// Azure OpenAI's Chat Completions streaming dialect (OpenAI's own).
    pub(crate) const STREAM_DIALECT: ChatCompletionsDialect =
        ChatCompletionsDialect::from_descriptor(&DESCRIPTOR);

    /// Azure OpenAI's capability sheet.
    pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
        name: "azure.openai",
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        emits_complete_single_chunk_tool_calls: false,
        composes_native_output_with_tools: true,
        max_embedding_documents: Some(1024),
    };

    /// Plain-data Azure OpenAI provider configuration.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[non_exhaustive]
    pub struct Config {
        /// Resource endpoint, e.g. `https://my-resource.openai.azure.com`.
        pub endpoint: String,
        /// API version query parameter (defaults to [`DEFAULT_API_VERSION`]).
        pub api_version: String,
        /// Credential location; sent as the `api-key` header.
        pub api_key: ApiKeyLocation,
        /// Deployment identifier requests are built for (routed through the URL).
        pub model: String,
        /// Extra headers attached to every request.
        pub extra_headers: Vec<(String, String)>,
    }

    impl Config {
        /// Config for `model` on `endpoint`, reading `AZURE_API_KEY` from the
        /// environment.
        pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
            Self {
                endpoint: endpoint.into(),
                api_version: DEFAULT_API_VERSION.to_string(),
                api_key: ApiKeyLocation::Env("AZURE_API_KEY".to_string()),
                model: model.into(),
                extra_headers: Vec::new(),
            }
        }

        /// Config with an explicit API key.
        pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
            self.api_key = ApiKeyLocation::Inline(key.into());
            self
        }

        /// Override the API version.
        pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
            self.api_version = api_version.into();
            self
        }
    }

    /// The absolute deployment-scoped chat-completions URL, e.g.
    /// `{endpoint}/openai/deployments/{model}/chat/completions?api-version={v}`.
    ///
    /// Azure routes the deployment (model) through the URL path and versions
    /// the API via a query parameter, so this is a complete URL rather than a
    /// path relative to a base URL.
    pub(crate) fn completion_path(endpoint: &str, api_version: &str, model: &str) -> String {
        format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            endpoint,
            model.trim_start_matches('/'),
            api_version
        )
    }

    /// Azure OpenAI's chat-completions body assembly.
    ///
    /// Identical to OpenAI's: Azure's wire dialect has no body-level quirks
    /// (the deployment and API version live in the URL, not the body).
    pub(crate) fn build_body(
        model: &str,
        request: &CompletionRequest,
        options: CompletionModelOptions,
        stream: bool,
    ) -> Result<Vec<u8>, CompletionError> {
        let typed =
            openai_functions::compatible_typed_request(model, request, &DESCRIPTOR, options)?;
        let body = openai_functions::compatible_body_value(&typed, &DESCRIPTOR, stream)?;
        Ok(serde_json::to_vec(&body)?)
    }

    /// Build the serialized chat-completions request body for `request`. Pure.
    pub fn build_request_body(
        cfg: &Config,
        request: &CompletionRequest,
        stream: bool,
    ) -> Result<Vec<u8>, CompletionError> {
        build_body(
            &cfg.model,
            request,
            CompletionModelOptions::default(),
            stream,
        )
    }

    /// Build the complete HTTP request (URL, headers, body) for `request`.
    ///
    /// Pure except for credential resolution (`ApiKeyLocation::Env` reads the
    /// environment).
    pub fn build_request(
        cfg: &Config,
        request: &CompletionRequest,
        stream: bool,
    ) -> Result<http::Request<Vec<u8>>, CompletionError> {
        // Absolute deployment-scoped URL, e.g.
        // `{endpoint}/openai/deployments/{model}/chat/completions?api-version={v}`.
        let url = completion_path(
            cfg.endpoint.trim_end_matches('/'),
            &cfg.api_version,
            &cfg.model,
        );
        let body = build_request_body(cfg, request, stream)?;

        let mut builder =
            http::Request::post(url).header(http::header::CONTENT_TYPE, "application/json");
        if let Some(key) = cfg
            .api_key
            .resolve()
            .map_err(|e| CompletionError::RequestError(Box::new(e)))?
        {
            builder = builder.header("api-key", key);
        }
        for (name, value) in &cfg.extra_headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        builder
            .body(body)
            .map_err(|e| CompletionError::RequestError(Box::new(e)))
    }

    /// Parse a chat-completions response body into the normalized
    /// [`completion::CompletionResponse`]. Pure.
    pub fn parse_response(
        status: http::StatusCode,
        body: &str,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        openai_functions::compatible_parse_response::<crate::providers::openai::CompletionResponse>(
            status,
            body,
            DESCRIPTOR.name,
        )
    }

    /// Open a streaming completion for `request`.
    pub async fn open_stream(
        cfg: &Config,
        rt: &HttpRuntime,
        request: CompletionRequest,
    ) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError> {
        let req = build_request(cfg, &request, true)?;
        Ok(openai_functions::compatible_open_stream(
            rt,
            req,
            STREAM_DIALECT,
        ))
    }

    /// The deployment-scoped modality URL, e.g.
    /// `{endpoint}/openai/deployments/{model}/{suffix}?api-version={v}`.
    fn deployment_url(cfg: &Config, suffix: &str) -> String {
        format!(
            "{}/openai/deployments/{}/{}?api-version={}",
            cfg.endpoint.trim_end_matches('/'),
            cfg.model.trim_start_matches('/'),
            suffix,
            cfg.api_version
        )
    }

    fn api_key_request(
        cfg: &Config,
        url: String,
        json_content_type: bool,
    ) -> Result<http::request::Builder, crate::http_client::Error> {
        let mut builder = http::Request::post(url);
        if json_content_type {
            builder = builder.header(http::header::CONTENT_TYPE, "application/json");
        }
        if let Some(key) = cfg
            .api_key
            .resolve()
            .map_err(|e| crate::http_client::Error::Instance(e.into()))?
        {
            builder = builder.header("api-key", key);
        }
        for (name, value) in &cfg.extra_headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        Ok(builder)
    }

    /// Build the multipart form for a transcription `request`. Pure.
    ///
    /// Faithful to the classic Azure path: the deployment rides in the URL,
    /// so the form has no `model` field, and the classic path never sent
    /// `language` either.
    pub fn build_transcription_form(
        request: crate::transcription::TranscriptionRequest,
    ) -> Result<crate::http_client::MultipartForm, crate::transcription::TranscriptionError> {
        use crate::http_client::{MultipartForm, multipart::Part};
        use crate::transcription::TranscriptionError;

        let mut body =
            MultipartForm::new().part(Part::bytes("file", request.data).filename(request.filename));
        if let Some(prompt) = request.prompt {
            body = body.text("prompt", prompt);
        }
        if let Some(ref temperature) = request.temperature {
            body = body.text("temperature", temperature.to_string());
        }
        if let Some(ref additional_params) = request.additional_params {
            let params = additional_params.as_object().ok_or_else(|| {
                TranscriptionError::RequestError(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "additional transcription parameters must be a JSON object",
                )))
            })?;
            for (key, value) in params {
                body = body.text(key.to_owned(), value.to_string());
            }
        }
        Ok(body)
    }

    /// Transcribe `request` with the deployment's `audio/translations`
    /// endpoint (the same route the classic client uses). Responses share
    /// OpenAI's transcription envelope.
    pub async fn transcribe(
        cfg: &Config,
        rt: &HttpRuntime,
        request: crate::transcription::TranscriptionRequest,
    ) -> Result<
        crate::transcription::TranscriptionResponse<
            crate::providers::openai::TranscriptionResponse,
        >,
        crate::transcription::TranscriptionError,
    > {
        let form = build_transcription_form(request)?;
        let url = deployment_url(cfg, "audio/translations");
        let req = api_key_request(cfg, url, false)?
            .body(form)
            .map_err(|e| crate::transcription::TranscriptionError::RequestError(Box::new(e)))?;
        let (status, body) = rt.send_multipart(req).await?;
        openai_functions::parse_transcription_response(status, &body)
    }

    /// Build the serialized image-generation request body. Pure.
    ///
    /// Azure always requests `b64_json` (no `gpt-image` carve-out, matching
    /// the classic path).
    #[cfg(feature = "image")]
    pub fn build_image_generation_body(
        model: &str,
        request: &crate::image_generation::ImageGenerationRequest,
    ) -> Result<Vec<u8>, crate::image_generation::ImageGenerationError> {
        Ok(serde_json::to_vec(&serde_json::json!({
            "model": model,
            "prompt": request.prompt,
            "size": format!("{}x{}", request.width, request.height),
            "response_format": "b64_json",
        }))?)
    }

    /// Generate an image with the deployment's `images/generations` endpoint.
    /// Responses share OpenAI's image-generation envelope.
    #[cfg(feature = "image")]
    pub async fn generate_image(
        cfg: &Config,
        rt: &HttpRuntime,
        request: crate::image_generation::ImageGenerationRequest,
    ) -> Result<
        crate::image_generation::ImageGenerationResponse<
            crate::providers::openai::ImageGenerationResponse,
        >,
        crate::image_generation::ImageGenerationError,
    > {
        let body = build_image_generation_body(&cfg.model, &request)?;
        let url = deployment_url(cfg, "images/generations");
        let req = api_key_request(cfg, url, true)?.body(body).map_err(|e| {
            crate::image_generation::ImageGenerationError::RequestError(Box::new(e))
        })?;
        let (status, body) = rt.send_bytes(req).await?;
        openai_functions::parse_image_generation_response(status, &body)
    }

    /// Generate speech with the deployment's `audio/speech` endpoint. The
    /// request body shares OpenAI's TTS shape; success bodies are raw audio.
    #[cfg(feature = "audio")]
    pub async fn generate_audio(
        cfg: &Config,
        rt: &HttpRuntime,
        request: crate::audio_generation::AudioGenerationRequest,
    ) -> Result<
        crate::audio_generation::AudioGenerationResponse<bytes::Bytes>,
        crate::audio_generation::AudioGenerationError,
    > {
        let body = openai_functions::build_audio_generation_body(&cfg.model, &request)?;
        let url = deployment_url(cfg, "audio/speech");
        let req = api_key_request(cfg, url, true)?.body(body).map_err(|e| {
            crate::audio_generation::AudioGenerationError::RequestError(Box::new(e))
        })?;
        let (status, body) = rt.send_bytes(req).await?;
        openai_functions::parse_audio_generation_response(status, body)
    }

    /// Send `request` to Azure OpenAI and return the normalized response.
    pub async fn complete(
        cfg: &Config,
        rt: &HttpRuntime,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        let req = build_request(cfg, &request, false)?;
        let (status, body) = rt.send(req).await?;
        parse_response(status, &body)
    }

    // ================================================================
    // Embeddings
    // ================================================================

    /// Plain-data Azure OpenAI embeddings configuration.
    ///
    /// A sibling of [`Config`]: embeddings target their own deployment
    /// (`model`) and optionally request a dimension count, which do not
    /// belong on the completion configuration.
    #[derive(Debug, Clone, PartialEq, serde::Serialize, Deserialize)]
    #[non_exhaustive]
    pub struct EmbeddingConfig {
        /// Resource endpoint, e.g. `https://my-resource.openai.azure.com`.
        pub endpoint: String,
        /// API version query parameter (defaults to [`DEFAULT_API_VERSION`]).
        pub api_version: String,
        /// Credential location; sent as the `api-key` header.
        pub api_key: crate::providers::descriptor::ApiKeyLocation,
        /// Embedding deployment identifier (routed through the URL).
        pub model: String,
        /// Requested embedding dimensions, sent verbatim as `dimensions`
        /// when set (models that reject the field, like
        /// `text-embedding-ada-002`, should leave it unset).
        pub dimensions: Option<usize>,
        /// Extra headers attached to every request.
        pub extra_headers: Vec<(String, String)>,
    }

    impl EmbeddingConfig {
        /// Config for the `model` deployment on `endpoint`, reading
        /// `AZURE_API_KEY` from the environment.
        pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
            Self {
                endpoint: endpoint.into(),
                api_version: DEFAULT_API_VERSION.to_string(),
                api_key: ApiKeyLocation::Env("AZURE_API_KEY".to_string()),
                model: model.into(),
                dimensions: None,
                extra_headers: Vec::new(),
            }
        }

        /// Config with an explicit API key.
        pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
            self.api_key = ApiKeyLocation::Inline(key.into());
            self
        }

        /// Override the API version.
        pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
            self.api_version = api_version.into();
            self
        }

        /// Request `dimensions`-sized embeddings.
        pub fn with_dimensions(mut self, dimensions: usize) -> Self {
            self.dimensions = Some(dimensions);
            self
        }
    }

    /// Build the complete HTTP embeddings request for one chunk of `texts`.
    ///
    /// Pure except for credential resolution. The URL is the classic
    /// deployment-scoped shape:
    /// `{endpoint}/openai/deployments/{model}/embeddings?api-version={v}`.
    pub fn build_embedding_request(
        cfg: &EmbeddingConfig,
        texts: &[String],
    ) -> Result<http::Request<Vec<u8>>, crate::embeddings::EmbeddingError> {
        use crate::embeddings::EmbeddingError;

        let body = super::build_embedding_body(texts, cfg.dimensions)?;
        let url = format!(
            "{}/openai/deployments/{}/embeddings?api-version={}",
            cfg.endpoint.trim_end_matches('/'),
            cfg.model.trim_start_matches('/'),
            cfg.api_version
        );
        let mut builder =
            http::Request::post(url).header(http::header::CONTENT_TYPE, "application/json");
        if let Some(key) = cfg
            .api_key
            .resolve()
            .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?
        {
            builder = builder.header("api-key", key);
        }
        for (name, value) in &cfg.extra_headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        builder
            .body(body)
            .map_err(|e| EmbeddingError::ProviderError(e.to_string()))
    }

    /// Parse an embeddings response into the normalized
    /// [`crate::embeddings::EmbeddingResponse`]. Pure.
    pub fn parse_embedding_response(
        status: http::StatusCode,
        body: &str,
        documents: Vec<String>,
    ) -> Result<crate::embeddings::EmbeddingResponse, crate::embeddings::EmbeddingError> {
        super::parse_embedding_response(status, body, documents)
    }

    /// Embed `texts`, chunking to honor [`DESCRIPTOR`]'s
    /// `max_embedding_documents`; embeddings are returned in input order.
    pub async fn embed(
        cfg: &EmbeddingConfig,
        rt: &HttpRuntime,
        texts: Vec<String>,
    ) -> Result<crate::embeddings::EmbeddingResponse, crate::embeddings::EmbeddingError> {
        crate::embeddings::batching::embed_chunked(
            rt,
            texts,
            DESCRIPTOR.max_embedding_documents,
            |chunk| build_embedding_request(cfg, chunk),
            parse_embedding_response,
        )
        .await
    }

    /// Embed caller-defined batches, returning one order-aligned
    /// [`OneOrMany`](crate::OneOrMany) group per input batch plus summed
    /// usage.
    pub async fn embed_batches(
        cfg: &EmbeddingConfig,
        rt: &HttpRuntime,
        texts: Vec<Vec<String>>,
    ) -> Result<
        (
            Vec<crate::OneOrMany<crate::embeddings::Embedding>>,
            crate::completion::Usage,
        ),
        crate::embeddings::EmbeddingError,
    > {
        let (counts, flat) = crate::embeddings::batching::split_batches(texts);
        let response = embed(cfg, rt, flat).await?;
        let groups = crate::embeddings::batching::group_batches(&counts, response.embeddings)?;
        Ok((groups, response.usage))
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::OneOrMany;

        fn sample_request() -> CompletionRequest {
            CompletionRequest {
                model: None,
                preamble: None,
                chat_history: OneOrMany::one(crate::message::Message::user("hello")),
                documents: Vec::new(),
                tools: Vec::new(),
                temperature: Some(0.5),
                max_tokens: Some(64),
                tool_choice: None,
                additional_params: None,
                output_schema: None,
                record_telemetry_content: false,
            }
        }

        #[test]
        fn build_request_sets_url_and_model() {
            let cfg = Config::new("https://my-resource.openai.azure.com", "gpt-4o-deploy")
                .with_api_key("secret");
            let req = build_request(&cfg, &sample_request(), false).expect("build");
            assert_eq!(
                req.uri(),
                "https://my-resource.openai.azure.com/openai/deployments/gpt-4o-deploy/chat/completions?api-version=2024-10-21"
            );
            assert_eq!(
                req.headers().get("api-key").and_then(|v| v.to_str().ok()),
                Some("secret")
            );
            let value: serde_json::Value = serde_json::from_slice(req.body()).expect("json");
            assert_eq!(value["model"], "gpt-4o-deploy");
        }

        #[test]
        fn parse_response_normalizes() {
            let body = serde_json::json!({
                "id": "chatcmpl-1",
                "model": "gpt-4o-2024",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "logprobs": null,
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
            })
            .to_string();
            let response = parse_response(http::StatusCode::OK, &body).expect("parse");
            assert_eq!(response.provider, "azure.openai");
            assert_eq!(response.usage.input_tokens, 3);
            assert_eq!(response.usage.total_tokens, 5);
        }
    }
}
