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

use super::openai::{TranscriptionResponse, send_compatible_streaming_request};

use crate::json_utils::merge;
use crate::streaming::StreamingCompletionResponse;
use crate::{
    client::ClientBuilderError,
    completion::{self, CompletionError, CompletionRequest},
    embeddings::{self, EmbeddingError},
    json_utils,
    providers::openai,
    transcription::{self, TranscriptionError},
};
use reqwest::header::AUTHORIZATION;
use reqwest::multipart::Part;
use serde::Deserialize;
use serde_json::json;
// ================================================================
// Main Azure OpenAI Client
// ================================================================

const DEFAULT_API_VERSION: &str = "2024-10-21";

pub struct ClientBuilder<'a> {
    auth: AzureOpenAIAuth,
    api_version: Option<&'a str>,
    azure_endpoint: &'a str,
    http_client: Option<reqwest::Client>,
}

impl<'a> ClientBuilder<'a> {
    pub fn new(auth: impl Into<AzureOpenAIAuth>, endpoint: &'a str) -> Self {
        Self {
            auth: auth.into(),
            api_version: None,
            azure_endpoint: endpoint,
            http_client: None,
        }
    }

    /// API version to use (e.g., "2024-10-21" for GA, "2024-10-01-preview" for preview)
    pub fn api_version(mut self, api_version: &'a str) -> Self {
        self.api_version = Some(api_version);
        self
    }

    /// Azure OpenAI endpoint URL, for example: https://{your-resource-name}.openai.azure.com
    pub fn azure_endpoint(mut self, azure_endpoint: &'a str) -> Self {
        self.azure_endpoint = azure_endpoint;
        self
    }

    pub fn custom_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = Some(client);
        self
    }

    pub fn build(self) -> Result<Client, ClientBuilderError> {
        let http_client = if let Some(http_client) = self.http_client {
            http_client
        } else {
            reqwest::Client::builder().build()?
        };

        let api_version = self.api_version.unwrap_or(DEFAULT_API_VERSION);

        Ok(Client {
            api_version: api_version.to_string(),
            azure_endpoint: self.azure_endpoint.to_string(),
            auth: self.auth,
            http_client,
        })
    }
}

#[derive(Clone)]
pub struct Client {
    api_version: String,
    azure_endpoint: String,
    auth: AzureOpenAIAuth,
    http_client: reqwest::Client,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("azure_endpoint", &self.azure_endpoint)
            .field("http_client", &self.http_client)
            .field("auth", &"<REDACTED>")
            .field("api_version", &self.api_version)
            .finish()
    }
}

#[derive(Clone)]
pub enum AzureOpenAIAuth {
    ApiKey(String),
    Token(String),
}

impl std::fmt::Debug for AzureOpenAIAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiKey(_) => write!(f, "API key <REDACTED>"),
            Self::Token(_) => write!(f, "Token <REDACTED>"),
        }
    }
}

impl From<String> for AzureOpenAIAuth {
    fn from(token: String) -> Self {
        AzureOpenAIAuth::Token(token)
    }
}

impl AzureOpenAIAuth {
    fn as_header(&self) -> (reqwest::header::HeaderName, reqwest::header::HeaderValue) {
        match self {
            AzureOpenAIAuth::ApiKey(api_key) => (
                "api-key".parse().expect("Header value should parse"),
                api_key.parse().expect("API key should parse"),
            ),
            AzureOpenAIAuth::Token(token) => (
                AUTHORIZATION,
                format!("Bearer {token}")
                    .parse()
                    .expect("Token should parse"),
            ),
        }
    }
}

impl Client {
    /// Create a new Azure OpenAI client builder.
    ///
    /// # Example
    /// ```
    /// use rig::providers::azure::{ClientBuilder, self};
    ///
    /// // Initialize the Azure OpenAI client
    /// let azure = Client::builder("your-azure-api-key", "https://{your-resource-name}.openai.azure.com")
    ///    .build()
    /// ```
    pub fn builder(auth: impl Into<AzureOpenAIAuth>, endpoint: &str) -> ClientBuilder<'_> {
        ClientBuilder::new(auth, endpoint)
    }

    /// Creates a new Azure OpenAI client. For more control, use the `builder` method.
    ///
    /// # Panics
    /// - If the reqwest client cannot be built (if the TLS backend cannot be initialized).
    pub fn new(auth: impl Into<AzureOpenAIAuth>, endpoint: &str) -> Self {
        Self::builder(auth, endpoint)
            .build()
            .expect("Azure OpenAI client should build")
    }

    fn post_embedding(&self, deployment_id: &str) -> reqwest::RequestBuilder {
        let url = format!(
            "{}/openai/deployments/{}/embeddings?api-version={}",
            self.azure_endpoint, deployment_id, self.api_version
        )
        .replace("//", "/");

        let (key, value) = self.auth.as_header();
        self.http_client.post(url).header(key, value)
    }

    fn post_chat_completion(&self, deployment_id: &str) -> reqwest::RequestBuilder {
        let url = format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.azure_endpoint, deployment_id, self.api_version
        )
        .replace("//", "/");
        let (key, value) = self.auth.as_header();
        self.http_client.post(url).header(key, value)
    }

    fn post_transcription(&self, deployment_id: &str) -> reqwest::RequestBuilder {
        let url = format!(
            "{}/openai/deployments/{}/audio/translations?api-version={}",
            self.azure_endpoint, deployment_id, self.api_version
        )
        .replace("//", "/");
        let (key, value) = self.auth.as_header();
        self.http_client.post(url).header(key, value)
    }

    #[cfg(feature = "image")]
    fn post_image_generation(&self, deployment_id: &str) -> reqwest::RequestBuilder {
        let url = format!(
            "{}/openai/deployments/{}/images/generations?api-version={}",
            self.azure_endpoint, deployment_id, self.api_version
        )
        .replace("//", "/");
        let (key, value) = self.auth.as_header();
        self.http_client.post(url).header(key, value)
    }

    #[cfg(feature = "audio")]
    fn post_audio_generation(&self, deployment_id: &str) -> reqwest::RequestBuilder {
        let url = format!(
            "{}/openai/deployments/{}/audio/speech?api-version={}",
            self.azure_endpoint, deployment_id, self.api_version
        )
        .replace("//", "/");
        let (key, value) = self.auth.as_header();
        self.http_client.post(url).header(key, value)
    }
}

impl ProviderClient for Client {
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

        Self::builder(auth, &azure_endpoint)
            .api_version(&api_version)
            .build()
            .expect("Azure OpenAI client should build")
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::ApiKeyWithVersionAndHeader(api_key, version, header) =
            input
        else {
            panic!("Incorrect provider value type")
        };
        let auth = AzureOpenAIAuth::ApiKey(api_key.to_string());
        Self::builder(auth, &header)
            .api_version(&version)
            .build()
            .expect("Azure OpenAI client should build")
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;

    /// Create a completion model with the given name.
    ///
    /// # Example
    /// ```
    /// use rig::providers::azure::{Client, self};
    ///
    /// // Initialize the Azure OpenAI client
    /// let azure = Client::new("YOUR_API_KEY", "YOUR_API_VERSION", "YOUR_ENDPOINT");
    ///
    /// let gpt4 = azure.completion_model(azure::GPT_4);
    /// ```
    fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl EmbeddingsClient for Client {
    type EmbeddingModel = EmbeddingModel;

    /// Create an embedding model with the given name.
    /// Note: default embedding dimension of 0 will be used if model is not known.
    /// If this is the case, it's better to use function `embedding_model_with_ndims`
    ///
    /// # Example
    /// ```
    /// use rig::providers::azure::{Client, self};
    ///
    /// // Initialize the Azure OpenAI client
    /// let azure = Client::new("YOUR_API_KEY", "YOUR_API_VERSION", "YOUR_ENDPOINT");
    ///
    /// let embedding_model = azure.embedding_model(azure::TEXT_EMBEDDING_3_LARGE);
    /// ```
    fn embedding_model(&self, model: &str) -> EmbeddingModel {
        let ndims = match model {
            TEXT_EMBEDDING_3_LARGE => 3072,
            TEXT_EMBEDDING_3_SMALL | TEXT_EMBEDDING_ADA_002 => 1536,
            _ => 0,
        };
        EmbeddingModel::new(self.clone(), model, ndims)
    }

    /// Create an embedding model with the given name and the number of dimensions in the embedding generated by the model.
    ///
    /// # Example
    /// ```
    /// use rig::providers::azure::{Client, self};
    ///
    /// // Initialize the Azure OpenAI client
    /// let azure = Client::new("YOUR_API_KEY", "YOUR_API_VERSION", "YOUR_ENDPOINT");
    ///
    /// let embedding_model = azure.embedding_model("model-unknown-to-rig", 3072);
    /// ```
    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, ndims)
    }
}

impl TranscriptionClient for Client {
    type TranscriptionModel = TranscriptionModel;

    /// Create a transcription model with the given name.
    ///
    /// # Example
    /// ```
    /// use rig::providers::azure::{Client, self};
    ///
    /// // Initialize the Azure OpenAI client
    /// let azure = Client::new("YOUR_API_KEY", "YOUR_API_VERSION", "YOUR_ENDPOINT");
    ///
    /// let whisper = azure.transcription_model("model-unknown-to-rig");
    /// ```
    fn transcription_model(&self, model: &str) -> TranscriptionModel {
        TranscriptionModel::new(self.clone(), model)
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
pub struct EmbeddingModel {
    client: Client,
    pub model: String,
    ndims: usize,
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;

    fn ndims(&self) -> usize {
        self.ndims
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents = documents.into_iter().collect::<Vec<_>>();

        let response = self
            .client
            .post_embedding(&self.model)
            .json(&json!({
                "input": documents,
            }))
            .send()
            .await?;

        if response.status().is_success() {
            match response.json::<ApiResponse<EmbeddingResponse>>().await? {
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
            Err(EmbeddingError::ProviderError(response.text().await?))
        }
    }
}

impl EmbeddingModel {
    pub fn new(client: Client, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.to_string(),
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

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    /// Name of the model (e.g.: gpt-4o-mini)
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }

    fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<serde_json::Value, CompletionError> {
        let mut full_history: Vec<openai::Message> = match &completion_request.preamble {
            Some(preamble) => vec![openai::Message::system(preamble)],
            None => vec![],
        };
        if let Some(docs) = completion_request.normalized_documents() {
            let docs: Vec<openai::Message> = docs.try_into()?;
            full_history.extend(docs);
        }
        let chat_history: Vec<openai::Message> = completion_request
            .chat_history
            .into_iter()
            .map(|message| message.try_into())
            .collect::<Result<Vec<Vec<openai::Message>>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        full_history.extend(chat_history);

        let request = if completion_request.tools.is_empty() {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
            })
        } else {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
                "tools": completion_request.tools.into_iter().map(openai::ToolDefinition::from).collect::<Vec<_>>(),
                "tool_choice": "auto",
            })
        };

        let request = if let Some(params) = completion_request.additional_params {
            json_utils::merge(request, params)
        } else {
            request
        };

        Ok(request)
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = openai::CompletionResponse;
    type StreamingResponse = openai::StreamingCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<openai::CompletionResponse>, CompletionError> {
        let request = self.create_completion_request(completion_request)?;

        let response = self
            .client
            .post_chat_completion(&self.model)
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let t = response.text().await?;
            tracing::debug!(target: "rig", "Azure completion error: {}", t);

            match serde_json::from_str::<ApiResponse<openai::CompletionResponse>>(&t)? {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "Azure completion token usage: {:?}",
                        response.usage.clone().map(|usage| format!("{usage}")).unwrap_or("N/A".to_string())
                    );
                    response.try_into()
                }
                ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
            }
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let mut request = self.create_completion_request(request)?;

        request = merge(
            request,
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        );

        let builder = self
            .client
            .post_chat_completion(self.model.as_str())
            .json(&request);

        send_compatible_streaming_request(builder).await
    }
}

// ================================================================
// Azure OpenAI Transcription API
// ================================================================

#[derive(Clone)]
pub struct TranscriptionModel {
    client: Client,
    /// Name of the model (e.g.: gpt-3.5-turbo-1106)
    pub model: String,
}

impl TranscriptionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl transcription::TranscriptionModel for TranscriptionModel {
    type Response = TranscriptionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn transcription(
        &self,
        request: transcription::TranscriptionRequest,
    ) -> Result<
        transcription::TranscriptionResponse<Self::Response>,
        transcription::TranscriptionError,
    > {
        let data = request.data;

        let mut body = reqwest::multipart::Form::new().part(
            "file",
            Part::bytes(data).file_name(request.filename.clone()),
        );

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

        let response = self
            .client
            .post_transcription(&self.model)
            .multipart(body)
            .send()
            .await?;

        if response.status().is_success() {
            match response
                .json::<ApiResponse<TranscriptionResponse>>()
                .await?
            {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(api_error_response) => Err(TranscriptionError::ProviderError(
                    api_error_response.message,
                )),
            }
        } else {
            Err(TranscriptionError::ProviderError(response.text().await?))
        }
    }
}

// ================================================================
// Azure OpenAI Image Generation API
// ================================================================
#[cfg(feature = "image")]
pub use image_generation::*;
#[cfg(feature = "image")]
mod image_generation {
    use crate::client::ImageGenerationClient;
    use crate::image_generation;
    use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
    use crate::providers::azure::{ApiResponse, Client};
    use crate::providers::openai::ImageGenerationResponse;
    use serde_json::json;

    #[derive(Clone)]
    pub struct ImageGenerationModel {
        client: Client,
        pub model: String,
    }
    impl image_generation::ImageGenerationModel for ImageGenerationModel {
        type Response = ImageGenerationResponse;

        #[cfg_attr(feature = "worker", worker::send)]
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

            let response = self
                .client
                .post_image_generation(&self.model)
                .json(&request)
                .send()
                .await?;

            if !response.status().is_success() {
                return Err(ImageGenerationError::ProviderError(format!(
                    "{}: {}",
                    response.status(),
                    response.text().await?
                )));
            }

            let t = response.text().await?;

            match serde_json::from_str::<ApiResponse<ImageGenerationResponse>>(&t)? {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(err) => Err(ImageGenerationError::ProviderError(err.message)),
            }
        }
    }

    impl ImageGenerationClient for Client {
        type ImageGenerationModel = ImageGenerationModel;

        fn image_generation_model(&self, model: &str) -> Self::ImageGenerationModel {
            ImageGenerationModel {
                client: self.clone(),
                model: model.to_string(),
            }
        }
    }
}
// ================================================================
// Azure OpenAI Audio Generation API
// ================================================================

use crate::client::{
    CompletionClient, EmbeddingsClient, ProviderClient, TranscriptionClient, VerifyClient,
    VerifyError,
};
#[cfg(feature = "audio")]
pub use audio_generation::*;

#[cfg(feature = "audio")]
mod audio_generation {
    use super::Client;
    use crate::audio_generation;
    use crate::audio_generation::{
        AudioGenerationError, AudioGenerationRequest, AudioGenerationResponse,
    };
    use crate::client::AudioGenerationClient;
    use bytes::Bytes;
    use serde_json::json;

    #[derive(Clone)]
    pub struct AudioGenerationModel {
        client: Client,
        model: String,
    }

    impl audio_generation::AudioGenerationModel for AudioGenerationModel {
        type Response = Bytes;

        #[cfg_attr(feature = "worker", worker::send)]
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

            let response = self
                .client
                .post_audio_generation("/audio/speech")
                .json(&request)
                .send()
                .await?;

            if !response.status().is_success() {
                return Err(AudioGenerationError::ProviderError(format!(
                    "{}: {}",
                    response.status(),
                    response.text().await?
                )));
            }

            let bytes = response.bytes().await?;

            Ok(AudioGenerationResponse {
                audio: bytes.to_vec(),
                response: bytes,
            })
        }
    }

    impl AudioGenerationClient for Client {
        type AudioGenerationModel = AudioGenerationModel;

        fn audio_generation_model(&self, model: &str) -> Self::AudioGenerationModel {
            AudioGenerationModel {
                client: self.clone(),
                model: model.to_string(),
            }
        }
    }
}

impl VerifyClient for Client {
    #[cfg_attr(feature = "worker", worker::send)]
    async fn verify(&self) -> Result<(), VerifyError> {
        // There is currently no way to verify the Azure OpenAI API key or token without
        // consuming tokens
        Ok(())
    }
}

#[cfg(test)]
mod azure_tests {
    use super::*;

    use crate::OneOrMany;
    use crate::completion::CompletionModel;
    use crate::embeddings::EmbeddingModel;

    #[tokio::test]
    #[ignore]
    async fn test_azure_embedding() {
        let _ = tracing_subscriber::fmt::try_init();

        let client = Client::from_env();
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

        let client = Client::from_env();
        let model = client.completion_model(GPT_4O_MINI);
        let completion = model
            .completion(CompletionRequest {
                preamble: Some("You are a helpful assistant.".to_string()),
                chat_history: OneOrMany::one("Hello!".into()),
                documents: vec![],
                max_tokens: Some(100),
                temperature: Some(0.0),
                tools: vec![],
                additional_params: None,
            })
            .await
            .unwrap();

        tracing::info!("Azure completion: {:?}", completion);
    }
}
