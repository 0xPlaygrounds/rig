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

use crate::client::{self, ApiKey, DebugExt, Provider, ProviderBuilder, ProviderClient};
use crate::http_client::{self, HttpClientExt, bearer_auth_header};
use crate::providers::internal::transcription::OpenAiTranscriptionClient;
use crate::providers::openai;
// ================================================================
// Main Azure OpenAI Client
// ================================================================

const DEFAULT_API_VERSION: &str = "2024-10-21";
const DEFAULT_AUDIO_API_VERSION: &str = "2025-04-01-preview";

#[derive(Debug, Clone)]
pub struct AzureExt {
    endpoint: String,
    api_version: String,
    audio_api_version: String,
}

impl DebugExt for AzureExt {
    fn fields(&self) -> impl Iterator<Item = (&'static str, &dyn std::fmt::Debug)> {
        [
            ("endpoint", (&self.endpoint as &dyn Debug)),
            ("api_version", (&self.api_version as &dyn Debug)),
            ("audio_api_version", (&self.audio_api_version as &dyn Debug)),
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
    audio_api_version: String,
}

impl Default for AzureExtBuilder {
    fn default() -> Self {
        Self {
            endpoint: None,
            api_version: DEFAULT_API_VERSION.into(),
            audio_api_version: DEFAULT_AUDIO_API_VERSION.into(),
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

client::impl_capabilities!(
    AzureExt,
    completion = CompletionModel<H>,
    embeddings = EmbeddingModel<H>,
    transcription = TranscriptionModel<H>,
    image_generation = ImageGenerationModel<H>,
    audio_generation = AudioGenerationModel<H>,
);

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
            audio_api_version,
            ..
        } = builder.ext().clone();

        match endpoint {
            Some(endpoint) => Ok(AzureExt {
                endpoint,
                api_version,
                audio_api_version,
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

    /// API version for audio generation requests.
    ///
    /// This defaults to `2025-04-01-preview`, the first deployment-scoped
    /// Azure API release that exposes text-to-speech.
    pub fn audio_api_version(mut self, api_version: &str) -> Self {
        self.ext_mut().audio_api_version = api_version.into();

        self
    }
}

impl<H> client::ClientBuilder<AzureExtBuilder, AzureOpenAIAuth, H> {
    /// Azure OpenAI endpoint URL, for example: https://{your-resource-name}.openai.azure.com
    pub fn azure_endpoint(self, endpoint: String) -> ClientBuilder<H> {
        self.over_ext(
            |AzureExtBuilder {
                 api_version,
                 audio_api_version,
                 ..
             }| AzureExtBuilder {
                endpoint: Some(endpoint),
                api_version,
                audio_api_version,
            },
        )
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

    #[cfg(feature = "audio")]
    fn post_audio_generation(
        &self,
        deployment_id: &str,
    ) -> http_client::Result<http_client::Builder> {
        let url = format!(
            "{}/openai/deployments/{}/audio/speech?api-version={}",
            self.endpoint(),
            deployment_id.trim_start_matches('/'),
            self.ext().audio_api_version
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

// ================================================================
// Azure OpenAI Embedding API
// ================================================================

/// `text-embedding-3-large` embedding model
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
/// `text-embedding-3-small` embedding model
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// `text-embedding-ada-002` embedding model
pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

/// Azure OpenAI embedding model, driven by the shared OpenAI-compatible
/// embeddings path. `EmbeddingModel::make` (and the client's
/// `embedding_model` helpers) default unknown dimensions from the model
/// identifier, exactly like OpenAI.
pub type EmbeddingModel<T = reqwest::Client> =
    openai::embedding::GenericEmbeddingModel<AzureExt, T>;

impl openai::embedding::OpenAIEmbeddingsCompatible for AzureExt {
    const PROVIDER_NAME: &'static str = "azure.openai";

    // Azure addresses the deployment through the URL, so the request body
    // carries no `model` field.
    const SENDS_MODEL_FIELD: bool = false;

    fn embeddings_path_for_model(&self, model: &str) -> String {
        format!(
            "{}/openai/deployments/{}/embeddings?api-version={}",
            self.endpoint,
            model.trim_start_matches('/'),
            self.api_version
        )
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
    const PROVIDER_NAME: &'static str = "azure.openai";

    type StreamingUsage = openai::Usage;

    type Response = openai::CompletionResponse;

    // Azure routes the deployment (model) through the URL path and versions
    // the API via a query parameter; the client base URL is blank so this
    // absolute URL passes through `build_uri` untouched.
    fn completion_path(&self, model: &str) -> String {
        format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.endpoint,
            model.trim_start_matches('/'),
            self.api_version
        )
    }
}

// ================================================================
// Azure OpenAI Transcription API
// ================================================================

/// Azure OpenAI transcription model; `model` identifies the Azure deployment.
pub type TranscriptionModel<T = reqwest::Client> =
    crate::providers::internal::transcription::OpenAiTranscriptionModel<Client<T>>;

impl<T> OpenAiTranscriptionClient for Client<T>
where
    T: HttpClientExt + Clone + 'static,
{
    const MODEL_IN_FORM: bool = false;

    fn transcription_request(
        &self,
        model: &str,
    ) -> crate::http_client::Result<crate::http_client::Builder> {
        self.post_transcription(model)
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
    use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
    use crate::providers::azure::AzureExt;
    use crate::providers::internal::image_generation::{
        GenericImageGenerationModel, JsonImageGenerationProvider,
    };
    use crate::providers::openai::ImageGenerationResponse;
    use serde_json::json;

    /// Azure OpenAI image generation model; `model` identifies the deployment.
    pub type ImageGenerationModel<T = reqwest::Client> = GenericImageGenerationModel<AzureExt, T>;

    impl JsonImageGenerationProvider for AzureExt {
        const IMAGE_GENERATION_PATH: &'static str = "";
        type Response = ImageGenerationResponse;

        fn image_generation_request_builder<H>(
            client: &crate::client::Client<Self, H>,
            model: &str,
        ) -> Result<crate::http_client::Builder, ImageGenerationError>
        where
            H: HttpClientExt,
        {
            Ok(client.post_image_generation(model)?)
        }

        fn image_generation_request_body(
            _model: &str,
            generation_request: ImageGenerationRequest,
        ) -> Result<serde_json::Value, ImageGenerationError> {
            let request = json!({
                "prompt": generation_request.prompt,
                "size": format!("{}x{}", generation_request.width, generation_request.height),
                "response_format": "b64_json"
            });

            Ok(request)
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
    use super::AzureExt;
    use crate::audio_generation::AudioGenerationError;
    use crate::http_client::HttpClientExt;
    use crate::providers::internal::audio_generation::{
        GenericAudioGenerationModel, RawAudioGenerationProvider,
    };

    /// Azure OpenAI audio generation model; `model` identifies the deployment.
    pub type AudioGenerationModel<T = reqwest::Client> = GenericAudioGenerationModel<AzureExt, T>;

    impl RawAudioGenerationProvider for AzureExt {
        const AUDIO_GENERATION_PATH: &'static str = "";

        fn audio_generation_request_builder<H>(
            client: &crate::client::Client<Self, H>,
            model: &str,
        ) -> Result<crate::http_client::Builder, AudioGenerationError>
        where
            H: HttpClientExt,
        {
            Ok(client.post_audio_generation(model)?)
        }

        fn audio_generation_request_body(
            _model: &str,
            request: crate::audio_generation::AudioGenerationRequest,
        ) -> Result<serde_json::Value, AudioGenerationError> {
            Ok(serde_json::json!({
                "input": request.text,
                "voice": request.voice,
                "speed": request.speed,
            }))
        }
    }
}

#[cfg(test)]
mod azure_tests {
    use super::*;
    use crate::client::{completion::CompletionClient, embeddings::EmbeddingsClient};
    use crate::completion::CompletionModel;
    use crate::completion::{CompletionError, CompletionRequest};
    use crate::embeddings::EmbeddingError;
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
    async fn image_generation_client_routes_to_the_deployment() {
        use crate::client::image_generation::ImageGenerationClient;
        use crate::image_generation::{ImageGenerationModel as _, ImageGenerationRequest};
        use crate::test_utils::RecordingHttpClient;

        let http_client =
            RecordingHttpClient::new(r#"{"created":0,"data":[{"b64_json":"aW1hZ2U="}]}"#);
        let client = test_client(http_client.clone());
        let model = client.image_generation_model("image-deployment");

        let response = model
            .image_generation(ImageGenerationRequest {
                prompt: "draw a cat".to_owned(),
                width: 256,
                height: 256,
                additional_params: None,
            })
            .await
            .expect("image generation should succeed");

        assert_eq!(response.image, b"image");
        let requests = http_client.requests();
        assert_eq!(
            requests[0].uri,
            "https://example.openai.azure.com/openai/deployments/image-deployment/images/generations?api-version=2024-10-21"
        );
        let body: serde_json::Value =
            serde_json::from_slice(&requests[0].body).expect("request body should be JSON");
        assert!(body.get("model").is_none());
        assert_eq!(body["response_format"], "b64_json");
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
    #[test]
    fn audio_api_version_can_be_overridden() {
        let client = Client::builder()
            .api_key("test-key")
            .azure_endpoint("https://example.openai.azure.com".to_owned())
            .audio_api_version("2026-01-01-preview")
            .build()
            .expect("build client");
        let request = client
            .post_audio_generation("tts-deployment")
            .expect("build audio request")
            .body(Vec::<u8>::new())
            .expect("finish audio request");

        assert_eq!(
            request.uri(),
            "https://example.openai.azure.com/openai/deployments/tts-deployment/audio/speech?api-version=2026-01-01-preview"
        );
    }

    #[cfg(feature = "audio")]
    #[tokio::test]
    async fn audio_generation_routes_to_the_deployment() {
        use crate::audio_generation::{AudioGenerationModel as _, AudioGenerationRequest};
        use crate::client::audio_generation::AudioGenerationClient;
        use crate::test_utils::RecordingHttpClient;

        let http_client = RecordingHttpClient::new("audio");
        let client = test_client(http_client.clone());
        let model = client.audio_generation_model("tts-deployment");

        let response = model
            .audio_generation(AudioGenerationRequest {
                text: "hello".to_owned(),
                voice: "alloy".to_owned(),
                speed: 1.0,
                additional_params: None,
            })
            .await
            .expect("audio generation should succeed");

        assert_eq!(response.audio, b"audio");
        let requests = http_client.requests();
        assert_eq!(
            requests[0].uri,
            "https://example.openai.azure.com/openai/deployments/tts-deployment/audio/speech?api-version=2025-04-01-preview"
        );
        let body: serde_json::Value =
            serde_json::from_slice(&requests[0].body).expect("request body should be JSON");
        assert!(body.get("model").is_none());
        assert_eq!(body["input"], "hello");
        assert_eq!(body["voice"], "alloy");
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
        use crate::transcription::{TranscriptionError, TranscriptionModel as _};

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
            .transcription_request()
            .data(vec![0u8; 16])
            .send()
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
    async fn transcription_routes_deployment_in_url_not_multipart_body() {
        use crate::test_utils::RecordingHttpClient;
        use crate::transcription::TranscriptionModel as _;

        let http_client = RecordingHttpClient::new(r#"{"text":"transcribed"}"#);
        let client = Client::builder()
            .api_key("test-key")
            .azure_endpoint("https://example.openai.azure.com".to_owned())
            .http_client(http_client.clone())
            .build()
            .expect("build client");
        let model = TranscriptionModel::new(client, "whisper-deployment");

        let response = model
            .transcription_request()
            .data(vec![1, 2, 3])
            .filename(Some("audio.mp3".to_owned()))
            .send()
            .await
            .expect("transcription should succeed");

        assert_eq!(response.text, "transcribed");
        let request = http_client
            .requests()
            .into_iter()
            .next()
            .expect("request should be captured");
        assert_eq!(
            request.uri,
            "https://example.openai.azure.com/openai/deployments/whisper-deployment/audio/translations?api-version=2024-10-21"
        );
        let body = String::from_utf8_lossy(&request.body);
        assert!(!body.contains("name=\"model\""), "{body}");
        assert!(
            body.contains("name=\"file\"; filename=\"audio.mp3\""),
            "{body}"
        );
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
        let model = super::EmbeddingModel::make(&client, TEXT_EMBEDDING_3_SMALL, None);

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
    async fn embedding_preserves_deployment_url_and_body_and_reports_usage() {
        use crate::embeddings::EmbeddingModel as _;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{
            "object": "list",
            "model": "text-embedding-3-small",
            "usage": { "prompt_tokens": 4, "total_tokens": 4 },
            "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2] }]
        }"#;
        let http_client = RecordingHttpClient::new(body);
        let client = Client::builder()
            .api_key("test-key")
            .azure_endpoint("https://example.openai.azure.com".to_string())
            .http_client(http_client.clone())
            .build()
            .expect("build client");
        let model = super::EmbeddingModel::make(&client, TEXT_EMBEDDING_3_SMALL, None);

        let response = model
            .embed_texts_with_usage(vec!["Hello, world!".to_string()])
            .await
            .expect("embedding should succeed");

        // Usage is now surfaced instead of the zero-usage default.
        assert_eq!(response.usage.input_tokens, 4);
        assert_eq!(response.usage.total_tokens, 4);
        assert_eq!(response.embeddings.len(), 1);

        // The deployment stays in the URL and the body carries no `model`
        // field, matching the hand-rolled request this replaced.
        let requests = http_client.requests();
        assert_eq!(
            requests[0].uri,
            format!(
                "https://example.openai.azure.com/openai/deployments/{TEXT_EMBEDDING_3_SMALL}/embeddings?api-version=2024-10-21"
            )
        );
        let request_body: serde_json::Value =
            serde_json::from_slice(&requests[0].body).expect("request body should be JSON");
        assert_eq!(request_body.get("model"), None);
        assert_eq!(request_body["dimensions"], serde_json::json!(1_536));
        assert_eq!(request_body["input"], serde_json::json!(["Hello, world!"]));
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
                chat_history: vec!["Hello!".into()],
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
                chat_history: vec!["Hello!".into()],
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
                chat_history: vec!["Hello!".into()],
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
