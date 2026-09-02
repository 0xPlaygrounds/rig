//! Azure OpenAI API client and Rig integration
//!
//! # Example
//! ```ignore
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

use crate::client::{self, ApiKey, DebugExt, Provider, ProviderBuilder};
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

pub type Client<H = crate::http_client::BoxedHttpClient> = client::Client<AzureExt, H>;
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
    pub fn api_version(mut self, api_version: impl Into<String>) -> Self {
        self.ext_mut().api_version = api_version.into();

        self
    }

    /// API version for audio generation requests.
    ///
    /// This defaults to `2025-04-01-preview`, the first deployment-scoped
    /// Azure API release that exposes text-to-speech.
    pub fn audio_api_version(mut self, api_version: impl Into<String>) -> Self {
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

impl crate::client::ProviderFromEnv for AzureExt {
    type Input = AzureOpenAIClientParams;
    /// Create a new Azure OpenAI client from the `AZURE_API_KEY` or `AZURE_TOKEN`, `AZURE_API_VERSION`, and `AZURE_ENDPOINT` environment variables.
    fn from_env_with<H>(
        http: H,
    ) -> Result<crate::client::Client<Self, H>, crate::client::ProviderClientError>
    where
        H: crate::http_client::HttpClientExt,
        Self::Builder: crate::client::ProviderBuilder<Extension<H> = Self>,
    {
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

        crate::client::Client::<Self, crate::markers::Missing>::builder()
            .api_key(auth)
            .azure_endpoint(azure_endpoint)
            .api_version(&api_version)
            .http_client(http)
            .build()
            .map_err(Into::into)
    }

    fn from_val_with<H>(
        AzureOpenAIClientParams {
            api_key,
            version,
            header,
        }: Self::Input,
        http: H,
    ) -> Result<crate::client::Client<Self, H>, crate::client::ProviderClientError>
    where
        H: crate::http_client::HttpClientExt,
        Self::Builder: crate::client::ProviderBuilder<Extension<H> = Self>,
    {
        let auth = AzureOpenAIAuth::ApiKey(api_key);

        crate::client::Client::<Self, crate::markers::Missing>::builder()
            .api_key(auth)
            .azure_endpoint(header)
            .api_version(&version)
            .http_client(http)
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
pub type EmbeddingModel<T = crate::http_client::BoxedHttpClient> =
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
pub type CompletionModel<H = crate::http_client::BoxedHttpClient> =
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
pub type TranscriptionModel<T = crate::http_client::BoxedHttpClient> =
    crate::providers::internal::transcription::OpenAiTranscriptionModel<Client<T>>;

impl<T> OpenAiTranscriptionClient for Client<T>
where
    T: HttpClientExt + Clone + 'static,
{
    const MODEL_IN_FORM: bool = false;
    const PROVIDER_NAME: &'static str = "azure.openai";
    const REQUEST_ID_HEADER: Option<&'static str> = None;

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
    pub type ImageGenerationModel<T> = GenericImageGenerationModel<AzureExt, T>;

    impl JsonImageGenerationProvider for AzureExt {
        const IMAGE_GENERATION_PATH: &'static str = "";
        const PROVIDER_NAME: &'static str = "azure.openai";
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
    pub type AudioGenerationModel<T> = GenericAudioGenerationModel<AzureExt, T>;

    impl RawAudioGenerationProvider for AzureExt {
        const AUDIO_GENERATION_PATH: &'static str = "";
        const PROVIDER_NAME: &'static str = "azure.openai";

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
mod azure_tests;
