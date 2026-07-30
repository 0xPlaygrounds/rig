//! Hyperbolic Inference API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig_core::{client::CompletionClient, providers::hyperbolic};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = hyperbolic::Client::new("YOUR_API_KEY")?;
//!
//! let llama_3_1_8b = client.completion_model(hyperbolic::LLAMA_3_1_8B);
//! # Ok(())
//! # }
//! ```

use crate::client::{self, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder};
use crate::client::{BearerAuth, ProviderClient};
use crate::http_client::{self, HttpClientExt};
use crate::providers::descriptor::ChatCompletionsDialect;
use crate::providers::descriptor::ProviderDescriptor;

// ================================================================
// Main Hyperbolic Client
// ================================================================
const HYPERBOLIC_API_BASE_URL: &str = "https://api.hyperbolic.xyz";

#[derive(Debug, Default, Clone, Copy)]
pub struct HyperbolicExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct HyperbolicBuilder;

type HyperbolicApiKey = BearerAuth;

impl Provider for HyperbolicExt {
    type Builder = HyperbolicBuilder;

    const VERIFY_PATH: &'static str = "/models";
}

impl<H> Capabilities<H> for HyperbolicExt {
    type Completion = Capable<CompletionModel<H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Capable<ImageGenerationModel<H>>;
    #[cfg(feature = "audio")]
    type AudioGeneration = Capable<AudioGenerationModel<H>>;
    type Rerank = Nothing;
}

impl DebugExt for HyperbolicExt {}

impl crate::providers::openai::completion::OpenAICompatibleProvider for HyperbolicExt {
    const DESCRIPTOR: ProviderDescriptor = functions::DESCRIPTOR;
    const STREAM_DIALECT: ChatCompletionsDialect = functions::STREAM_DIALECT;

    type Response = crate::providers::openai::CompletionResponse;

    fn completion_path(&self, model: &str) -> String {
        functions::completion_path(model)
    }

    fn build_body(
        &self,
        model: &str,
        request: &crate::completion::CompletionRequest,
        options: crate::providers::openai::completion::CompletionModelOptions,
        stream: bool,
    ) -> Result<Vec<u8>, crate::completion::CompletionError> {
        functions::build_body(model, request, options, stream)
    }
}

impl ProviderBuilder for HyperbolicBuilder {
    type Extension<H>
        = HyperbolicExt
    where
        H: HttpClientExt;
    type ApiKey = HyperbolicApiKey;

    const BASE_URL: &'static str = HYPERBOLIC_API_BASE_URL;

    fn build<H>(
        _builder: &crate::client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(HyperbolicExt)
    }
}

pub type Client<H = reqwest::Client> = client::Client<HyperbolicExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<HyperbolicBuilder, HyperbolicApiKey, H>;

impl ProviderClient for Client {
    type Input = HyperbolicApiKey;
    type Error = crate::client::ProviderClientError;

    /// Create a new Hyperbolic client from the `HYPERBOLIC_API_KEY` environment variable.
    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("HYPERBOLIC_API_KEY")?;
        Self::new(&api_key).map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(input).map_err(Into::into)
    }
}

#[cfg(any(feature = "image", feature = "audio"))]
use serde::Deserialize;

#[cfg(any(feature = "image", feature = "audio"))]
#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[cfg(any(feature = "image", feature = "audio"))]
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

// ================================================================
// Hyperbolic Completion API
// ================================================================

/// Meta Llama 3.1b Instruct model with 8B parameters.
pub const LLAMA_3_1_8B: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct";
/// Meta Llama 3.3b Instruct model with 70B parameters.
pub const LLAMA_3_3_70B: &str = "meta-llama/Llama-3.3-70B-Instruct";
/// Meta Llama 3.1b Instruct model with 70B parameters.
pub const LLAMA_3_1_70B: &str = "meta-llama/Meta-Llama-3.1-70B-Instruct";
/// Meta Llama 3 Instruct model with 70B parameters.
pub const LLAMA_3_70B: &str = "meta-llama/Meta-Llama-3-70B-Instruct";
/// Hermes 3 Instruct model with 70B parameters.
pub const HERMES_3_70B: &str = "NousResearch/Hermes-3-Llama-3.1-70b";
/// Deepseek v2.5 model.
pub const DEEPSEEK_2_5: &str = "deepseek-ai/DeepSeek-V2.5";
/// Qwen 2.5 model with 72B parameters.
pub const QWEN_2_5_72B: &str = "Qwen/Qwen2.5-72B-Instruct";
/// Meta Llama 3.2b Instruct model with 3B parameters.
pub const LLAMA_3_2_3B: &str = "meta-llama/Llama-3.2-3B-Instruct";
/// Qwen 2.5 Coder Instruct model with 32B parameters.
pub const QWEN_2_5_CODER_32B: &str = "Qwen/Qwen2.5-Coder-32B-Instruct";
/// Preview (latest) version of Qwen model with 32B parameters.
pub const QWEN_QWQ_PREVIEW_32B: &str = "Qwen/QwQ-32B-Preview";
/// Deepseek R1 Zero model.
pub const DEEPSEEK_R1_ZERO: &str = "deepseek-ai/DeepSeek-R1-Zero";
/// Deepseek R1 model.
pub const DEEPSEEK_R1: &str = "deepseek-ai/DeepSeek-R1";

/// Hyperbolic completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = reqwest::Client> =
    crate::providers::openai::completion::GenericCompletionModel<HyperbolicExt, H>;

/// Raw completion payload, shared with the OpenAI Chat Completions path.
pub type CompletionResponse = crate::providers::openai::CompletionResponse;

// =======================================
// Hyperbolic Image Generation API
// =======================================

#[cfg(feature = "image")]
pub use image_generation::*;

#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
mod image_generation {
    use super::Client;
    use crate::http_client::HttpClientExt;
    use crate::image_generation;
    use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};

    use base64::Engine;
    use base64::prelude::BASE64_STANDARD;
    use serde::Deserialize;

    pub const SDXL1_0_BASE: &str = "SDXL1.0-base";
    pub const SD2: &str = "SD2";
    pub const SD1_5: &str = "SD1.5";
    pub const SSD: &str = "SSD";
    pub const SDXL_TURBO: &str = "SDXL-turbo";
    pub const SDXL_CONTROLNET: &str = "SDXL-ControlNet";
    pub const SD1_5_CONTROLNET: &str = "SD1.5-ControlNet";

    #[derive(Clone)]
    pub struct ImageGenerationModel<T> {
        client: Client<T>,
        pub model: String,
    }

    impl<T> ImageGenerationModel<T> {
        pub(crate) fn new(client: Client<T>, model: impl Into<String>) -> Self {
            Self {
                client,
                model: model.into(),
            }
        }

        pub fn with_model(client: Client<T>, model: &str) -> Self {
            Self {
                client,
                model: model.into(),
            }
        }
    }

    #[derive(Clone, Deserialize)]
    pub struct Image {
        image: String,
    }

    #[derive(Clone, Deserialize)]
    pub struct ImageGenerationResponse {
        images: Vec<Image>,
    }

    impl TryFrom<ImageGenerationResponse>
        for image_generation::ImageGenerationResponse<ImageGenerationResponse>
    {
        type Error = ImageGenerationError;

        fn try_from(value: ImageGenerationResponse) -> Result<Self, Self::Error> {
            let image = value
                .images
                .first()
                .ok_or_else(|| ImageGenerationError::ResponseError("missing image data".into()))?;
            let data = BASE64_STANDARD
                .decode(&image.image)
                .map_err(|err| ImageGenerationError::ResponseError(err.to_string()))?;

            Ok(Self {
                image: data,
                response: value,
            })
        }
    }

    impl<T> image_generation::ImageGenerationModel for ImageGenerationModel<T>
    where
        T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
    {
        type Response = ImageGenerationResponse;

        type Client = Client<T>;

        fn make(client: &Self::Client, model: impl Into<String>) -> Self {
            Self::new(client.clone(), model)
        }

        async fn image_generation(
            &self,
            generation_request: ImageGenerationRequest,
        ) -> Result<image_generation::ImageGenerationResponse<Self::Response>, ImageGenerationError>
        {
            let body = crate::providers::hyperbolic::functions::build_image_generation_body(
                &self.model,
                &generation_request,
            )?;

            let request = self
                .client
                .post("/v1/image/generation")?
                .header("Content-Type", "application/json")
                .body(body)
                .map_err(|e| ImageGenerationError::HttpError(e.into()))?;

            let response = self.client.send::<_, bytes::Bytes>(request).await?;

            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();
            crate::providers::hyperbolic::functions::parse_image_generation_response(
                status,
                &response_body,
            )
        }
    }
}

// ======================================
// Hyperbolic Audio Generation API
// ======================================
#[cfg(feature = "audio")]
pub use audio_generation::*;

#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
mod audio_generation {
    use super::Client;
    use crate::audio_generation;
    use crate::audio_generation::{AudioGenerationError, AudioGenerationRequest};
    use crate::http_client::{self, HttpClientExt};
    use base64::Engine;
    use base64::prelude::BASE64_STANDARD;
    use bytes::Bytes;
    use serde::Deserialize;

    #[derive(Clone)]
    pub struct AudioGenerationModel<T> {
        client: Client<T>,
        pub language: String,
    }

    #[derive(Clone, Deserialize)]
    pub struct AudioGenerationResponse {
        audio: String,
    }

    impl TryFrom<AudioGenerationResponse>
        for audio_generation::AudioGenerationResponse<AudioGenerationResponse>
    {
        type Error = AudioGenerationError;

        fn try_from(value: AudioGenerationResponse) -> Result<Self, Self::Error> {
            let data = BASE64_STANDARD
                .decode(&value.audio)
                .map_err(|err| AudioGenerationError::ResponseError(err.to_string()))?;

            Ok(Self {
                audio: data,
                response: value,
            })
        }
    }

    impl<T> audio_generation::AudioGenerationModel for AudioGenerationModel<T>
    where
        T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
    {
        type Response = AudioGenerationResponse;
        type Client = Client<T>;

        fn make(client: &Self::Client, language: impl Into<String>) -> Self {
            Self {
                client: client.clone(),
                language: language.into(),
            }
        }

        async fn audio_generation(
            &self,
            request: AudioGenerationRequest,
        ) -> Result<audio_generation::AudioGenerationResponse<Self::Response>, AudioGenerationError>
        {
            let body = crate::providers::hyperbolic::functions::build_audio_generation_body(
                &self.language,
                &request,
            )?;

            let req = self
                .client
                .post("/v1/audio/generation")?
                .body(body)
                .map_err(http_client::Error::from)?;

            let response = self.client.send::<_, Bytes>(req).await?;
            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();
            crate::providers::hyperbolic::functions::parse_audio_generation_response(
                status,
                &response_body,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn hyperbolic_body_drops_tools_and_tool_choice() {
        use crate::providers::openai::completion::CompletionModelOptions;

        let request = crate::completion::CompletionRequest {
            tools: vec![crate::completion::ToolDefinition {
                name: "lookup".to_string(),
                description: "Lookup".to_string(),
                parameters: serde_json::json!({"type":"object","properties":{},"required":[]}),
            }],
            tool_choice: Some(crate::message::ToolChoice::Required),
            output_schema: Some(schemars::schema_for!(serde_json::Value)),
            ..crate::completion::CompletionRequest::from_prompt("hello")
        };

        let bytes = super::functions::build_body(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            &request,
            CompletionModelOptions::default(),
            false,
        )
        .expect("body should build");

        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("body should be json");
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("response_format").is_none());
    }

    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::hyperbolic::Client::new("dummy-key").expect("Client::new() failed");
        let builder: crate::providers::hyperbolic::ClientBuilder =
            crate::providers::hyperbolic::Client::builder().api_key("dummy-key");
        let _client_from_builder = builder.build().expect("Client::builder() failed");
    }

    #[tokio::test]
    async fn completion_non_success_preserves_status_and_body() {
        use crate::client::CompletionClient;
        use crate::completion::{CompletionError, CompletionModel};
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"boom"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = super::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model(super::LLAMA_3_1_8B);
        let request = crate::completion::CompletionRequest::from_prompt("hello");

        let error = model
            .completion(request)
            .await
            .expect_err("completion should fail with non-success status");

        assert!(matches!(error, CompletionError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn completion_2xx_error_envelope_preserves_status_and_body() {
        use crate::client::CompletionClient;
        use crate::completion::{CompletionError, CompletionModel};
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"message":"boom"}"#;
        let http_client = RecordingHttpClient::new(body); // 200 OK
        let client = super::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model(super::LLAMA_3_1_8B);
        let request = crate::completion::CompletionRequest::from_prompt("hello");

        let error = model
            .completion(request)
            .await
            .expect_err("completion should fail with provider error envelope");

        match &error {
            CompletionError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::OK));
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }

    #[cfg(feature = "image")]
    #[tokio::test]
    async fn image_generation_non_success_preserves_status_and_body() {
        use crate::client::image_generation::ImageGenerationClient;
        use crate::image_generation::{
            ImageGenerationError, ImageGenerationModel as _, ImageGenerationRequest,
        };
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"boom"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = super::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.image_generation_model(super::SDXL1_0_BASE);

        let request = ImageGenerationRequest {
            prompt: "draw a cat".to_string(),
            width: 256,
            height: 256,
            additional_params: None,
        };

        let error = model
            .image_generation(request)
            .await
            .err()
            .expect("image generation should fail with non-success status");

        assert!(matches!(error, ImageGenerationError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[cfg(feature = "image")]
    #[tokio::test]
    async fn image_generation_2xx_error_envelope_preserves_status_and_body() {
        use crate::client::image_generation::ImageGenerationClient;
        use crate::image_generation::{
            ImageGenerationError, ImageGenerationModel as _, ImageGenerationRequest,
        };
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"message":"boom"}"#;
        let http_client = RecordingHttpClient::new(body); // 200 OK
        let client = super::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.image_generation_model(super::SDXL1_0_BASE);

        let request = ImageGenerationRequest {
            prompt: "draw a cat".to_string(),
            width: 256,
            height: 256,
            additional_params: None,
        };

        let error = model
            .image_generation(request)
            .await
            .err()
            .expect("image generation should fail with provider error envelope");

        match &error {
            ImageGenerationError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::OK));
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }

    #[cfg(feature = "audio")]
    #[tokio::test]
    async fn audio_generation_non_success_preserves_status_and_body() {
        use crate::audio_generation::{
            AudioGenerationError, AudioGenerationModel as _, AudioGenerationRequest,
        };
        use crate::client::audio_generation::AudioGenerationClient;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"boom"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = super::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.audio_generation_model("EN");

        let request = AudioGenerationRequest {
            text: "hello".to_string(),
            voice: "default".to_string(),
            speed: 1.0,
            additional_params: None,
        };

        let error = model
            .audio_generation(request)
            .await
            .err()
            .expect("audio generation should fail with non-success status");

        assert!(matches!(error, AudioGenerationError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[cfg(feature = "audio")]
    #[tokio::test]
    async fn audio_generation_2xx_error_envelope_preserves_status_and_body() {
        use crate::audio_generation::{
            AudioGenerationError, AudioGenerationModel as _, AudioGenerationRequest,
        };
        use crate::client::audio_generation::AudioGenerationClient;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"message":"boom"}"#;
        let http_client = RecordingHttpClient::new(body); // 200 OK
        let client = super::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.audio_generation_model("EN");

        let request = AudioGenerationRequest {
            text: "hello".to_string(),
            voice: "default".to_string(),
            speed: 1.0,
            additional_params: None,
        };

        let error = model
            .audio_generation(request)
            .await
            .err()
            .expect("audio generation should fail with provider error envelope");

        match &error {
            AudioGenerationError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::OK));
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }
}

pub mod functions {
    //! Hyperbolic chat completions as config + pure functions.
    //!
    //! The data-oriented face of the Hyperbolic provider, mirroring
    //! [`crate::providers::openai::functions`]: a serde [`Config`], a
    //! [`DESCRIPTOR`] capability sheet, and pure
    //! [`build_request`]/[`parse_response`] free functions plus the async
    //! [`complete`]/[`open_stream`] wrappers over
    //! [`HttpRuntime`](crate::http_runtime::HttpRuntime). The request/parse
    //! mechanics are shared with the other OpenAI-compatible providers via
    //! `openai::functions`'s stage helpers; this module owns Hyperbolic's own
    //! dialect steps, paths, and provider name.

    use serde::{Deserialize, Serialize};

    use crate::completion::{self, CompletionError, CompletionRequest};
    use crate::http_runtime::HttpRuntime;
    use crate::providers::descriptor::ChatCompletionsDialect;
    use crate::providers::descriptor::{ApiKeyLocation, ProviderDescriptor};
    use crate::providers::openai::completion::CompletionModelOptions;
    use crate::providers::openai::functions as openai_functions;

    /// Default Hyperbolic API base URL.
    pub const DEFAULT_BASE_URL: &str = "https://api.hyperbolic.xyz";

    /// Hyperbolic's Chat Completions streaming dialect.
    pub(crate) const STREAM_DIALECT: ChatCompletionsDialect =
        ChatCompletionsDialect::from_descriptor(&DESCRIPTOR);

    /// Hyperbolic's capability sheet.
    pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
        name: "hyperbolic",
        supports_tools: false,
        supports_response_format: false,
        stream_include_usage: true,
        emits_complete_single_chunk_tool_calls: false,
        composes_native_output_with_tools: false,
        max_embedding_documents: None,
    };

    /// Plain-data Hyperbolic provider configuration.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[non_exhaustive]
    pub struct Config {
        /// API base URL (defaults to [`DEFAULT_BASE_URL`]).
        pub base_url: String,
        /// Credential location.
        pub api_key: ApiKeyLocation,
        /// Model identifier requests are built for.
        pub model: String,
        /// Extra headers attached to every request.
        pub extra_headers: Vec<(String, String)>,
    }

    impl Config {
        /// Config for `model` reading `HYPERBOLIC_API_KEY` from the environment.
        pub fn new(model: impl Into<String>) -> Self {
            Self {
                base_url: DEFAULT_BASE_URL.to_string(),
                api_key: ApiKeyLocation::Env("HYPERBOLIC_API_KEY".to_string()),
                model: model.into(),
                extra_headers: Vec::new(),
            }
        }

        /// Config for `model` with an explicit API key.
        pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
            self.api_key = ApiKeyLocation::Inline(key.into());
            self
        }

        /// Override the API base URL.
        pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
            self.base_url = base_url.into();
            self
        }
    }

    /// The chat-completions request path for `model`.
    ///
    /// The client base URL is the bare host; image/audio generation build their
    /// own v1 paths.
    pub(crate) fn completion_path(_model: &str) -> String {
        "/v1/chat/completions".to_string()
    }

    /// Hyperbolic's straight-line chat-completions body assembly.
    ///
    /// Tool-exchange remnants that shared chat histories may carry are stripped
    /// from the serialized body; content-part arrays are kept as-is for
    /// Hyperbolic's vision models. Tool calling and structured output are
    /// unsupported and dropped during the typed conversion (see [`DESCRIPTOR`]).
    pub(crate) fn build_body(
        model: &str,
        request: &CompletionRequest,
        options: CompletionModelOptions,
        stream: bool,
    ) -> Result<Vec<u8>, CompletionError> {
        let typed =
            openai_functions::compatible_typed_request(model, request, &DESCRIPTOR, options)?;
        let mut body = openai_functions::compatible_body_value(&typed, &DESCRIPTOR, stream)?;
        // Strip tool-exchange remnants that shared chat histories may carry;
        // content-part arrays are kept as-is for Hyperbolic's vision models.
        if let Some(messages) = body
            .get_mut("messages")
            .and_then(serde_json::Value::as_array_mut)
        {
            crate::providers::openai::completion::sanitize_plain_text_history(
                messages, None, false, false,
            );
        }

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
        openai_functions::compatible_http_request(
            &cfg.base_url,
            &completion_path(&cfg.model),
            &cfg.api_key,
            &cfg.extra_headers,
            build_request_body(cfg, request, stream)?,
        )
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

    /// Build the serialized image-generation request body. Pure.
    #[cfg(feature = "image")]
    pub fn build_image_generation_body(
        model: &str,
        request: &crate::image_generation::ImageGenerationRequest,
    ) -> Result<Vec<u8>, crate::image_generation::ImageGenerationError> {
        let mut body = serde_json::json!({
            "model_name": model,
            "prompt": request.prompt,
            "height": request.height,
            "width": request.width,
        });
        if let Some(params) = request.additional_params.clone() {
            crate::json_utils::merge_inplace(&mut body, params);
        }
        Ok(serde_json::to_vec(&body)?)
    }

    /// Parse an image-generation response body. Pure.
    #[cfg(feature = "image")]
    pub fn parse_image_generation_response(
        status: http::StatusCode,
        body: &[u8],
    ) -> Result<
        crate::image_generation::ImageGenerationResponse<super::ImageGenerationResponse>,
        crate::image_generation::ImageGenerationError,
    > {
        use crate::image_generation::ImageGenerationError;

        if !status.is_success() {
            return Err(ImageGenerationError::from_http_response(
                status,
                String::from_utf8_lossy(body),
            ));
        }
        match serde_json::from_slice::<super::ApiResponse<super::ImageGenerationResponse>>(body)? {
            super::ApiResponse::Ok(response) => response.try_into(),
            super::ApiResponse::Err(err) => {
                tracing::warn!(message = %err.message, "provider returned an error response");
                Err(ImageGenerationError::from_http_response(
                    status,
                    String::from_utf8_lossy(body),
                ))
            }
        }
    }

    /// Generate an image with Hyperbolic's `/v1/image/generation` endpoint.
    #[cfg(feature = "image")]
    pub async fn generate_image(
        cfg: &Config,
        rt: &HttpRuntime,
        request: crate::image_generation::ImageGenerationRequest,
    ) -> Result<
        crate::image_generation::ImageGenerationResponse<super::ImageGenerationResponse>,
        crate::image_generation::ImageGenerationError,
    > {
        use crate::image_generation::ImageGenerationError;

        let body = build_image_generation_body(&cfg.model, &request)?;
        let url = format!("{}/v1/image/generation", cfg.base_url.trim_end_matches('/'));
        let req = crate::providers::openai::functions::bearer_post(
            url,
            &cfg.api_key,
            &cfg.extra_headers,
            true,
        )?
        .body(body)
        .map_err(|e| ImageGenerationError::RequestError(Box::new(e)))?;
        let (status, body) = rt.send_bytes(req).await?;
        parse_image_generation_response(status, &body)
    }

    /// Build the serialized audio-generation request body. Pure.
    ///
    /// Hyperbolic's TTS routes on a language rather than a model id;
    /// `language` mirrors the classic model handle's constructor argument.
    #[cfg(feature = "audio")]
    pub fn build_audio_generation_body(
        language: &str,
        request: &crate::audio_generation::AudioGenerationRequest,
    ) -> Result<Vec<u8>, crate::audio_generation::AudioGenerationError> {
        Ok(serde_json::to_vec(&serde_json::json!({
            "language": language,
            "speaker": request.voice,
            "text": request.text,
            "speed": request.speed,
        }))?)
    }

    /// Parse an audio-generation response body (base64 audio in a JSON
    /// envelope). Pure.
    #[cfg(feature = "audio")]
    pub fn parse_audio_generation_response(
        status: http::StatusCode,
        body: &[u8],
    ) -> Result<
        crate::audio_generation::AudioGenerationResponse<super::AudioGenerationResponse>,
        crate::audio_generation::AudioGenerationError,
    > {
        use crate::audio_generation::AudioGenerationError;

        if !status.is_success() {
            return Err(AudioGenerationError::from_http_response(
                status,
                String::from_utf8_lossy(body),
            ));
        }
        match serde_json::from_slice::<super::ApiResponse<super::AudioGenerationResponse>>(body)? {
            super::ApiResponse::Ok(response) => response.try_into(),
            super::ApiResponse::Err(err) => {
                tracing::warn!(message = %err.message, "provider returned an error response");
                Err(AudioGenerationError::from_http_response(
                    status,
                    String::from_utf8_lossy(body),
                ))
            }
        }
    }

    /// Generate speech with Hyperbolic's `/v1/audio/generation` endpoint.
    ///
    /// `cfg.model` carries the language, mirroring the classic
    /// `AudioGenerationModel` handle.
    #[cfg(feature = "audio")]
    pub async fn generate_audio(
        cfg: &Config,
        rt: &HttpRuntime,
        request: crate::audio_generation::AudioGenerationRequest,
    ) -> Result<
        crate::audio_generation::AudioGenerationResponse<super::AudioGenerationResponse>,
        crate::audio_generation::AudioGenerationError,
    > {
        use crate::audio_generation::AudioGenerationError;

        let body = build_audio_generation_body(&cfg.model, &request)?;
        let url = format!("{}/v1/audio/generation", cfg.base_url.trim_end_matches('/'));
        let req = crate::providers::openai::functions::bearer_post(
            url,
            &cfg.api_key,
            &cfg.extra_headers,
            true,
        )?
        .body(body)
        .map_err(|e| AudioGenerationError::RequestError(Box::new(e)))?;
        let (status, body) = rt.send_bytes(req).await?;
        parse_audio_generation_response(status, &body)
    }

    /// Send `request` to Hyperbolic and return the normalized response.
    pub async fn complete(
        cfg: &Config,
        rt: &HttpRuntime,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        let req = build_request(cfg, &request, false)?;
        let (status, body) = rt.send(req).await?;
        parse_response(status, &body)
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
            let cfg = Config::new("test-model").with_api_key("secret");
            let req = build_request(&cfg, &sample_request(), false).expect("build");
            assert_eq!(req.uri(), "https://api.hyperbolic.xyz/v1/chat/completions");
            let value: serde_json::Value = serde_json::from_slice(req.body()).expect("json");
            assert_eq!(value["model"], "test-model");
        }

        #[test]
        fn parse_response_normalizes() {
            let body = serde_json::json!({
                "id": "chatcmpl-1",
                "model": "test-model",
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
            assert_eq!(response.provider, "hyperbolic");
            assert_eq!(response.usage.input_tokens, 3);
            assert_eq!(response.usage.total_tokens, 5);
        }
    }
}
