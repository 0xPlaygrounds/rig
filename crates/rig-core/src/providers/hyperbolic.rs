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
    const PROVIDER_NAME: &'static str = "hyperbolic";

    // Hyperbolic's structured-output support is unverified; keep the
    // pre-migration behavior of dropping `output_schema` with a warning.
    const SUPPORTS_RESPONSE_FORMAT: bool = false;

    type StreamingUsage = crate::providers::openai::Usage;

    type Response = crate::providers::openai::CompletionResponse;

    fn prepare_request(
        &self,
        request: &mut crate::providers::openai::completion::CompletionRequest,
    ) -> Result<(), crate::completion::CompletionError> {
        // Hyperbolic does not support tool calling; drop tools rather than
        // sending parameters its API may reject.
        crate::providers::openai::completion::strip_unsupported_tools(request, "Hyperbolic");

        Ok(())
    }

    // The client base URL is the bare host; image/audio generation build
    // their own v1 paths.
    fn completion_path(&self, _model: &str) -> String {
        "/v1/chat/completions".to_string()
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

/// A Hyperbolic completion object.
///
/// For more information, see this link: <https://docs.hyperbolic.xyz/reference/create_chat_completion_v1_chat_completions_post>
/// Hyperbolic completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = reqwest::Client> =
    crate::providers::openai::completion::GenericCompletionModel<HyperbolicExt, H>;

// =======================================
// Hyperbolic Image Generation API
// =======================================

#[cfg(feature = "image")]
pub use image_generation::*;

#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
mod image_generation {
    use super::{ApiResponse, Client};
    use crate::http_client::HttpClientExt;
    use crate::image_generation;
    use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
    use crate::json_utils::merge_inplace;
    use base64::Engine;
    use base64::prelude::BASE64_STANDARD;
    use serde::Deserialize;
    use serde_json::json;

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
            let mut request = json!({
                "model_name": self.model,
                "prompt": generation_request.prompt,
                "height": generation_request.height,
                "width": generation_request.width,
            });

            if let Some(params) = generation_request.additional_params {
                merge_inplace(&mut request, params);
            }

            let body = serde_json::to_vec(&request)?;

            let request = self
                .client
                .post("/v1/image/generation")?
                .header("Content-Type", "application/json")
                .body(body)
                .map_err(|e| ImageGenerationError::HttpError(e.into()))?;

            let response = self.client.send::<_, bytes::Bytes>(request).await?;

            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if !status.is_success() {
                return Err(ImageGenerationError::from_http_response(
                    status,
                    String::from_utf8_lossy(&response_body),
                ));
            }

            match serde_json::from_slice::<ApiResponse<ImageGenerationResponse>>(&response_body)? {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(err) => {
                    tracing::warn!(message = %err.message, "provider returned an error response");
                    Err(ImageGenerationError::from_http_response(
                        status,
                        String::from_utf8_lossy(&response_body),
                    ))
                }
            }
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
    use super::{ApiResponse, Client};
    use crate::audio_generation;
    use crate::audio_generation::{AudioGenerationError, AudioGenerationRequest};
    use crate::http_client::{self, HttpClientExt};
    use base64::Engine;
    use base64::prelude::BASE64_STANDARD;
    use bytes::Bytes;
    use serde::Deserialize;
    use serde_json::json;

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
            let request = json!({
                "language": self.language,
                "speaker": request.voice,
                "text": request.text,
                "speed": request.speed
            });

            let body = serde_json::to_vec(&request)?;

            let req = self
                .client
                .post("/v1/audio/generation")?
                .body(body)
                .map_err(http_client::Error::from)?;

            let response = self.client.send::<_, Bytes>(req).await?;
            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if !status.is_success() {
                return Err(AudioGenerationError::from_http_response(
                    status,
                    String::from_utf8_lossy(&response_body),
                ));
            }

            match serde_json::from_slice::<ApiResponse<AudioGenerationResponse>>(&response_body)? {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(err) => {
                    tracing::warn!(message = %err.message, "provider returned an error response");
                    Err(AudioGenerationError::from_http_response(
                        status,
                        String::from_utf8_lossy(&response_body),
                    ))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn hyperbolic_prepare_request_drops_tools_and_tool_choice() {
        use crate::providers::openai::completion::{
            CompletionRequest as OpenAICompletionRequest, OpenAICompatibleProvider,
            OpenAIRequestParams,
        };

        let request = crate::completion::CompletionRequestBuilder::new(
            crate::test_utils::MockCompletionModel::default(),
            "hello",
        )
        .tool(crate::completion::ToolDefinition {
            name: "lookup".to_string(),
            description: "Lookup".to_string(),
            parameters: serde_json::json!({"type":"object","properties":{},"required":[]}),
        })
        .tool_choice(crate::message::ToolChoice::Required)
        .output_schema(schemars::schema_for!(serde_json::Value))
        .build();

        let mut request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
            model: "meta-llama/Meta-Llama-3.1-8B-Instruct".to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
            supports_response_format: super::HyperbolicExt::SUPPORTS_RESPONSE_FORMAT,
        })
        .expect("request should convert");
        super::HyperbolicExt
            .prepare_request(&mut request)
            .expect("prepare_request should succeed");

        let body = serde_json::to_value(request).expect("request should serialize");
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
        let request = model.completion_request("hello").build();

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
        let request = model.completion_request("hello").build();

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
