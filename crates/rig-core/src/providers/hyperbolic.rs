//! Hyperbolic Inference API client and Rig integration
//!
//! # Example
//! ```ignore
//! use rig_core::{client::CompletionClient, providers::hyperbolic};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = hyperbolic::Client::new("YOUR_API_KEY")?;
//!
//! let llama_3_1_8b = client.completion_model(hyperbolic::LLAMA_3_1_8B);
//! # Ok(())
//! # }
//! ```

use crate::client::BearerAuth;
use crate::client::{self, DebugExt, Provider};

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

client::impl_capabilities!(
    HyperbolicExt,
    completion = CompletionModel<H>,
    image_generation = ImageGenerationModel<H>,
    audio_generation = AudioGenerationModel<H>,
);

impl DebugExt for HyperbolicExt {}

impl crate::providers::openai::completion::OpenAICompatibleProvider for HyperbolicExt {
    const PROVIDER_NAME: &'static str = "hyperbolic";

    // Hyperbolic's structured-output support is unverified; keep the
    // pre-migration behavior of dropping `output_schema` with a warning.
    const SUPPORTS_RESPONSE_FORMAT: bool = false;

    // Hyperbolic does not support tool calling; `tools`/`tool_choice` are
    // dropped with a warning during request conversion.
    const SUPPORTS_TOOLS: bool = false;

    type StreamingUsage = crate::providers::openai::Usage;

    type Response = crate::providers::openai::CompletionResponse;

    fn finalize_request_body(
        &self,
        body: &mut serde_json::Value,
    ) -> Result<(), crate::completion::CompletionError> {
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

        Ok(())
    }

    // The client base URL is the bare host; image/audio generation build
    // their own v1 paths.
    fn completion_path(&self, _model: &str) -> String {
        "/v1/chat/completions".to_string()
    }
}

client::impl_default_provider_builder!(
    HyperbolicBuilder => HyperbolicExt,
    api_key = HyperbolicApiKey,
    base_url = HYPERBOLIC_API_BASE_URL,
);

pub type Client<H> = client::Client<HyperbolicExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<HyperbolicBuilder, HyperbolicApiKey, H>;

client::impl_provider_from_env!(
    HyperbolicExt,
    input = HyperbolicApiKey,
    api_key_env = "HYPERBOLIC_API_KEY",
);

#[cfg(feature = "audio")]
use crate::providers::openai::client::ApiResponse;

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
pub type CompletionModel<H> =
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
    use super::HyperbolicExt;
    use crate::image_generation;
    use crate::image_generation::{
        ImageGenerationError, ImageGenerationRequest, NormalizeImageGenerationResponse,
    };
    use crate::json_utils::merge_inplace;
    use crate::providers::internal::image_generation::{
        GenericImageGenerationModel, JsonImageGenerationProvider, decode_base64_image,
    };
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    pub const SDXL1_0_BASE: &str = "SDXL1.0-base";
    pub const SD2: &str = "SD2";
    pub const SD1_5: &str = "SD1.5";
    pub const SSD: &str = "SSD";
    pub const SDXL_TURBO: &str = "SDXL-turbo";
    pub const SDXL_CONTROLNET: &str = "SDXL-ControlNet";
    pub const SD1_5_CONTROLNET: &str = "SD1.5-ControlNet";

    /// Hyperbolic image generation model.
    pub type ImageGenerationModel<T> = GenericImageGenerationModel<HyperbolicExt, T>;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Image {
        pub image: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ImageGenerationResponse {
        pub images: Vec<Image>,
    }

    impl NormalizeImageGenerationResponse for ImageGenerationResponse {
        fn normalize(
            self,
            provider: &str,
        ) -> Result<image_generation::ImageGenerationResponse, ImageGenerationError> {
            let image = decode_base64_image(
                &self,
                |response| response.images.first().map(|image| image.image.as_str()),
                "missing image data",
                None,
            )?;
            Ok(image_generation::ImageGenerationResponse::new(
                image, provider,
            ))
        }
    }

    impl JsonImageGenerationProvider for HyperbolicExt {
        const IMAGE_GENERATION_PATH: &'static str = "/v1/image/generation";
        const PROVIDER_NAME: &'static str = "hyperbolic";
        type Response = ImageGenerationResponse;

        fn image_generation_request_body(
            model: &str,
            generation_request: ImageGenerationRequest,
        ) -> Result<serde_json::Value, ImageGenerationError> {
            let mut request = json!({
                "model_name": model,
                "prompt": generation_request.prompt,
                "height": generation_request.height,
                "width": generation_request.width,
            });

            if let Some(params) = generation_request.additional_params {
                merge_inplace(&mut request, params);
            }

            Ok(request)
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
    use crate::audio_generation::{
        AudioGenerationError, AudioGenerationRequest, NormalizeAudioGenerationResponse,
    };
    use crate::http_client::{self, HttpClientExt};
    use base64::Engine;
    use base64::prelude::BASE64_STANDARD;
    use bytes::Bytes;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Clone)]
    pub struct AudioGenerationModel<T> {
        client: Client<T>,
        pub language: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AudioGenerationResponse {
        pub audio: String,
    }

    impl NormalizeAudioGenerationResponse for AudioGenerationResponse {
        fn normalize(
            self,
            provider: &str,
        ) -> Result<audio_generation::AudioGenerationResponse, AudioGenerationError> {
            let data = BASE64_STANDARD
                .decode(&self.audio)
                .map_err(|err| AudioGenerationError::ResponseError(err.to_string()))?;

            Ok(audio_generation::AudioGenerationResponse::new(
                data, provider,
            ))
        }
    }

    impl<T> AudioGenerationModel<T>
    where
        T: HttpClientExt + Clone + crate::wasm_compat::WasmCompatSend + 'static,
    {
        /// Perform the generation and return Hyperbolic's native response
        /// (base64 audio) instead of the normalized
        /// [`audio_generation::AudioGenerationResponse`]. Same request,
        /// transport, parser, and error path as
        /// [`audio_generation::AudioGenerationModel::audio_generation`].
        pub async fn raw_audio_generation(
            &self,
            request: AudioGenerationRequest,
        ) -> Result<AudioGenerationResponse, AudioGenerationError> {
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
                ApiResponse::Ok(response) => Ok(response),
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

    impl<T> audio_generation::AudioGenerationModel for AudioGenerationModel<T>
    where
        T: HttpClientExt + Clone + crate::wasm_compat::WasmCompatSend + 'static,
    {
        async fn audio_generation(
            &self,
            request: AudioGenerationRequest,
        ) -> Result<audio_generation::AudioGenerationResponse, AudioGenerationError> {
            crate::telemetry::instrument_modality(
                "hyperbolic",
                &self.language,
                crate::telemetry::ModalityOperation::AudioGeneration,
                async {
                    let response = self.raw_audio_generation(request).await?;
                    let captured = serde_json::to_value(&response)?;
                    Ok(response.normalize("hyperbolic")?.with_raw(captured))
                },
            )
            .await
        }
    }

    impl<T> crate::client::ConstructAudioGenerationModel<Client<T>> for AudioGenerationModel<T>
    where
        T: HttpClientExt + Clone + crate::wasm_compat::WasmCompatSend + 'static,
    {
        fn construct(client: &Client<T>, language: String) -> Self {
            Self {
                client: client.clone(),
                language,
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
            supports_image_tool_results: false,
            supports_tools: false,
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
        let _client = crate::providers::hyperbolic::Client::new_with(
            "dummy-key",
            crate::test_utils::RecordingHttpClient::new(""),
        )
        .expect("Client::new() failed");
        let builder: crate::providers::hyperbolic::ClientBuilder =
            crate::providers::hyperbolic::Client::builder().api_key("dummy-key");
        let _client_from_builder = builder
            .http_client(crate::test_utils::RecordingHttpClient::new(""))
            .build()
            .expect("Client::builder() failed");
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
            .expect_err("image generation should fail with non-success status");

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
            .expect_err("image generation should fail with provider error envelope");

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
            .expect_err("audio generation should fail with non-success status");

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
            .expect_err("audio generation should fail with provider error envelope");

        match &error {
            AudioGenerationError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::OK));
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }
}
