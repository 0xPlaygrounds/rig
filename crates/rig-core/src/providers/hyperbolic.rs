//! Hyperbolic Inference API integration.
//!
//! # Example
//! ```no_run
//! use rig_core::providers::hyperbolic;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! # let request = rig_core::completion::CompletionRequest::from_prompt("hello");
//! let cfg = hyperbolic::functions::Config::from_env(hyperbolic::LLAMA_3_1_8B)?;
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//! let response = hyperbolic::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```

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
    use crate::image_generation;
    use crate::image_generation::ImageGenerationError;

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
}

// ======================================
// Hyperbolic Audio Generation API
// ======================================
#[cfg(feature = "audio")]
pub use audio_generation::*;

#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
mod audio_generation {
    use crate::audio_generation;
    use crate::audio_generation::AudioGenerationError;
    use base64::Engine;
    use base64::prelude::BASE64_STANDARD;
    use serde::Deserialize;

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

    #[tokio::test]
    async fn completion_non_success_preserves_status_and_body() {
        use crate::completion::CompletionError;
        use crate::http_runtime::HttpRuntime;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"boom"}}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::with_error_response(
            http::StatusCode::SERVICE_UNAVAILABLE,
            body,
        ));
        let cfg = super::functions::Config::new(super::LLAMA_3_1_8B).with_api_key("test-key");
        let request = crate::completion::CompletionRequest::from_prompt("hello");

        let error = super::functions::complete(&cfg, &rt, request)
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
        use crate::completion::CompletionError;
        use crate::http_runtime::HttpRuntime;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"message":"boom"}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::new(body)); // 200 OK
        let cfg = super::functions::Config::new(super::LLAMA_3_1_8B).with_api_key("test-key");
        let request = crate::completion::CompletionRequest::from_prompt("hello");

        let error = super::functions::complete(&cfg, &rt, request)
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
        use crate::http_runtime::HttpRuntime;
        use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"boom"}}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::with_error_response(
            http::StatusCode::SERVICE_UNAVAILABLE,
            body,
        ));
        let cfg = super::functions::Config::new(super::SDXL1_0_BASE).with_api_key("test-key");

        let request = ImageGenerationRequest {
            prompt: "draw a cat".to_string(),
            width: 256,
            height: 256,
            additional_params: None,
        };

        let error = super::functions::generate_image(&cfg, &rt, request)
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
        use crate::http_runtime::HttpRuntime;
        use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"message":"boom"}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::new(body)); // 200 OK
        let cfg = super::functions::Config::new(super::SDXL1_0_BASE).with_api_key("test-key");

        let request = ImageGenerationRequest {
            prompt: "draw a cat".to_string(),
            width: 256,
            height: 256,
            additional_params: None,
        };

        let error = super::functions::generate_image(&cfg, &rt, request)
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
        use crate::audio_generation::{AudioGenerationError, AudioGenerationRequest};
        use crate::http_runtime::HttpRuntime;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"boom"}}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::with_error_response(
            http::StatusCode::SERVICE_UNAVAILABLE,
            body,
        ));
        let cfg = super::functions::Config::new("EN").with_api_key("test-key");

        let request = AudioGenerationRequest {
            text: "hello".to_string(),
            voice: "default".to_string(),
            speed: 1.0,
            additional_params: None,
        };

        let error = super::functions::generate_audio(&cfg, &rt, request)
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
        use crate::audio_generation::{AudioGenerationError, AudioGenerationRequest};
        use crate::http_runtime::HttpRuntime;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"message":"boom"}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::new(body)); // 200 OK
        let cfg = super::functions::Config::new("EN").with_api_key("test-key");

        let request = AudioGenerationRequest {
            text: "hello".to_string(),
            voice: "default".to_string(),
            speed: 1.0,
            additional_params: None,
        };

        let error = super::functions::generate_audio(&cfg, &rt, request)
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
    use crate::providers::descriptor::{
        ApiKeyLocation, ConfigError, ProviderDescriptor, required_env_var,
    };
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

        /// Config for `model` built from the process environment.
        ///
        /// Reads `HYPERBOLIC_API_KEY` (required) — the same variable the deleted
        /// `hyperbolic::Client::from_env` read. The credential is validated eagerly
        /// but stored as [`ApiKeyLocation::Env`], so the secret is read at
        /// request time rather than held inside the config.
        ///
        /// # Errors
        /// [`ConfigError`] when a required variable is missing or invalid.
        pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
            let cfg = Self::new(model);
            required_env_var("HYPERBOLIC_API_KEY")?;
            Ok(cfg)
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
    /// `language` is taken from `cfg.model`.
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
    /// `cfg.model` carries the language rather than a model id.
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
