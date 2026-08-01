//! Groq API integration.
//!
//! # Example
//! ```no_run
//! use rig_core::providers::groq;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! # let request = rig_core::completion::CompletionRequest::from_prompt("hello");
//! let cfg = groq::functions::Config::from_env(groq::LLAMA_3_1_8B_INSTANT)?;
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//! let response = groq::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

fn apply_native_tools_to_additional_params(
    extra: &mut Map<String, Value>,
    native_tools: Vec<Value>,
) {
    if native_tools.is_empty() {
        return;
    }

    let mut compound_custom = match extra.remove("compound_custom") {
        Some(Value::Object(map)) => map,
        _ => Map::new(),
    };

    let mut enabled_tools = match compound_custom.remove("enabled_tools") {
        Some(Value::Array(values)) => values,
        _ => Vec::new(),
    };

    for native_tool in native_tools {
        let already_enabled = enabled_tools
            .iter()
            .any(|existing| native_tools_match(existing, &native_tool));
        if !already_enabled {
            enabled_tools.push(native_tool);
        }
    }

    compound_custom.insert("enabled_tools".to_string(), Value::Array(enabled_tools));
    extra.insert(
        "compound_custom".to_string(),
        Value::Object(compound_custom),
    );
}

fn native_tools_match(lhs: &Value, rhs: &Value) -> bool {
    if let (Some(lhs_type), Some(rhs_type)) = (native_tool_kind(lhs), native_tool_kind(rhs)) {
        return lhs_type == rhs_type;
    }

    lhs == rhs
}

fn native_tool_kind(value: &Value) -> Option<&str> {
    match value {
        Value::String(kind) => Some(kind),
        Value::Object(map) => map.get("type").and_then(Value::as_str),
        _ => None,
    }
}

// ================================================================
// Groq Completion API
// ================================================================

/// The `deepseek-r1-distill-llama-70b` model. Used for chat completion.
pub const DEEPSEEK_R1_DISTILL_LLAMA_70B: &str = "deepseek-r1-distill-llama-70b";
/// The `gemma2-9b-it` model. Used for chat completion.
pub const GEMMA2_9B_IT: &str = "gemma2-9b-it";
/// The `llama-3.1-8b-instant` model. Used for chat completion.
pub const LLAMA_3_1_8B_INSTANT: &str = "llama-3.1-8b-instant";
/// The `llama-3.2-11b-vision-preview` model. Used for chat completion.
pub const LLAMA_3_2_11B_VISION_PREVIEW: &str = "llama-3.2-11b-vision-preview";
/// The `llama-3.2-1b-preview` model. Used for chat completion.
pub const LLAMA_3_2_1B_PREVIEW: &str = "llama-3.2-1b-preview";
/// The `llama-3.2-3b-preview` model. Used for chat completion.
pub const LLAMA_3_2_3B_PREVIEW: &str = "llama-3.2-3b-preview";
/// The `llama-3.2-90b-vision-preview` model. Used for chat completion.
pub const LLAMA_3_2_90B_VISION_PREVIEW: &str = "llama-3.2-90b-vision-preview";
/// The `llama-3.2-70b-specdec` model. Used for chat completion.
pub const LLAMA_3_2_70B_SPECDEC: &str = "llama-3.2-70b-specdec";
/// The `llama-3.2-70b-versatile` model. Used for chat completion.
pub const LLAMA_3_2_70B_VERSATILE: &str = "llama-3.2-70b-versatile";
/// The `llama-guard-3-8b` model. Used for chat completion.
pub const LLAMA_GUARD_3_8B: &str = "llama-guard-3-8b";
/// The `llama3-70b-8192` model. Used for chat completion.
pub const LLAMA_3_70B_8192: &str = "llama3-70b-8192";
/// The `llama3-8b-8192` model. Used for chat completion.
pub const LLAMA_3_8B_8192: &str = "llama3-8b-8192";
/// The `mixtral-8x7b-32768` model. Used for chat completion.
pub const MIXTRAL_8X7B_32768: &str = "mixtral-8x7b-32768";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningFormat {
    Parsed,
    Raw,
    Hidden,
}

/// Additional parameters to send to the Groq API. Serialize this into the
/// request's `additional_params` to set Groq's reasoning options.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GroqAdditionalParameters {
    /// The reasoning format. See Groq's API docs for more details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_format: Option<ReasoningFormat>,
    /// Whether or not to include reasoning. See Groq's API docs for more details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_reasoning: Option<bool>,
    /// Any other properties not included by default on this struct (that you want to send)
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Map<String, serde_json::Value>>,
}

// ================================================================
// Groq Transcription API
// ================================================================

pub const WHISPER_LARGE_V3: &str = "whisper-large-v3";
pub const WHISPER_LARGE_V3_TURBO: &str = "whisper-large-v3-turbo";
pub const DISTIL_WHISPER_LARGE_V3_EN: &str = "distil-whisper-large-v3-en";
#[cfg(test)]
mod tests {
    use crate::completion::CompletionRequest;
    use crate::providers::openai::completion::{
        CompletionModelOptions, CompletionRequest as OpenAICompletionRequest, OpenAIRequestParams,
    };

    #[test]
    fn groq_request_maps_output_schema_max_tokens_and_specific_tool_choice() {
        let request = CompletionRequest::builder("Return JSON")
            .max_tokens(64)
            .tools(vec![crate::completion::ToolDefinition {
                name: "choose_beta".to_string(),
                description: "Choose beta".to_string(),
                parameters: serde_json::json!({"type":"object","properties":{},"required":[]}),
            }])
            .tool_choice(crate::message::ToolChoice::Specific {
                function_names: vec!["choose_beta".to_string()],
            })
            .output_schema(schemars::schema_for!(serde_json::Value))
            .build();

        let request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
            model: "llama-3.3-70b-versatile".to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
            supports_response_format: true,
            supports_tools: true,
        })
        .expect("Groq request should convert");
        let json = serde_json::to_value(request).expect("request should serialize");

        assert_eq!(json["max_tokens"], 64);
        assert_eq!(
            json["tool_choice"],
            serde_json::json!({"type":"function","function":{"name":"choose_beta"}})
        );
        // The shared path defers `response_format` while tools are present and
        // no tool result exists yet (see `should_apply_response_format`).
        assert_eq!(json["response_format"], serde_json::Value::Null);

        let no_tools_request = CompletionRequest {
            output_schema: Some(schemars::schema_for!(serde_json::Value)),
            ..CompletionRequest::from_prompt("Return JSON")
        };
        let no_tools_request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
            model: "llama-3.3-70b-versatile".to_string(),
            request: no_tools_request,
            strict_tools: false,
            tool_result_array_content: false,
            supports_response_format: true,
            supports_tools: true,
        })
        .expect("request should convert");
        let json = serde_json::to_value(no_tools_request).expect("request should serialize");
        assert_eq!(json["response_format"]["type"], "json_schema");
        assert_eq!(json["response_format"]["json_schema"]["strict"], true);
    }

    #[test]
    fn groq_build_body_merges_native_tools_into_compound_custom() {
        let request = CompletionRequest::builder("search")
            .tools(vec![crate::completion::ToolDefinition {
                name: "local_tool".to_string(),
                description: "A local function tool".to_string(),
                parameters: serde_json::json!({"type":"object","properties":{},"required":[]}),
            }])
            .additional_params(serde_json::json!({
                "tools": [{"type": "browser_search"}, {"type": "browser_search"}],
            }))
            .build();

        let body = super::functions::build_body(
            "llama-3.3-70b-versatile",
            &request,
            CompletionModelOptions::default(),
            false,
        )
        .expect("build_body should succeed");

        let json: serde_json::Value = serde_json::from_slice(&body).expect("body should be JSON");
        assert_eq!(
            json["compound_custom"]["enabled_tools"],
            serde_json::json!([{"type": "browser_search"}])
        );
        // The rig-level function tool array must survive the native-tool merge.
        assert_eq!(json["tools"][0]["function"]["name"], "local_tool");
    }

    #[test]
    fn groq_reasoning_params_flatten_into_request_body() {
        let additional_params = serde_json::to_value(super::GroqAdditionalParameters {
            reasoning_format: Some(super::ReasoningFormat::Parsed),
            include_reasoning: Some(true),
            extra: None,
        })
        .expect("params should serialize");
        let request = CompletionRequest {
            additional_params: Some(additional_params),
            ..CompletionRequest::from_prompt("Think about it")
        };

        let request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
            model: "llama-3.3-70b-versatile".to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
            supports_response_format: true,
            supports_tools: true,
        })
        .expect("request should convert");
        let json = serde_json::to_value(request).expect("request should serialize");

        assert_eq!(json["reasoning_format"], "parsed");
        assert_eq!(json["include_reasoning"], true);
    }

    #[tokio::test]
    async fn completion_preserves_raw_provider_error_json_on_api_error_envelope() {
        use crate::completion::CompletionError;
        use crate::http_runtime::HttpRuntime;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"message":"model overloaded","type":"server_error","code":"503"}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::with_error_response(
            http::StatusCode::ACCEPTED,
            body,
        ));
        let cfg = super::functions::Config::new("llama-3.3-70b-versatile").with_api_key("test-key");
        let request = crate::completion::CompletionRequest::from_prompt("hello");

        let error = super::functions::complete(&cfg, &rt, request)
            .await
            .expect_err("completion should fail with provider error envelope");

        match &error {
            CompletionError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::ACCEPTED));
                assert_eq!(error.provider_response_body(), Some(body));
                let json = error
                    .provider_response_json()
                    .expect("raw body should be valid JSON")
                    .expect("parsed JSON should be present");
                assert_eq!(json["code"], "503");
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn completion_http_non_success_preserves_status_and_body() {
        use crate::completion::CompletionError;
        use crate::http_runtime::HttpRuntime;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"service unavailable","code":"503"}}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::with_error_response(
            http::StatusCode::SERVICE_UNAVAILABLE,
            body,
        ));
        let cfg = super::functions::Config::new("llama-3.3-70b-versatile").with_api_key("test-key");
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
    async fn transcription_http_non_success_preserves_status_and_body() {
        use crate::http_runtime::HttpRuntime;
        use crate::test_utils::RecordingHttpClient;
        use crate::transcription::{TranscriptionError, TranscriptionRequest};

        let body = r#"{"error":{"message":"bad audio","code":"400"}}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::with_error_response(
            http::StatusCode::BAD_REQUEST,
            body,
        ));
        let cfg = super::functions::Config::new("whisper-large-v3").with_api_key("test-key");

        let error =
            match super::functions::transcribe(&cfg, &rt, TranscriptionRequest::new(vec![0u8; 16]))
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
}

crate::providers::client::define_http_client! {
    config = functions::Config,
    default_base_url = functions::DEFAULT_BASE_URL,
    api_key_required = true,
}

pub mod functions {
    //! Groq chat completions as config + pure functions.
    //!
    //! The data-oriented face of the Groq provider, mirroring
    //! [`crate::providers::openai::functions`]: a serde [`Config`], a
    //! [`DESCRIPTOR`] capability sheet, and pure
    //! [`build_request`]/[`parse_response`] free functions plus the async
    //! [`complete`]/[`open_stream`] wrappers over
    //! [`HttpRuntime`]. The request/parse
    //! mechanics are shared with the other OpenAI-compatible providers via
    //! `openai::functions`; this module instantiates them with Groq's
    //! [`DESCRIPTOR`] so Groq's paths, hooks, and provider name apply.

    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    use crate::completion::{self, CompletionError, CompletionRequest};
    use crate::http_runtime::HttpRuntime;
    use crate::providers::descriptor::ChatCompletionsDialect;
    use crate::providers::descriptor::{
        ApiKeyLocation, ConfigError, ProviderDescriptor, required_env_var,
    };
    use crate::providers::openai::completion::CompletionModelOptions;
    use crate::providers::openai::functions as openai_functions;

    /// Default Groq API base URL.
    pub const DEFAULT_BASE_URL: &str = "https://api.groq.com/openai/v1";

    /// Groq's Chat Completions streaming dialect.
    pub(crate) const STREAM_DIALECT: ChatCompletionsDialect =
        ChatCompletionsDialect::from_descriptor(&DESCRIPTOR);

    /// Groq's capability sheet.
    pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
        name: "groq",
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        emits_complete_single_chunk_tool_calls: true,
        composes_native_output_with_tools: true,
        max_embedding_documents: None,
        verify_path: Some("/models"),
    };

    /// Plain-data Groq provider configuration.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[non_exhaustive]
    pub struct Config {
        /// Reusable HTTP connection data.
        #[serde(flatten)]
        pub connection: crate::providers::HttpConnectionConfig,
        /// Model identifier requests are built for.
        pub model: String,
    }

    crate::providers::client::impl_http_connection_config!(Config);

    impl Config {
        /// Config for `model` reading `GROQ_API_KEY` from the environment.
        pub fn new(model: impl Into<String>) -> Self {
            Self {
                connection: crate::providers::HttpConnectionConfig::new(
                    DEFAULT_BASE_URL.to_string(),
                    ApiKeyLocation::Env("GROQ_API_KEY".to_string()),
                ),
                model: model.into(),
            }
        }

        /// Config for `model` built from the process environment.
        ///
        /// Reads `GROQ_API_KEY` (required) — the same variable the deleted
        /// `groq::Client::from_env` read. The credential is validated eagerly
        /// but stored as [`ApiKeyLocation::Env`], so the secret is read at
        /// request time rather than held inside the config.
        ///
        /// # Errors
        /// [`ConfigError`] when a required variable is missing or invalid.
        pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
            let cfg = Self::new(model);
            required_env_var("GROQ_API_KEY")?;
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

    /// The chat-completions request path for `model`.
    pub(crate) fn completion_path(_model: &str) -> String {
        "/chat/completions".to_string()
    }

    /// Groq's straight-line chat-completions body assembly.
    ///
    /// One dialect quirk: Groq's provider-native tools (`browser_search`,
    /// `code_interpreter`, ...) arrive via `additional_params.tools`, where they
    /// would clobber the function-tool array on serialization, so they are folded
    /// into `compound_custom.enabled_tools` (deduplicated by tool type).
    pub(crate) fn build_body(
        model: &str,
        request: &CompletionRequest,
        options: CompletionModelOptions,
        stream: bool,
    ) -> Result<Vec<u8>, CompletionError> {
        let mut typed =
            openai_functions::compatible_typed_request(model, request, &DESCRIPTOR, options)?;
        // Groq's provider-native tools (`browser_search`, `code_interpreter`,
        // ...) arrive via `additional_params.tools`. Left in place they would
        // clobber the function-tool array on serialization, so fold them into
        // `compound_custom.enabled_tools` (deduplicated by tool type).
        if let Some(map) = typed
            .additional_params
            .as_mut()
            .and_then(Value::as_object_mut)
            && let Some(raw_tools) = map.remove("tools")
        {
            let native_tools = serde_json::from_value::<Vec<Value>>(raw_tools).map_err(|err| {
                CompletionError::RequestError(
                    format!("Invalid Groq `additional_params.tools` payload: {err}").into(),
                )
            })?;
            super::apply_native_tools_to_additional_params(map, native_tools);
        }
        let body = openai_functions::compatible_body_value(&typed, &DESCRIPTOR, stream)?;
        Ok(serde_json::to_vec(&body)?)
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

    /// Transcribe `request` with Groq's OpenAI-compatible
    /// `/audio/transcriptions` endpoint.
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
        use crate::transcription::TranscriptionError;

        let form = openai_functions::build_transcription_form(&cfg.model, request)?;
        let url = format!(
            "{}/audio/transcriptions",
            cfg.base_url.trim_end_matches('/')
        );
        let mut builder = http::Request::post(url);
        if let Some(key) = cfg
            .api_key
            .resolve()
            .map_err(|e| TranscriptionError::RequestError(Box::new(e)))?
        {
            builder = builder.header(http::header::AUTHORIZATION, format!("Bearer {key}"));
        }
        for (name, value) in &cfg.extra_headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        let req = builder
            .body(form)
            .map_err(|e| TranscriptionError::RequestError(Box::new(e)))?;
        let (status, body) = rt.send_multipart(req).await?;
        openai_functions::parse_transcription_response(status, &body)
    }

    /// Send `request` to Groq and return the normalized response.
    pub async fn complete(
        cfg: &Config,
        rt: &HttpRuntime,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        let req = build_request(cfg, &request, false)?;
        let (status, body) = rt.send(req).await?;
        parse_response(status, &body)
    }

    /// Verify that `cfg`'s credential is accepted by the provider.
    ///
    /// The data-oriented replacement for the deleted `VerifyClient::verify`: the
    /// endpoint is [`DESCRIPTOR`]'s `verify_path` (`/models`, the value the
    /// deleted `Provider::VERIFY_PATH` carried) and the status mapping is the
    /// classic one — see [`crate::providers::verify`].
    ///
    /// # Errors
    /// [`VerifyError`](crate::providers::verify::VerifyError): invalid
    /// authentication on `401`/`403`, otherwise the preserved provider response
    /// or a transport failure.
    pub async fn verify(
        cfg: &Config,
        rt: &HttpRuntime,
    ) -> Result<(), crate::providers::verify::VerifyError> {
        crate::providers::verify::verify_bearer(
            &DESCRIPTOR,
            &cfg.base_url,
            &cfg.api_key,
            &cfg.extra_headers,
            rt,
        )
        .await
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::OneOrMany;

        fn sample_request() -> CompletionRequest {
            CompletionRequest {
                model: None,
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
            assert_eq!(req.uri(), "https://api.groq.com/openai/v1/chat/completions");
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
            assert_eq!(response.provider, "groq");
            assert_eq!(response.usage.input_tokens, 3);
            assert_eq!(response.usage.total_tokens, 5);
        }
    }
}
