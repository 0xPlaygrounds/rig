//! Moonshot AI (Kimi) API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig_core::providers::moonshot;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = moonshot::functions::Config::from_env(moonshot::KIMI_K2_5)?;
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//! let request = rig_core::completion::CompletionRequest::from_prompt("Hello!");
//! let response = moonshot::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Anthropic-compatible example
//! ```no_run
//! use rig_core::providers::{anthropic, moonshot};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Resolves `MOONSHOT_ANTHROPIC_API_BASE`, then a normalized
//! // `MOONSHOT_API_BASE`, then `moonshot::ANTHROPIC_API_BASE_URL`.
//! let cfg = moonshot::functions::anthropic_config_from_env(moonshot::KIMI_K2_5)?;
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//! let request = rig_core::completion::CompletionRequest::from_prompt("Hello!");
//! let response = anthropic::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Custom base URL
//! The default base URL is `https://api.moonshot.ai/v1`. For China access, use
//! [`CHINA_API_BASE_URL`]:
//! ```no_run
//! use rig_core::providers::moonshot;
//!
//! let cfg = moonshot::functions::Config::new(moonshot::KIMI_K2_5)
//!     .with_api_key("YOUR_API_KEY")
//!     .with_base_url(moonshot::CHINA_API_BASE_URL);
//! ```

// ================================================================
// Moonshot base URLs
// ================================================================
/// Global OpenAI-compatible base URL.
pub const GLOBAL_API_BASE_URL: &str = "https://api.moonshot.ai/v1";
/// China OpenAI-compatible base URL.
pub const CHINA_API_BASE_URL: &str = "https://api.moonshot.cn/v1";
/// Anthropic-compatible base URL.
pub const ANTHROPIC_API_BASE_URL: &str = "https://api.moonshot.ai/anthropic";

/// The Anthropic base-URL override, resolved from the process environment.
///
/// `primary_env` wins; otherwise `fallback_env` (an OpenAI-compatible base URL)
/// is mapped onto the Anthropic entrypoint by
/// [`normalize_anthropic_base_url`]. Pure logic lives in
/// [`resolve_anthropic_base_override`].
///
/// # Errors
/// [`ConfigError`](crate::providers::descriptor::ConfigError) when a variable
/// is set but invalid.
fn anthropic_base_override_from_env(
    primary_env: &'static str,
    fallback_env: &'static str,
) -> Result<Option<String>, crate::providers::descriptor::ConfigError> {
    let primary = crate::providers::descriptor::optional_env_var(primary_env)?;
    let fallback = crate::providers::descriptor::optional_env_var(fallback_env)?;

    Ok(resolve_anthropic_base_override(
        primary.as_deref(),
        fallback.as_deref(),
    ))
}

fn resolve_anthropic_base_override(
    primary: Option<&str>,
    fallback: Option<&str>,
) -> Option<String> {
    primary
        .map(str::to_owned)
        .or_else(|| fallback.and_then(normalize_anthropic_base_url))
}

fn normalize_anthropic_base_url(base_url: &str) -> Option<String> {
    if base_url.contains("/anthropic") {
        return Some(base_url.to_owned());
    }

    let mut url = url::Url::parse(base_url).ok()?;
    if !matches!(url.path(), "/v1" | "/v1/") {
        return None;
    }
    url.set_path("/anthropic");
    Some(url.to_string())
}

// ================================================================
// Moonshot Completion API
// ================================================================

/// Moonshot v1 128K context model (legacy)
pub const MOONSHOT_CHAT: &str = "moonshot-v1-128k";

/// Kimi K2 — Mixture-of-Experts model (1T total params, 32B active)
pub const KIMI_K2: &str = "kimi-k2";

/// Kimi K2.5 — Native multimodal agentic model with 256K context
pub const KIMI_K2_5: &str = "kimi-k2.5";

#[cfg(test)]
mod tests {
    use super::{functions, normalize_anthropic_base_url, resolve_anthropic_base_override};
    use crate::completion::CompletionRequest;
    use crate::message::{
        AssistantContent, Message, Reasoning, ToolCall, ToolChoice, ToolFunction,
    };
    use crate::providers::openai::completion::CompletionModelOptions;

    fn prepared_body(request: CompletionRequest, model: &str) -> serde_json::Value {
        let bytes =
            functions::build_body(model, &request, CompletionModelOptions::default(), false)
                .expect("body should build");
        serde_json::from_slice(&bytes).expect("body should be json")
    }

    #[test]
    fn moonshot_preserves_reasoning_content_in_assistant_history() {
        let assistant = Message::Assistant {
            id: None,
            content: crate::OneOrMany::many(vec![
                AssistantContent::Reasoning(Reasoning::new("tool planning")),
                AssistantContent::ToolCall(ToolCall {
                    id: "call_1".to_string(),
                    call_id: None,
                    function: ToolFunction {
                        name: "lookup".to_string(),
                        arguments: serde_json::json!({}),
                    },
                    signature: None,
                    additional_params: None,
                }),
            ])
            .expect("assistant content"),
        };

        let request = CompletionRequest {
            model: Some("kimi-k2-thinking".to_string()),
            preamble: None,
            chat_history: crate::OneOrMany::one(assistant),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let body = prepared_body(request, "kimi-k2-thinking");
        assert_eq!(
            body["messages"][0]["reasoning_content"],
            serde_json::json!("tool planning")
        );
    }

    #[test]
    fn moonshot_joins_multiple_reasoning_blocks_with_newline() {
        // A replayed assistant turn carrying two distinct reasoning blocks must
        // keep them newline-separated on the wire, not glued together.
        let assistant = Message::Assistant {
            id: None,
            content: crate::OneOrMany::many(vec![
                AssistantContent::Reasoning(Reasoning::new("first thought")),
                AssistantContent::Reasoning(Reasoning::new("second thought")),
                AssistantContent::Text("done".into()),
            ])
            .expect("assistant content"),
        };

        let request = CompletionRequest {
            model: Some("kimi-k2-thinking".to_string()),
            preamble: None,
            chat_history: crate::OneOrMany::one(assistant),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let body = prepared_body(request, "kimi-k2-thinking");
        assert_eq!(
            body["messages"][0]["reasoning_content"],
            serde_json::json!("first thought\nsecond thought")
        );
    }

    #[test]
    fn moonshot_specific_tool_choice_is_rejected() {
        let request = CompletionRequest {
            model: Some("kimi-k2.5".to_string()),
            preamble: None,
            chat_history: crate::OneOrMany::one(Message::user("Use a tool.")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: Some(ToolChoice::Specific {
                function_names: vec!["lookup".to_string()],
            }),
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let error = functions::build_body(
            "kimi-k2.5",
            &request,
            CompletionModelOptions::default(),
            false,
        )
        .expect_err("specific tool choice should be rejected");
        assert!(error.to_string().contains("specific tool"));
    }

    #[test]
    fn moonshot_required_tool_choice_is_coerced() {
        let request = CompletionRequest {
            model: Some("kimi-k2.5".to_string()),
            preamble: None,
            chat_history: crate::OneOrMany::one(Message::user("Use a tool.")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: Some(ToolChoice::Required),
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let body = prepared_body(request, "kimi-k2.5");
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(
            body["messages"]
                .as_array()
                .and_then(|messages| messages.last())
                .and_then(|message| message.get("content"))
                .and_then(|content| content.as_str()),
            Some("Please select a tool to handle the current issue.")
        );
    }

    #[test]
    fn normalize_openai_style_base_to_anthropic_base() {
        assert_eq!(
            normalize_anthropic_base_url("https://api.moonshot.ai/v1").as_deref(),
            Some("https://api.moonshot.ai/anthropic")
        );
        assert_eq!(
            normalize_anthropic_base_url("https://api.moonshot.cn/v1").as_deref(),
            Some("https://api.moonshot.cn/anthropic")
        );
        assert_eq!(
            normalize_anthropic_base_url("https://proxy.example.com/v1").as_deref(),
            Some("https://proxy.example.com/anthropic")
        );
    }

    #[test]
    fn normalize_preserves_existing_anthropic_base() {
        assert_eq!(
            normalize_anthropic_base_url("https://proxy.example.com/anthropic").as_deref(),
            Some("https://proxy.example.com/anthropic")
        );
    }

    #[test]
    fn anthropic_primary_override_wins() {
        let override_url = resolve_anthropic_base_override(
            Some("https://primary.example.com/anthropic"),
            Some("https://api.moonshot.cn/v1"),
        );

        assert_eq!(
            override_url.as_deref(),
            Some("https://primary.example.com/anthropic")
        );
    }
}

pub mod functions {
    //! Moonshot chat completions as config + pure functions.
    //!
    //! The data-oriented face of the Moonshot provider, mirroring
    //! [`crate::providers::openai::functions`]: a serde [`Config`], a
    //! [`DESCRIPTOR`] capability sheet, and pure
    //! [`build_request`]/[`parse_response`] free functions plus the async
    //! [`complete`]/[`open_stream`] wrappers over
    //! [`HttpRuntime`](crate::http_runtime::HttpRuntime). The request/parse
    //! mechanics are shared with the other OpenAI-compatible providers via
    //! `openai::functions`'s stage helpers; this module owns Moonshot's own
    //! dialect steps, paths, and provider name.

    use serde::{Deserialize, Serialize};

    use crate::completion::{self, CompletionError, CompletionRequest};
    use crate::http_runtime::HttpRuntime;
    use crate::providers::anthropic::functions as anthropic_functions;
    use crate::providers::descriptor::ChatCompletionsDialect;
    use crate::providers::descriptor::{
        ApiKeyLocation, ConfigError, ProviderDescriptor, optional_env_var, required_env_var,
    };
    use crate::providers::openai;
    use crate::providers::openai::completion::CompletionModelOptions;
    use crate::providers::openai::functions as openai_functions;

    /// Default Moonshot API base URL.
    pub const DEFAULT_BASE_URL: &str = "https://api.moonshot.ai/v1";

    /// Moonshot's Chat Completions streaming dialect.
    pub(crate) const STREAM_DIALECT: ChatCompletionsDialect =
        ChatCompletionsDialect::from_descriptor(&DESCRIPTOR);

    /// Moonshot's capability sheet.
    pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
        name: "moonshot",
        supports_tools: true,
        supports_response_format: false,
        stream_include_usage: true,
        emits_complete_single_chunk_tool_calls: false,
        composes_native_output_with_tools: false,
        max_embedding_documents: None,
    };

    /// Plain-data Moonshot provider configuration.
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
        /// Config for `model` reading `MOONSHOT_API_KEY` from the environment.
        pub fn new(model: impl Into<String>) -> Self {
            Self {
                base_url: DEFAULT_BASE_URL.to_string(),
                api_key: ApiKeyLocation::Env("MOONSHOT_API_KEY".to_string()),
                model: model.into(),
                extra_headers: Vec::new(),
            }
        }

        /// Config for `model` built from the process environment.
        ///
        /// Reads `MOONSHOT_API_KEY` (required) and `MOONSHOT_API_BASE` (optional
        /// override of [`DEFAULT_BASE_URL`]) — the same variables the deleted
        /// `moonshot::Client::from_env` read. The credential is validated
        /// eagerly but stored as [`ApiKeyLocation::Env`], so the secret is read
        /// at request time rather than held inside the config.
        ///
        /// # Errors
        /// [`ConfigError`] when a required variable is missing or invalid.
        pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
            let mut cfg = Self::new(model);
            required_env_var("MOONSHOT_API_KEY")?;
            if let Some(base_url) = optional_env_var("MOONSHOT_API_BASE")? {
                cfg.base_url = base_url;
            }
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
    pub(crate) fn completion_path(_model: &str) -> String {
        "/chat/completions".to_string()
    }

    /// Moonshot's straight-line chat-completions body assembly.
    ///
    /// Two tool-choice quirks: forcing one specific tool is unsupported and
    /// fails fast, and `tool_choice: "required"` is coerced to `auto` with an
    /// extra steering user message appended. Moonshot's API also rejects the
    /// `json_schema` response format, so `output_schema` is dropped during the
    /// typed conversion (see [`DESCRIPTOR`]).
    pub(crate) fn build_body(
        model: &str,
        request: &CompletionRequest,
        options: CompletionModelOptions,
        stream: bool,
    ) -> Result<Vec<u8>, CompletionError> {
        let mut typed =
            openai_functions::compatible_typed_request(model, request, &DESCRIPTOR, options)?;
        // Moonshot only supports `auto`/`none` tool choices. Forcing one
        // specific tool has no workaround; fail fast like the pre-migration
        // conversion did (on main, `openai::ToolChoice::try_from` returned
        // "Provider doesn't support only using specific tools" for every
        // `ToolChoice::Specific`, single- or multi-name).
        if matches!(
            typed.tool_choice,
            Some(openai::completion::ToolChoice::Function { .. })
        ) {
            return Err(CompletionError::ProviderError(
                "Moonshot does not support forcing a specific tool".to_string(),
            ));
        }

        // Moonshot does not support `tool_choice: "required"`; coerce it to
        // `auto` and steer the model with an extra user message instead.
        if matches!(
            typed.tool_choice,
            Some(openai::completion::ToolChoice::Required)
        ) {
            tracing::warn!(
                "Moonshot does not support tool_choice=required; coercing to auto with an additional steering message"
            );
            typed.tool_choice = Some(openai::completion::ToolChoice::Auto);
            typed.messages.push(openai::Message::User {
                content: crate::OneOrMany::one(openai::UserContent::Text {
                    text: "Please select a tool to handle the current issue.".to_string(),
                }),
                name: None,
            });
        }

        let body = openai_functions::compatible_body_value(&typed, &DESCRIPTOR, stream)?;
        Ok(serde_json::to_vec(&body)?)
    }

    /// Anthropic-Messages surface configuration for `model`, built from the
    /// process environment.
    ///
    /// The replacement for the deleted classic `AnthropicClient`:
    /// Moonshot's Anthropic endpoint is reached through
    /// [`anthropic::functions`](crate::providers::anthropic::functions) with a
    /// Moonshot base URL and credential. Reads `MOONSHOT_API_KEY` (required) and
    /// resolves the base URL from `MOONSHOT_ANTHROPIC_API_BASE`, falling back to a
    /// normalized `MOONSHOT_API_BASE`, then to
    /// [`ANTHROPIC_API_BASE_URL`](super::ANTHROPIC_API_BASE_URL) — the same precedence and default
    /// the classic client used.
    ///
    /// `default_max_tokens` is forced to `4096` for every model, mirroring
    /// Moonshot's `AnthropicDialect`; `anthropic_version`/`anthropic_betas` keep
    /// the Anthropic defaults, which is what the classic builder used too.
    ///
    /// # Errors
    /// [`ConfigError`] when a required variable is missing or invalid.
    pub fn anthropic_config_from_env(
        model: impl Into<String>,
    ) -> Result<anthropic_functions::Config, ConfigError> {
        let mut cfg = anthropic_functions::Config::new(model);
        required_env_var("MOONSHOT_API_KEY")?;
        cfg.api_key = ApiKeyLocation::Env("MOONSHOT_API_KEY".to_string());
        cfg.base_url = anthropic_functions::normalize_base_url(super::ANTHROPIC_API_BASE_URL);
        cfg.default_max_tokens = Some(4096);
        if let Some(base_url) = super::anthropic_base_override_from_env(
            "MOONSHOT_ANTHROPIC_API_BASE",
            "MOONSHOT_API_BASE",
        )? {
            cfg.base_url = anthropic_functions::normalize_base_url(&base_url);
        }
        Ok(cfg)
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
        openai_functions::compatible_parse_response::<openai::CompletionResponse>(
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

    /// Send `request` to Moonshot and return the normalized response.
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
            assert_eq!(req.uri(), "https://api.moonshot.ai/v1/chat/completions");
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
            assert_eq!(response.provider, "moonshot");
            assert_eq!(response.usage.input_tokens, 3);
            assert_eq!(response.usage.total_tokens, 5);
        }
    }
}
