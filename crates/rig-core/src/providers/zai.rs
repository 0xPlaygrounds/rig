//! Z.AI API clients and Rig integrations.
//!
//! Z.AI exposes OpenAI-compatible APIs for both its general platform and
//! coding-focused platform, plus an Anthropic-compatible endpoint for tools
//! like Claude Code.
//!
//! # OpenAI-compatible example
//! ```no_run
//! use rig_core::providers::zai;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // `ZAI_API_BASE` overrides the base URL; pass `zai::CODING_API_BASE_URL`
//! // to `with_base_url` for the coding platform.
//! let cfg = zai::functions::Config::from_env(zai::GLM_4_6)?;
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//! let request = rig_core::completion::CompletionRequest::from_prompt("Hello!");
//! let response = zai::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Anthropic-compatible example
//! ```no_run
//! use rig_core::providers::{anthropic, zai};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Resolves `ZAI_ANTHROPIC_API_BASE`, then a normalized `ZAI_API_BASE`,
//! // then `zai::ANTHROPIC_API_BASE_URL`.
//! let cfg = zai::functions::anthropic_config_from_env(zai::GLM_4_6)?;
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//! let request = rig_core::completion::CompletionRequest::from_prompt("Hello!");
//! let response = anthropic::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```

/// General-purpose OpenAI-compatible base URL.
pub const GENERAL_API_BASE_URL: &str = "https://api.z.ai/api/paas/v4";
/// Coding-focused OpenAI-compatible base URL.
pub const CODING_API_BASE_URL: &str = "https://api.z.ai/api/coding/paas/v4";
/// Anthropic-compatible base URL.
pub const ANTHROPIC_API_BASE_URL: &str = "https://api.z.ai/api/anthropic";

/// `glm-4.6`
pub const GLM_4_6: &str = "glm-4.6";
/// `glm-4.6-air`
pub const GLM_4_6_AIR: &str = "glm-4.6-air";
/// `glm-4.6-x`
pub const GLM_4_6_X: &str = "glm-4.6-x";
/// `glm-4.5`
pub const GLM_4_5: &str = "glm-4.5";
/// `glm-4.5-air`
pub const GLM_4_5_AIR: &str = "glm-4.5-air";
/// `glm-4.5v`
pub const GLM_4_5V: &str = "glm-4.5v";
/// `glm-4.5-airx`
pub const GLM_4_5_AIRX: &str = "glm-4.5-airx";

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

    match base_url.trim_end_matches('/') {
        GENERAL_API_BASE_URL | CODING_API_BASE_URL => Some(ANTHROPIC_API_BASE_URL.to_owned()),
        _ => {
            let mut url = url::Url::parse(base_url).ok()?;
            if !matches!(
                url.path(),
                "/api/paas/v4" | "/api/paas/v4/" | "/api/coding/paas/v4" | "/api/coding/paas/v4/"
            ) {
                return None;
            }
            url.set_path("/api/anthropic");
            Some(url.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ANTHROPIC_API_BASE_URL, CODING_API_BASE_URL, GENERAL_API_BASE_URL,
        normalize_anthropic_base_url, resolve_anthropic_base_override,
    };

    #[test]
    fn normalize_openai_style_bases_to_anthropic_base() {
        assert_eq!(
            normalize_anthropic_base_url(GENERAL_API_BASE_URL).as_deref(),
            Some(ANTHROPIC_API_BASE_URL)
        );
        assert_eq!(
            normalize_anthropic_base_url(CODING_API_BASE_URL).as_deref(),
            Some(ANTHROPIC_API_BASE_URL)
        );
        assert_eq!(
            normalize_anthropic_base_url("https://proxy.example.com/api/paas/v4").as_deref(),
            Some("https://proxy.example.com/api/anthropic")
        );
        assert_eq!(
            normalize_anthropic_base_url("https://proxy.example.com/api/coding/paas/v4").as_deref(),
            Some("https://proxy.example.com/api/anthropic")
        );
    }

    #[test]
    fn normalize_preserves_existing_anthropic_base() {
        assert_eq!(
            normalize_anthropic_base_url("https://proxy.example.com/api/anthropic").as_deref(),
            Some("https://proxy.example.com/api/anthropic")
        );
    }

    #[test]
    fn anthropic_primary_override_wins() {
        let override_url = resolve_anthropic_base_override(
            Some("https://primary.example.com/api/anthropic"),
            Some(GENERAL_API_BASE_URL),
        );

        assert_eq!(
            override_url.as_deref(),
            Some("https://primary.example.com/api/anthropic")
        );
    }
}

pub mod functions {
    //! Z.ai chat completions as config + pure functions.
    //!
    //! The data-oriented face of the Z.ai provider, mirroring
    //! [`crate::providers::openai::functions`]: a serde [`Config`], a
    //! [`DESCRIPTOR`] capability sheet, and pure
    //! [`build_request`]/[`parse_response`] free functions plus the async
    //! [`complete`]/[`open_stream`] wrappers over
    //! [`HttpRuntime`](crate::http_runtime::HttpRuntime). The request/parse
    //! mechanics are shared with the other OpenAI-compatible providers via
    //! `openai::functions`' stage functions; this module is the source of truth
    //! for Z.ai's path, body assembly, and streaming dialect.

    use serde::{Deserialize, Serialize};

    use crate::completion::{self, CompletionError, CompletionRequest};
    use crate::http_runtime::HttpRuntime;
    use crate::providers::anthropic::functions as anthropic_functions;
    use crate::providers::descriptor::ChatCompletionsDialect;
    use crate::providers::descriptor::{
        ApiKeyLocation, ConfigError, ProviderDescriptor, optional_env_var, required_env_var,
    };
    use crate::providers::openai::completion::CompletionModelOptions;
    use crate::providers::openai::functions as openai_functions;

    /// Default Z.ai API base URL.
    pub const DEFAULT_BASE_URL: &str = "https://api.z.ai/api/paas/v4";

    /// Z.ai's Chat Completions streaming dialect.
    pub(crate) const STREAM_DIALECT: ChatCompletionsDialect =
        ChatCompletionsDialect::from_descriptor(&DESCRIPTOR);

    /// Z.ai's capability sheet.
    pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
        name: "zai",
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        emits_complete_single_chunk_tool_calls: false,
        composes_native_output_with_tools: true,
        max_embedding_documents: None,
    };

    /// Plain-data Z.ai provider configuration.
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
        /// Config for `model` reading `ZAI_API_KEY` from the environment.
        pub fn new(model: impl Into<String>) -> Self {
            Self {
                base_url: DEFAULT_BASE_URL.to_string(),
                api_key: ApiKeyLocation::Env("ZAI_API_KEY".to_string()),
                model: model.into(),
                extra_headers: Vec::new(),
            }
        }

        /// Config for `model` built from the process environment.
        ///
        /// Reads `ZAI_API_KEY` (required) and `ZAI_API_BASE` (optional
        /// override of [`DEFAULT_BASE_URL`]) — the same variables the deleted
        /// `zai::Client::from_env` read. The credential is validated eagerly
        /// but stored as [`ApiKeyLocation::Env`], so the secret is read at
        /// request time rather than held inside the config.
        ///
        /// # Errors
        /// [`ConfigError`] when a required variable is missing or invalid.
        pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
            let mut cfg = Self::new(model);
            required_env_var("ZAI_API_KEY")?;
            if let Some(base_url) = optional_env_var("ZAI_API_BASE")? {
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

    /// Anthropic-Messages surface configuration for `model`, built from the
    /// process environment.
    ///
    /// The replacement for the deleted classic `AnthropicClient`:
    /// Z.ai's Anthropic endpoint is reached through
    /// [`anthropic::functions`](crate::providers::anthropic::functions) with a
    /// Z.ai base URL and credential. Reads `ZAI_API_KEY` (required) and
    /// resolves the base URL from `ZAI_ANTHROPIC_API_BASE`, falling back to a
    /// normalized `ZAI_API_BASE`, then to
    /// [`ANTHROPIC_API_BASE_URL`](super::ANTHROPIC_API_BASE_URL) — the same
    /// precedence and default the classic client used.
    ///
    /// `default_max_tokens` is forced to `4096` for every model, mirroring
    /// Z.ai's `AnthropicDialect`; `anthropic_version`/`anthropic_betas` keep
    /// the Anthropic defaults, which is what the classic builder used too.
    ///
    /// # Errors
    /// [`ConfigError`] when a required variable is missing or invalid.
    pub fn anthropic_config_from_env(
        model: impl Into<String>,
    ) -> Result<anthropic_functions::Config, ConfigError> {
        let mut cfg = anthropic_functions::Config::new(model);
        required_env_var("ZAI_API_KEY")?;
        cfg.api_key = ApiKeyLocation::Env("ZAI_API_KEY".to_string());
        cfg.base_url = anthropic_functions::normalize_base_url(super::ANTHROPIC_API_BASE_URL);
        cfg.default_max_tokens = Some(4096);
        if let Some(base_url) =
            super::anthropic_base_override_from_env("ZAI_ANTHROPIC_API_BASE", "ZAI_API_BASE")?
        {
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

    /// Z.ai's straight-line chat-completions body assembly.
    ///
    /// Z.ai speaks the reference dialect: no wire-level quirks, so the body is
    /// the shared typed conversion serialized as-is.
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

    /// The chat-completions request path for `model`.
    pub(crate) fn completion_path(_model: &str) -> String {
        "/chat/completions".to_string()
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

    /// Send `request` to Z.ai and return the normalized response.
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
            assert_eq!(req.uri(), "https://api.z.ai/api/paas/v4/chat/completions");
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
            assert_eq!(response.provider, "zai");
            assert_eq!(response.usage.input_tokens, 3);
            assert_eq!(response.usage.total_tokens, 5);
        }
    }
}
