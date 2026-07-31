//! MiniMax API clients and Rig integrations.
//!
//! MiniMax exposes both OpenAI-compatible and Anthropic-compatible chat APIs,
//! with distinct global and China entrypoints.
//!
//! # OpenAI-compatible example
//! ```no_run
//! use rig_core::providers::minimax;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // `MINIMAX_API_BASE` overrides the base URL; pass
//! // `minimax::CHINA_API_BASE_URL` to `with_base_url` for the China entrypoint.
//! let cfg = minimax::functions::Config::from_env(minimax::MINIMAX_M2_7)?;
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//! let request = rig_core::completion::CompletionRequest::from_prompt("Hello!");
//! let response = minimax::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Anthropic-compatible example
//! ```no_run
//! use rig_core::providers::{anthropic, minimax};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Resolves `MINIMAX_ANTHROPIC_API_BASE`, then a normalized
//! // `MINIMAX_API_BASE`, then `minimax::GLOBAL_ANTHROPIC_API_BASE_URL`.
//! let cfg = minimax::functions::anthropic_config_from_env(minimax::MINIMAX_M2)?;
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//! let request = rig_core::completion::CompletionRequest::from_prompt("Hello!");
//! let response = anthropic::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```

/// Global OpenAI-compatible base URL.
pub const GLOBAL_API_BASE_URL: &str = "https://api.minimax.io/v1";
/// China OpenAI-compatible base URL.
pub const CHINA_API_BASE_URL: &str = "https://api.minimaxi.com/v1";
/// Global Anthropic-compatible base URL.
pub const GLOBAL_ANTHROPIC_API_BASE_URL: &str = "https://api.minimax.io/anthropic";
/// China Anthropic-compatible base URL.
pub const CHINA_ANTHROPIC_API_BASE_URL: &str = "https://api.minimaxi.com/anthropic";

/// `MiniMax-M2.7`
pub const MINIMAX_M2_7: &str = "MiniMax-M2.7";
/// `MiniMax-M2.7-highspeed`
pub const MINIMAX_M2_7_HIGHSPEED: &str = "MiniMax-M2.7-highspeed";
/// `MiniMax-M2.5`
pub const MINIMAX_M2_5: &str = "MiniMax-M2.5";
/// `MiniMax-M2.5-highspeed`
pub const MINIMAX_M2_5_HIGHSPEED: &str = "MiniMax-M2.5-highspeed";
/// `MiniMax-M2.1`
pub const MINIMAX_M2_1: &str = "MiniMax-M2.1";
/// `MiniMax-M2.1-highspeed`
pub const MINIMAX_M2_1_HIGHSPEED: &str = "MiniMax-M2.1-highspeed";
/// `MiniMax-M2`
pub const MINIMAX_M2: &str = "MiniMax-M2";

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
        GLOBAL_API_BASE_URL => Some(GLOBAL_ANTHROPIC_API_BASE_URL.to_owned()),
        CHINA_API_BASE_URL => Some(CHINA_ANTHROPIC_API_BASE_URL.to_owned()),
        _ => {
            let mut url = url::Url::parse(base_url).ok()?;
            if !matches!(url.path(), "/v1" | "/v1/") {
                return None;
            }
            url.set_path("/anthropic");
            Some(url.to_string())
        }
    }
}
#[cfg(test)]
mod tests {
    use super::{
        CHINA_ANTHROPIC_API_BASE_URL, CHINA_API_BASE_URL, GLOBAL_ANTHROPIC_API_BASE_URL,
        GLOBAL_API_BASE_URL, normalize_anthropic_base_url, resolve_anthropic_base_override,
    };

    #[test]
    fn normalize_openai_bases_to_anthropic_bases() {
        assert_eq!(
            normalize_anthropic_base_url(GLOBAL_API_BASE_URL).as_deref(),
            Some(GLOBAL_ANTHROPIC_API_BASE_URL)
        );
        assert_eq!(
            normalize_anthropic_base_url(CHINA_API_BASE_URL).as_deref(),
            Some(CHINA_ANTHROPIC_API_BASE_URL)
        );
        assert_eq!(
            normalize_anthropic_base_url("https://proxy.example.com/v1").as_deref(),
            Some("https://proxy.example.com/anthropic")
        );
    }

    #[test]
    fn normalize_preserves_existing_anthropic_base() {
        assert_eq!(
            normalize_anthropic_base_url(CHINA_ANTHROPIC_API_BASE_URL).as_deref(),
            Some(CHINA_ANTHROPIC_API_BASE_URL)
        );
    }

    #[test]
    fn anthropic_primary_override_wins() {
        let override_url = resolve_anthropic_base_override(
            Some("https://primary.example.com/anthropic"),
            Some(CHINA_API_BASE_URL),
        );

        assert_eq!(
            override_url.as_deref(),
            Some("https://primary.example.com/anthropic")
        );
    }
}

pub mod functions {
    //! MiniMax chat completions as config + pure functions.
    //!
    //! The data-oriented face of the MiniMax provider, mirroring
    //! [`crate::providers::openai::functions`]: a serde [`Config`], a
    //! [`DESCRIPTOR`] capability sheet, and pure
    //! [`build_request`]/[`parse_response`] free functions plus the async
    //! [`complete`]/[`open_stream`] wrappers over
    //! [`HttpRuntime`]. The request/parse
    //! mechanics are shared with the other OpenAI-compatible providers via
    //! `openai::functions`' stage functions; this module is the source of truth
    //! for MiniMax's path, body assembly, and streaming dialect.

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

    /// Default MiniMax API base URL.
    pub const DEFAULT_BASE_URL: &str = "https://api.minimax.io/v1";

    /// MiniMax's Chat Completions streaming dialect.
    pub(crate) const STREAM_DIALECT: ChatCompletionsDialect =
        ChatCompletionsDialect::from_descriptor(&DESCRIPTOR);

    /// MiniMax's capability sheet.
    pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
        name: "minimax",
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        emits_complete_single_chunk_tool_calls: false,
        composes_native_output_with_tools: true,
        max_embedding_documents: None,
        verify_path: Some("/models"),
    };

    /// Plain-data MiniMax provider configuration.
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
        /// Config for `model` reading `MINIMAX_API_KEY` from the environment.
        pub fn new(model: impl Into<String>) -> Self {
            Self {
                base_url: DEFAULT_BASE_URL.to_string(),
                api_key: ApiKeyLocation::Env("MINIMAX_API_KEY".to_string()),
                model: model.into(),
                extra_headers: Vec::new(),
            }
        }

        /// Config for `model` built from the process environment.
        ///
        /// Reads `MINIMAX_API_KEY` (required) and `MINIMAX_API_BASE` (optional
        /// override of [`DEFAULT_BASE_URL`]) — the same variables the deleted
        /// `minimax::Client::from_env` read. The credential is validated
        /// eagerly but stored as [`ApiKeyLocation::Env`], so the secret is read
        /// at request time rather than held inside the config.
        ///
        /// # Errors
        /// [`ConfigError`] when a required variable is missing or invalid.
        pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
            let mut cfg = Self::new(model);
            required_env_var("MINIMAX_API_KEY")?;
            if let Some(base_url) = optional_env_var("MINIMAX_API_BASE")? {
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
    /// MiniMax's Anthropic endpoint is reached through
    /// [`anthropic::functions`](crate::providers::anthropic::functions) with a
    /// MiniMax base URL and credential. Reads `MINIMAX_API_KEY` (required) and
    /// resolves the base URL from `MINIMAX_ANTHROPIC_API_BASE`, falling back to a
    /// normalized `MINIMAX_API_BASE`, then to
    /// [`GLOBAL_ANTHROPIC_API_BASE_URL`](super::GLOBAL_ANTHROPIC_API_BASE_URL) — the same precedence and default
    /// the classic client used.
    ///
    /// `default_max_tokens` is forced to `4096` for every model, mirroring
    /// MiniMax's `AnthropicDialect`; `anthropic_version`/`anthropic_betas` keep
    /// the Anthropic defaults, which is what the classic builder used too.
    ///
    /// # Errors
    /// [`ConfigError`] when a required variable is missing or invalid.
    pub fn anthropic_config_from_env(
        model: impl Into<String>,
    ) -> Result<anthropic_functions::Config, ConfigError> {
        let mut cfg = anthropic_functions::Config::new(model);
        required_env_var("MINIMAX_API_KEY")?;
        cfg.api_key = ApiKeyLocation::Env("MINIMAX_API_KEY".to_string());
        cfg.base_url =
            anthropic_functions::normalize_base_url(super::GLOBAL_ANTHROPIC_API_BASE_URL);
        cfg.default_max_tokens = Some(4096);
        if let Some(base_url) = super::anthropic_base_override_from_env(
            "MINIMAX_ANTHROPIC_API_BASE",
            "MINIMAX_API_BASE",
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

    /// MiniMax's straight-line chat-completions body assembly.
    ///
    /// MiniMax speaks the reference dialect: no wire-level quirks, so the body
    /// is the shared typed conversion serialized as-is.
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

    /// Send `request` to MiniMax and return the normalized response.
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
            assert_eq!(req.uri(), "https://api.minimax.io/v1/chat/completions");
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
            assert_eq!(response.provider, "minimax");
            assert_eq!(response.usage.input_tokens, 3);
            assert_eq!(response.usage.total_tokens, 5);
        }
    }
}
