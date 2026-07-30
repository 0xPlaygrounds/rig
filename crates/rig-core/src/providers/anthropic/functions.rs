//! Anthropic Messages API as config + pure functions.
//!
//! The data-oriented face of the Anthropic provider: a serde [`Config`], a
//! [`DESCRIPTOR`] capability sheet, and free functions decomposing a
//! completion into pure parts — [`build_request_body`] / [`build_request`]
//! (data → HTTP request, no IO) and [`parse_response`] (bytes → normalized
//! [`completion::CompletionResponse`], no IO) — plus async [`complete`] and
//! [`open_stream`] wrappers over [`HttpRuntime`].
//!
//! The pure functions delegate to the typed wire conversions in
//! [`super::completion`] (`AnthropicCompletionRequest`'s `TryFrom` for
//! blocking bodies, `create_streaming_request_body` for streaming bodies), and
//! [`open_stream`] drives the shared SSE state machine
//! (`streaming::stream_anthropic_sse`).

use http::header::CONTENT_TYPE;
use serde::{Deserialize, Serialize};

use super::completion::{
    ANTHROPIC_VERSION_LATEST, AnthropicCompletionRequest, AnthropicRequestParams, ApiResponse,
    CacheTtl, default_max_tokens_for_model,
};
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_runtime::HttpRuntime;
use crate::providers::descriptor::{
    ApiKeyLocation, ConfigError, ProviderDescriptor, optional_env_var, required_env_var,
};
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder};

/// Default Anthropic API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// Anthropic's capability sheet.
///
/// Anthropic's native structured outputs (constrained decoding) are designed
/// to compose with strict tool use, so the schema constraint does not
/// suppress tool calls (see issue #1928) —
/// `composes_native_output_with_tools` is `true`.
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    name: "anthropic",
    supports_tools: true,
    supports_response_format: true,
    stream_include_usage: false,
    emits_complete_single_chunk_tool_calls: false,
    composes_native_output_with_tools: true,
    max_embedding_documents: None,
    verify_path: Some("/v1/models"),
};

/// `max_tokens` fallback for models outside the known table, matching the
/// value the deleted classic model applied.
pub const DEFAULT_MAX_TOKENS_FALLBACK: u64 = 2_048;

/// Plain-data Anthropic provider configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Config {
    /// API base URL (defaults to [`DEFAULT_BASE_URL`]).
    pub base_url: String,
    /// Credential location; sent as the `x-api-key` header.
    pub api_key: ApiKeyLocation,
    /// Model identifier requests are built for.
    pub model: String,
    /// Extra headers attached to every request.
    pub extra_headers: Vec<(String, String)>,
    /// Value of the required `anthropic-version` header.
    pub anthropic_version: String,
    /// Beta feature flags, joined into an `anthropic-beta` header when
    /// non-empty.
    pub anthropic_betas: Vec<String>,
    /// Fallback `max_tokens` applied when the request leaves it unset —
    /// Anthropic's API requires it. Mirrors the per-model defaults the
    /// model type resolves at construction.
    pub default_max_tokens: Option<u64>,
    /// Mark tool, system and message blocks with `cache_control` breakpoints,
    /// budgeted and ordered against Anthropic's four-breakpoint request limit.
    ///
    /// Fine-grained, per-block control. Prefer [`Self::automatic_caching`] for
    /// multi-turn conversations; the two compose, in which case the top-level
    /// automatic breakpoint owns the moving message cache point while rig still
    /// marks tools and the system prompt when the budget permits.
    pub prompt_caching: bool,
    /// Add a top-level `cache_control` field, enabling Anthropic's automatic
    /// caching: the API places the breakpoint on the last cacheable block and
    /// advances it as the conversation grows. No beta header required.
    pub automatic_caching: bool,
    /// TTL for the top-level `cache_control`. `None` omits the field, which the
    /// API reads as its five-minute default.
    pub automatic_caching_ttl: Option<CacheTtl>,
}

impl Config {
    /// Config for `model` reading `ANTHROPIC_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        let model = model.into();
        // Classic parity: the deleted `GenericCompletionModel::make` fell back
        // to 2048 for models outside the known table, so an unknown model
        // never failed the `max_tokens`-is-required check.
        let default_max_tokens =
            Some(default_max_tokens_for_model(&model).unwrap_or(DEFAULT_MAX_TOKENS_FALLBACK));
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: ApiKeyLocation::Env("ANTHROPIC_API_KEY".to_string()),
            model,
            extra_headers: Vec::new(),
            anthropic_version: ANTHROPIC_VERSION_LATEST.to_string(),
            anthropic_betas: Vec::new(),
            default_max_tokens,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
        }
    }

    /// Enable manual `cache_control` breakpoints on tools, system and message
    /// blocks. See [`Config::prompt_caching`].
    pub fn with_prompt_caching(mut self) -> Self {
        self.prompt_caching = true;
        self
    }

    /// Enable Anthropic's automatic prompt caching with the API's default TTL.
    /// See [`Config::automatic_caching`].
    pub fn with_automatic_caching(mut self) -> Self {
        self.automatic_caching = true;
        self
    }

    /// Enable Anthropic's automatic prompt caching with a one-hour TTL.
    pub fn with_automatic_caching_1h(mut self) -> Self {
        self.automatic_caching = true;
        self.automatic_caching_ttl = Some(CacheTtl::OneHour);
        self
    }

    /// Config for `model` built from the process environment.
    ///
    /// Reads `ANTHROPIC_API_KEY` (required) and `ANTHROPIC_BASE_URL` (optional
    /// override of [`DEFAULT_BASE_URL`]) — the same variables the deleted
    /// `anthropic::Client::from_env` read. The credential is validated eagerly
    /// but stored as [`ApiKeyLocation::Env`], so the secret is read at request
    /// time rather than held inside the config.
    ///
    /// # Errors
    /// [`ConfigError`] when a required variable is missing or invalid.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
        let mut cfg = Self::new(model);
        required_env_var("ANTHROPIC_API_KEY")?;
        if let Some(base_url) = optional_env_var("ANTHROPIC_BASE_URL")? {
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

/// Normalize an Anthropic Messages base URL.
///
/// Strips a trailing slash and any `/v1`, `/messages`, or `/v1/messages`
/// suffix, so a base URL copied from API documentation still composes into a
/// single well-formed request path. This is the classic client's
/// `anthropic::client::normalize_anthropic_base_url`, moved here so the
/// Anthropic-compatible providers (Z.ai, MiniMax, Moonshot, Xiaomi MiMo) keep
/// byte-identical URLs after the client layer's deletion.
pub fn normalize_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');

    if let Some(stripped) = trimmed.strip_suffix("/v1/messages") {
        stripped.to_string()
    } else if let Some(stripped) = trimmed.strip_suffix("/messages") {
        stripped.to_string()
    } else if let Some(stripped) = trimmed.strip_suffix("/v1") {
        stripped.to_string()
    } else {
        trimmed.to_string()
    }
}

fn resolve_model_and_max_tokens(
    cfg: &Config,
    request: &CompletionRequest,
) -> Result<(String, u64), CompletionError> {
    let model = request.model.clone().unwrap_or_else(|| cfg.model.clone());
    let max_tokens = request
        .max_tokens
        .or(cfg.default_max_tokens)
        .ok_or_else(|| {
            CompletionError::RequestError("`max_tokens` must be set for Anthropic".into())
        })?;
    Ok((model, max_tokens))
}

/// Build the serialized Messages API request body for `request`.
///
/// Pure: the exact bytes the wire sees. Delegates to the same typed
/// conversion as the trait path (`AnthropicCompletionRequest::try_from` for
/// blocking, the shared streaming body builder when `stream` is set), so the
/// two paths stay byte-identical.
pub fn build_request_body(
    cfg: &Config,
    request: &CompletionRequest,
    stream: bool,
) -> Result<Vec<u8>, CompletionError> {
    let (model, max_tokens) = resolve_model_and_max_tokens(cfg, request)?;
    let mut request = request.clone();
    if stream {
        let body = super::streaming::create_streaming_request_body(
            model,
            request,
            max_tokens,
            cfg.prompt_caching,
            cfg.automatic_caching,
            cfg.automatic_caching_ttl,
        )?;
        Ok(serde_json::to_vec(&body)?)
    } else {
        request.max_tokens = Some(max_tokens);
        let typed = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: &model,
            request,
            prompt_caching: cfg.prompt_caching,
            automatic_caching: cfg.automatic_caching,
            automatic_caching_ttl: cfg.automatic_caching_ttl,
        })?;
        Ok(serde_json::to_vec(&typed)?)
    }
}

/// Build the complete HTTP request (URL, headers, body) for `request`.
///
/// Pure except for credential resolution (`ApiKeyLocation::Env` reads the
/// environment). Sets `x-api-key`, `anthropic-version`, and (when
/// configured) `anthropic-beta` headers.
pub fn build_request(
    cfg: &Config,
    request: &CompletionRequest,
    stream: bool,
) -> Result<http::Request<Vec<u8>>, CompletionError> {
    let url = format!("{}/v1/messages", cfg.base_url.trim_end_matches('/'));
    let body = build_request_body(cfg, request, stream)?;

    let mut builder = http::Request::post(url)
        .header(CONTENT_TYPE, "application/json")
        .header("anthropic-version", cfg.anthropic_version.as_str());
    if !cfg.anthropic_betas.is_empty() {
        builder = builder.header("anthropic-beta", cfg.anthropic_betas.join(","));
    }
    if let Some(key) = cfg
        .api_key
        .resolve()
        .map_err(|e| CompletionError::RequestError(Box::new(e)))?
    {
        builder = builder.header("x-api-key", key);
    }
    for (name, value) in &cfg.extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    builder
        .body(body)
        .map_err(|e| CompletionError::RequestError(Box::new(e)))
}

/// Parse a Messages API response body into the normalized
/// [`completion::CompletionResponse`]. Pure.
pub fn parse_response(
    status: http::StatusCode,
    body: &str,
) -> Result<completion::CompletionResponse, CompletionError> {
    if !status.is_success() {
        return Err(CompletionError::from_http_response(
            status,
            body.to_string(),
        ));
    }
    match serde_json::from_str::<ApiResponse<super::completion::CompletionResponse>>(body)? {
        ApiResponse::Message(response) => {
            let mut converted: completion::CompletionResponse = response.try_into()?;
            converted.provider = DESCRIPTOR.name.to_string();
            Ok(converted)
        }
        ApiResponse::Error(error) => {
            tracing::warn!(message = %error.message, "provider returned an error response");
            Err(CompletionError::from_http_response(
                status,
                body.to_string(),
            ))
        }
    }
}

/// Open a streaming completion for `request`, driving the same SSE machinery
/// as the trait path.
pub async fn open_stream(
    cfg: &Config,
    rt: &HttpRuntime,
    request: CompletionRequest,
) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError> {
    let model = request.model.clone().unwrap_or_else(|| cfg.model.clone());
    let span =
        CompletionSpanBuilder::new(DESCRIPTOR.name, &model, CompletionOperation::ChatStreaming)
            .system_instructions(
                request.preamble.as_deref(),
                request.record_telemetry_content,
            )
            .build();
    let req = build_request(cfg, &request, true)?;
    Ok(super::streaming::stream_anthropic_sse(
        rt.sse_events(req, false),
        DESCRIPTOR.name,
        span,
    ))
}

/// Send `request` to Anthropic and return the normalized response.
pub async fn complete(
    cfg: &Config,
    rt: &HttpRuntime,
    request: CompletionRequest,
) -> Result<completion::CompletionResponse, CompletionError> {
    let req = build_request(cfg, &request, false)?;
    let (status, body) = rt.send(req).await?;
    parse_response(status, &body)
}

/// Build one `GET /v1/models` page request for [`list_models`].
///
/// Pure except for credential resolution (`ApiKeyLocation::Env` reads the
/// environment). Sets the same `x-api-key` / `anthropic-version` /
/// `anthropic-beta` headers as [`build_request`].
pub fn build_list_models_request(
    cfg: &Config,
    after_id: Option<&str>,
) -> Result<http::Request<Vec<u8>>, crate::model::ModelListingError> {
    use crate::model::ModelListingError;

    let path = super::model_listing::list_models_path(after_id);
    let url = format!("{}{}", cfg.base_url.trim_end_matches('/'), path);
    // Classic parity: the deleted client stamped `accept`/`content-type` on
    // every request, bodyless GETs included, and the recordings match on them.
    let mut builder = http::Request::get(url)
        .header(http::header::ACCEPT, "*/*")
        .header(http::header::CONTENT_TYPE, "application/json")
        .header("anthropic-version", cfg.anthropic_version.as_str());
    if !cfg.anthropic_betas.is_empty() {
        builder = builder.header("anthropic-beta", cfg.anthropic_betas.join(","));
    }
    if let Some(key) = cfg
        .api_key
        .resolve()
        .map_err(|e| ModelListingError::request_error(e.to_string()))?
    {
        builder = builder.header("x-api-key", key);
    }
    for (name, value) in &cfg.extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    builder
        .body(Vec::new())
        .map_err(|e| ModelListingError::request_error(e.to_string()))
}

/// List the models available to `cfg`'s credentials, following cursor
/// pagination through all pages.
///
/// Parses through the pure
/// page parser (`model_listing::parse_models_page`).
pub async fn list_models(
    cfg: &Config,
    rt: &HttpRuntime,
) -> Result<crate::model::ModelList, crate::model::ModelListingError> {
    let mut all_models = Vec::new();
    let mut after_id: Option<String> = None;

    loop {
        let path = super::model_listing::list_models_path(after_id.as_deref());
        let req = build_list_models_request(cfg, after_id.as_deref())?;
        let (status, body) = rt.send_bytes(req).await?;
        let (models, next_after_id) =
            super::model_listing::parse_models_page(&path, status, &body)?;
        all_models.extend(models);

        match next_after_id {
            Some(cursor) => after_id = Some(cursor),
            None => break,
        }
    }

    Ok(crate::model::ModelList::new(all_models))
}
/// Build the `GET /v1/models` credential-verification request.
///
/// Anthropic authenticates with the `x-api-key` header, not a Bearer token,
/// so this builds the request itself rather than using
/// [`crate::providers::verify::verify_bearer`]. The `anthropic-version`
/// header is sent for the same reason [`build_request`] sends it: Anthropic
/// rejects requests without it.
///
/// # Errors
/// [`VerifyError`](crate::providers::verify::VerifyError) when the
/// credential cannot be resolved.
pub fn build_verify_request(
    cfg: &Config,
) -> Result<http::Request<Vec<u8>>, crate::providers::verify::VerifyError> {
    use crate::providers::verify::{VerifyError, verify_url};

    let url = verify_url(&DESCRIPTOR, &cfg.base_url)?;
    let mut builder =
        http::Request::get(url).header("anthropic-version", cfg.anthropic_version.as_str());
    if let Some(key) = cfg
        .api_key
        .resolve()
        .map_err(|e| VerifyError::ProviderError(e.to_string()))?
    {
        builder = builder.header("x-api-key", key);
    }
    for (name, value) in &cfg.extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    builder
        .body(Vec::new())
        .map_err(|e| VerifyError::ProviderError(e.to_string()))
}

/// Verify that `cfg`'s credential is accepted by Anthropic.
///
/// The data-oriented replacement for the deleted `VerifyClient::verify`: the
/// endpoint is [`DESCRIPTOR`]'s `verify_path` (`/v1/models`, the value the
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
    crate::providers::verify::send_verify(rt, build_verify_request(cfg)?).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        original: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let original = std::env::var(key).ok();
            // SAFETY: Tests in this module hold ENV_LOCK while mutating process
            // environment and restore the original value before releasing it.
            unsafe { std::env::set_var(key, value) };

            Self { key, original }
        }

        fn remove(key: &'static str) -> Self {
            let original = std::env::var(key).ok();
            // SAFETY: Tests in this module hold ENV_LOCK while mutating process
            // environment and restore the original value before releasing it.
            unsafe { std::env::remove_var(key) };

            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            // SAFETY: Tests in this module hold ENV_LOCK while mutating process
            // environment and restore the original value before releasing it.
            unsafe {
                match &self.original {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    // Ported from the deleted `anthropic::client` tests
    // (`from_env_uses_anthropic_base_url`) — the env-precedence assertion now
    // targets `Config::from_env`. `Config` stores the base URL verbatim and
    // `build_request` appends `/v1/messages`, so the normalization the classic
    // builder applied eagerly is asserted through `normalize_base_url`.
    #[test]
    fn from_env_uses_anthropic_base_url() {
        let _guard = ENV_LOCK.lock().expect("env lock should not be poisoned");
        let _api_key = EnvVarGuard::set("ANTHROPIC_API_KEY", "dummy-key");
        let _base_url = EnvVarGuard::set(
            "ANTHROPIC_BASE_URL",
            "https://anthropic-compatible.example/v1/messages",
        );

        let cfg = Config::from_env("claude-sonnet-4-6")
            .expect("Config::from_env should build with ANTHROPIC_BASE_URL");

        assert_eq!(
            cfg.base_url, "https://anthropic-compatible.example/v1/messages",
            "from_env should apply ANTHROPIC_BASE_URL verbatim"
        );
        assert_eq!(
            normalize_base_url(&cfg.base_url),
            "https://anthropic-compatible.example",
            "the existing Anthropic base URL normalization must still collapse the suffix"
        );
    }

    // Ported from the deleted `anthropic::client` test
    // `from_env_uses_default_base_url_when_anthropic_base_url_is_unset`.
    #[test]
    fn from_env_uses_default_base_url_when_anthropic_base_url_is_unset() {
        let _guard = ENV_LOCK.lock().expect("env lock should not be poisoned");
        let _api_key = EnvVarGuard::set("ANTHROPIC_API_KEY", "dummy-key");
        let _base_url = EnvVarGuard::remove("ANTHROPIC_BASE_URL");

        let cfg = Config::from_env("claude-sonnet-4-6")
            .expect("Config::from_env should build without ANTHROPIC_BASE_URL");

        assert_eq!(cfg.base_url, "https://api.anthropic.com");
    }

    #[test]
    fn build_list_models_request_sets_url_headers_and_cursor() {
        let cfg = Config::new("claude-sonnet-4-6").with_api_key("secret");
        let req = build_list_models_request(&cfg, None).expect("build");
        assert_eq!(req.method(), http::Method::GET);
        assert_eq!(req.uri(), "https://api.anthropic.com/v1/models");
        assert_eq!(
            req.headers().get("x-api-key").and_then(|v| v.to_str().ok()),
            Some("secret")
        );
        assert!(req.headers().get("anthropic-version").is_some());

        let paged = build_list_models_request(&cfg, Some("model-cursor")).expect("build");
        assert_eq!(
            paged.uri(),
            "https://api.anthropic.com/v1/models?after_id=model-cursor"
        );
    }
    use crate::OneOrMany;
    use crate::message::Message;

    fn sample_request() -> CompletionRequest {
        CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one(Message::user("hello")),
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
    fn build_request_body_matches_typed_conversion() {
        let cfg = Config::new("claude-sonnet-4-6").with_api_key("k");
        let body = build_request_body(&cfg, &sample_request(), false).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["model"], "claude-sonnet-4-6");
        assert_eq!(value["max_tokens"], 64);
        assert_eq!(value["temperature"], 0.5);
        assert!(value.get("stream").is_none());

        let streaming = build_request_body(&cfg, &sample_request(), true).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&streaming).expect("json");
        assert_eq!(value["stream"], true);
    }

    #[test]
    fn build_request_body_applies_default_max_tokens() {
        let cfg = Config::new("claude-sonnet-4-6").with_api_key("k");
        let mut request = sample_request();
        request.max_tokens = None;
        let body = build_request_body(&cfg, &request, false).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["max_tokens"], 64_000);
    }

    #[test]
    fn caching_is_off_by_default() {
        let cfg = Config::new("claude-sonnet-4-6").with_api_key("k");
        assert!(!cfg.prompt_caching);
        assert!(!cfg.automatic_caching);
        let body = build_request_body(&cfg, &sample_request(), false).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert!(value.get("cache_control").is_none());
    }

    #[test]
    fn automatic_caching_reaches_the_request_body() {
        let cfg = Config::new("claude-sonnet-4-6")
            .with_api_key("k")
            .with_automatic_caching();
        let body = build_request_body(&cfg, &sample_request(), false).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["cache_control"]["type"], "ephemeral");
        assert!(value["cache_control"].get("ttl").is_none());
    }

    #[test]
    fn automatic_caching_1h_sets_the_ttl() {
        let cfg = Config::new("claude-sonnet-4-6")
            .with_api_key("k")
            .with_automatic_caching_1h();
        let body = build_request_body(&cfg, &sample_request(), false).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["cache_control"]["ttl"], "1h");
    }

    #[test]
    fn prompt_caching_marks_the_streaming_body() {
        let cfg = Config::new("claude-sonnet-4-6")
            .with_api_key("k")
            .with_prompt_caching();
        let body = build_request_body(&cfg, &sample_request(), true).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        // The single user message is the last cacheable block, so it carries
        // the breakpoint the budgeter placed.
        assert_eq!(
            value["messages"][0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
    }

    #[test]
    fn build_request_sets_url_and_headers() {
        let cfg = Config::new("claude-sonnet-4-6").with_api_key("secret");
        let req = build_request(&cfg, &sample_request(), false).expect("build");
        assert_eq!(req.uri(), "https://api.anthropic.com/v1/messages");
        assert_eq!(
            req.headers().get("x-api-key").and_then(|v| v.to_str().ok()),
            Some("secret")
        );
        assert_eq!(
            req.headers()
                .get("anthropic-version")
                .and_then(|v| v.to_str().ok()),
            Some(ANTHROPIC_VERSION_LATEST)
        );
        assert!(req.headers().get("anthropic-beta").is_none());
    }

    #[test]
    fn build_request_honors_model_override() {
        let cfg = Config::new("claude-sonnet-4-6").with_api_key("k");
        let mut request = sample_request();
        request.model = Some("claude-haiku-4-5".to_string());
        let body = build_request_body(&cfg, &request, false).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["model"], "claude-haiku-4-5");
    }

    #[test]
    fn parse_response_normalizes() {
        let body = serde_json::json!({
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 3, "output_tokens": 2}
        })
        .to_string();
        let response = parse_response(http::StatusCode::OK, &body).expect("parse");
        assert_eq!(response.provider, "anthropic");
        assert_eq!(response.model.as_deref(), Some("claude-sonnet-4-6"));
        assert_eq!(response.message_id.as_deref(), Some("msg_1"));
        assert_eq!(response.usage.input_tokens, 3);
        assert_eq!(response.usage.total_tokens, 5);
        assert_eq!(response.finish_reason, Some(completion::FinishReason::Stop));
    }

    #[test]
    fn parse_response_surfaces_provider_error() {
        let body = serde_json::json!({
            "type": "error",
            "message": "boom"
        })
        .to_string();
        let error = parse_response(http::StatusCode::OK, &body).expect_err("should error");
        assert!(error.to_string().contains("boom"));
    }
}
