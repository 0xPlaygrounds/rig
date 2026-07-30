//! GitHub Copilot Chat Completions as config + pure functions.
//!
//! The data-oriented face of the Copilot provider: a serde [`Config`], a
//! [`DESCRIPTOR`] capability sheet, and free functions —
//! [`build_request`] (data → HTTP request) and [`parse_response`]
//! (bytes → normalized [`completion::CompletionResponse`], no IO) — plus the
//! async [`complete`] and [`open_stream`] wrappers over
//! [`HttpRuntime`](crate::http_runtime::HttpRuntime).
//!
//! Requests route by model: conversational models use `/chat/completions`
//! while codex-class models use `/responses`, sharing the provider's request
//! shaping and SSE machinery for both.
//!
//! # Credentials
//!
//! [`Config::api_key`] carries an **already-resolved Copilot chat token** (or
//! a long-lived Copilot API key), because a plain [`ApiKeyLocation`] cannot
//! express a stateful, refreshing credential.
//!
//! Resolution is therefore a construction-time step, not a request-time one:
//!
//! - [`Config::from_env`] — synchronous, API-key variables only;
//! - [`config_from_env`] — async, the full precedence the deleted
//!   `copilot::Client::from_env` had: API key, then GitHub access-token
//!   exchange, then cached OAuth (device flow and refresh included) via
//!   [`auth::Authenticator`](super::auth::Authenticator).
//!
//! Token-derived base-URL routing (`proxy-ep=` inside the token) is honored by
//! [`build_request`], mirroring the classic client. Because the resolved token
//! is stored in the config, long-lived processes should rebuild the config
//! when a token expires — the classic client refreshed per request.
//!
//! Note: Copilot requests carry a generated `x-request-id` header, so
//! [`build_request`] is pure up to that identifier (and credential
//! resolution); the body bytes are fully deterministic.

use http::header::CONTENT_TYPE;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{
    ChatApiResponse, ChatCompletionResponse, CompletionRoute, CopilotIntent, apply_headers,
    base_url_from_token, build_copilot_responses_request, default_headers, request_has_vision,
    request_initiator, route_for_model, send_copilot_chat_streaming_request,
    stream_copilot_responses_from_event_source,
};
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_runtime::HttpRuntime;
use crate::providers::descriptor::{
    ApiKeyLocation, ConfigError, ProviderDescriptor, optional_env_var,
};
use crate::providers::openai;
use crate::providers::openai::responses_api;
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder};

/// Default GitHub Copilot API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.githubcopilot.com";

/// Copilot's capability sheet (Chat Completions surface).
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    name: "copilot",
    supports_tools: true,
    supports_response_format: true,
    // Streaming requests always send `stream_options.include_usage`.
    stream_include_usage: true,
    emits_complete_single_chunk_tool_calls: false,
    composes_native_output_with_tools: true,
    max_embedding_documents: Some(1024),
};

/// Plain-data Copilot provider configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Config {
    /// API base URL (defaults to [`DEFAULT_BASE_URL`]). When left at the
    /// default, a `proxy-ep=` segment inside the resolved token overrides it,
    /// exactly like the classic client.
    pub base_url: String,
    /// Credential location; must resolve to a Copilot chat token or API key
    /// (see the module docs for how OAuth logins are resolved into one).
    pub api_key: ApiKeyLocation,
    /// Model identifier requests are built for.
    pub model: String,
    /// Request strict function tool schemas (classic `with_strict_tools`).
    pub strict_tools: bool,
    /// Send tool results as content arrays (classic
    /// `with_tool_result_array_content`).
    pub tool_result_array_content: bool,
    /// Conversation intent sent in the `openai-intent` request header
    /// (classic `with_intent` / `with_panel_intent` / `with_edits_intent`).
    #[serde(default)]
    pub intent: CopilotIntent,
    /// Extra headers attached to every request.
    pub extra_headers: Vec<(String, String)>,
}

impl Config {
    /// Config for `model` reading `GITHUB_COPILOT_API_KEY` from the
    /// environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: ApiKeyLocation::Env("GITHUB_COPILOT_API_KEY".to_string()),
            model: model.into(),
            strict_tools: false,
            tool_result_array_content: false,
            intent: CopilotIntent::Panel,
            extra_headers: Vec::new(),
        }
    }

    /// Config for `model` with an explicit Copilot token.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = ApiKeyLocation::Inline(key.into());
        self
    }

    /// Override the API base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set the conversation intent sent in the `openai-intent` header
    /// (classic `CompletionModel::with_intent`).
    pub fn with_intent(mut self, intent: CopilotIntent) -> Self {
        self.intent = intent;
        self
    }

    /// Config for `model` from an explicit Copilot API key in the environment.
    ///
    /// The synchronous subset of [`config_from_env`]: it requires one of
    /// `GITHUB_COPILOT_API_KEY` / `COPILOT_API_KEY` to be set and applies the
    /// `GITHUB_COPILOT_API_BASE` / `COPILOT_BASE_URL` override. Use
    /// [`config_from_env`] instead when the credential may come from a GitHub
    /// access token or a cached OAuth login, since both require an async
    /// exchange.
    ///
    /// # Errors
    /// [`ConfigError::InvalidConfiguration`] when no API-key variable is set.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
        let mut cfg = Self::new(model);
        let key_var = env_first_present(API_KEY_VARS)?.ok_or(ConfigError::InvalidConfiguration(
            "one of `GITHUB_COPILOT_API_KEY` or `COPILOT_API_KEY` must be set; \
                 use `config_from_env` for GitHub-token or OAuth credentials",
        ))?;
        cfg.api_key = ApiKeyLocation::Env(key_var.to_string());
        if let Some(base_url) = env_first_value(BASE_URL_VARS)? {
            cfg.base_url = base_url;
        }
        Ok(cfg)
    }
}

/// API-key variables, in the classic client's precedence order.
pub(crate) const API_KEY_VARS: &[&str] = &["GITHUB_COPILOT_API_KEY", "COPILOT_API_KEY"];
/// GitHub access-token variables, in the classic client's precedence order.
pub(crate) const ACCESS_TOKEN_VARS: &[&str] = &["COPILOT_GITHUB_ACCESS_TOKEN", "GITHUB_TOKEN"];
/// Base-URL override variables, in the classic client's precedence order.
pub(crate) const BASE_URL_VARS: &[&str] = &["GITHUB_COPILOT_API_BASE", "COPILOT_BASE_URL"];

/// The first of `vars` `get` reports as set, as a variable name.
///
/// The precedence rule itself, factored out of the environment so it is
/// testable without mutating the process environment.
pub(crate) fn first_present_in<F>(
    vars: &[&'static str],
    get: F,
) -> Result<Option<&'static str>, ConfigError>
where
    F: Fn(&'static str) -> Result<Option<String>, ConfigError>,
{
    for var in vars {
        if get(var)?.is_some() {
            return Ok(Some(var));
        }
    }
    Ok(None)
}

/// The value of the first of `vars` `get` reports as set.
pub(crate) fn first_value_in<F>(
    vars: &[&'static str],
    get: F,
) -> Result<Option<String>, ConfigError>
where
    F: Fn(&'static str) -> Result<Option<String>, ConfigError>,
{
    for var in vars {
        if let Some(value) = get(var)? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

/// The first of `vars` that is set in the process environment.
fn env_first_present(vars: &[&'static str]) -> Result<Option<&'static str>, ConfigError> {
    first_present_in(vars, optional_env_var)
}

/// The value of the first of `vars` that is set in the process environment.
fn env_first_value(vars: &[&'static str]) -> Result<Option<String>, ConfigError> {
    first_value_in(vars, optional_env_var)
}

/// Failure while resolving a Copilot configuration from the environment.
#[derive(Debug, thiserror::Error)]
pub enum ConfigFromEnvError {
    /// An environment variable was missing or unreadable.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// Credential exchange, refresh, or the device-code flow failed.
    #[error(transparent)]
    Auth(#[from] super::auth::AuthError),
}

/// Config for `model`, resolving the credential the way the deleted
/// `copilot::Client::from_env` did.
///
/// Precedence, unchanged from the classic client:
///
/// 1. `GITHUB_COPILOT_API_KEY` / `COPILOT_API_KEY` — used directly;
/// 2. `COPILOT_GITHUB_ACCESS_TOKEN` / `GITHUB_TOKEN` — exchanged for a Copilot
///    chat token (cached token file reused when still valid);
/// 3. otherwise the cached OAuth login, refreshing or running the device-code
///    flow as needed.
///
/// The base URL follows the same rules the classic client applied at request
/// time: an explicit `GITHUB_COPILOT_API_BASE` / `COPILOT_BASE_URL` override
/// wins; otherwise the endpoint reported by the token exchange is used; failing
/// that, [`build_request`] still derives it from the token's `proxy-ep=`
/// segment.
///
/// This is async because cases 2 and 3 perform IO. Use [`Config::from_env`]
/// for the synchronous API-key-only path.
///
/// # Errors
/// [`ConfigFromEnvError`] when a variable is unreadable or credential
/// resolution fails.
pub async fn config_from_env(model: impl Into<String>) -> Result<Config, ConfigFromEnvError> {
    let cfg = Config::new(model);
    let base_url_override = env_first_value(BASE_URL_VARS)?;

    let source = if let Some(key) = env_first_value(API_KEY_VARS)? {
        super::auth::AuthSource::ApiKey(key)
    } else if let Some(token) = env_first_value(ACCESS_TOKEN_VARS)? {
        super::auth::AuthSource::GitHubAccessToken(token)
    } else {
        super::auth::AuthSource::OAuth
    };

    // The classic `CopilotBuilder::default()` token-file layout, so a login
    // cached by the old client is still found.
    let token_dir = super::default_token_dir();
    let authenticator = super::auth::Authenticator::new(
        source,
        token_dir.as_ref().map(|dir| dir.join("access-token")),
        token_dir.map(|dir| dir.join("api-key.json")),
        super::auth::DeviceCodeHandler::default(),
        true,
    );

    apply_auth_context(cfg, &authenticator, base_url_override).await
}

/// Config for `model` using a caller-built `Authenticator`.
///
/// The escape hatch [`config_from_env`] does not cover: custom token-file
/// locations, a device-code prompt handler, or `allow_device_flow = false` for
/// unattended services (the classic builder's `token_dir`/`auth_file`/
/// `on_device_code`/`allow_device_flow` knobs).
///
/// # Errors
/// [`ConfigFromEnvError`] when credential resolution fails.
pub async fn config_from_auth(
    model: impl Into<String>,
    authenticator: &super::auth::Authenticator,
) -> Result<Config, ConfigFromEnvError> {
    let cfg = Config::new(model);
    let base_url_override = env_first_value(BASE_URL_VARS)?;
    apply_auth_context(cfg, authenticator, base_url_override).await
}

/// Resolve `authenticator` into `cfg`'s credential and base URL.
async fn apply_auth_context(
    mut cfg: Config,
    authenticator: &super::auth::Authenticator,
    base_url_override: Option<String>,
) -> Result<Config, ConfigFromEnvError> {
    let auth = authenticator.auth_context().await?;

    cfg.api_key = ApiKeyLocation::Inline(auth.api_key);
    match (base_url_override, auth.api_base) {
        (Some(base_url), _) => cfg.base_url = base_url,
        (None, Some(api_base)) => cfg.base_url = api_base,
        // Left at the default so `build_request` derives it from the token's
        // `proxy-ep=` segment, exactly as the classic client did.
        (None, None) => {}
    }
    Ok(cfg)
}

/// Build the serialized chat-completions request body for `request`.
///
/// Pure: the exact bytes the wire sees. `stream` adds `stream: true` and
/// `stream_options.include_usage` (per [`DESCRIPTOR`]), matching the classic
/// streaming path.
pub fn build_request_body(
    cfg: &Config,
    request: &CompletionRequest,
    stream: bool,
) -> Result<Vec<u8>, CompletionError> {
    if route_for_model(&cfg.model) == CompletionRoute::Responses {
        let mut typed = build_copilot_responses_request(cfg.model.clone(), request.clone())?;
        if stream {
            typed.stream = Some(true);
        }
        return Ok(serde_json::to_vec(&typed)?);
    }
    let typed =
        openai::completion::CompletionRequest::try_from(openai::completion::OpenAIRequestParams {
            model: cfg.model.clone(),
            request: request.clone(),
            strict_tools: cfg.strict_tools,
            tool_result_array_content: cfg.tool_result_array_content,
            supports_response_format: true,
            supports_tools: true,
        })?;
    if !stream {
        return Ok(serde_json::to_vec(&typed)?);
    }
    let mut value = serde_json::to_value(&typed)?;
    let object = value.as_object_mut().ok_or_else(|| {
        CompletionError::ResponseError("copilot request body must be a JSON object".into())
    })?;
    object.insert("stream".to_owned(), json!(true));
    object.insert(
        "stream_options".to_owned(),
        json!({ "include_usage": true }),
    );
    Ok(serde_json::to_vec(&value)?)
}

/// Build the complete HTTP request (URL, headers, body) for `request`.
///
/// Resolves the credential, applies token-derived base-URL routing, and
/// attaches Copilot's editor/session headers (including a generated
/// `x-request-id` — see module docs).
pub fn build_request(
    cfg: &Config,
    request: &CompletionRequest,
    stream: bool,
) -> Result<http::Request<Vec<u8>>, CompletionError> {
    let key = cfg
        .api_key
        .resolve()
        .map_err(|e| CompletionError::RequestError(Box::new(e)))?
        .ok_or_else(|| {
            CompletionError::RequestError("Copilot requires an API key or chat token".into())
        })?;

    let base_url = if cfg.base_url == DEFAULT_BASE_URL {
        base_url_from_token(&key).unwrap_or_else(|| cfg.base_url.clone())
    } else {
        cfg.base_url.clone()
    };
    let path = match route_for_model(&cfg.model) {
        CompletionRoute::Responses => "/responses",
        CompletionRoute::ChatCompletions => "/chat/completions",
    };
    let url = format!("{}{}", base_url.trim_end_matches('/'), path);
    let body = build_request_body(cfg, request, stream)?;

    let headers = default_headers(
        &key,
        request_initiator(request),
        request_has_vision(request),
        cfg.intent,
    );
    let mut builder = apply_headers(
        http::Request::post(url).header(CONTENT_TYPE, "application/json"),
        &headers,
    );
    for (name, value) in &cfg.extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    builder
        .body(body)
        .map_err(|e| CompletionError::RequestError(Box::new(e)))
}

/// Parse a `/responses`-route response body into the normalized
/// [`completion::CompletionResponse`]. Pure.
pub fn parse_responses_route_response(
    status: http::StatusCode,
    body: &str,
) -> Result<completion::CompletionResponse, CompletionError> {
    if !status.is_success() {
        return Err(CompletionError::from_http_response(
            status,
            body.to_string(),
        ));
    }
    let response = serde_json::from_str::<responses_api::CompletionResponse>(body)?;
    let mut core = completion::CompletionResponse::try_from(response)?;
    core.provider = "copilot".to_string();
    Ok(core)
}

/// Parse a chat-completions response body into the normalized
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
    match serde_json::from_str::<ChatApiResponse<ChatCompletionResponse>>(body)? {
        ChatApiResponse::Ok(response) => response.try_into(),
        ChatApiResponse::Err(err) => {
            tracing::warn!(
                message = %err.error_message(),
                "provider returned an error response"
            );
            Err(CompletionError::from_http_response(
                status,
                body.to_string(),
            ))
        }
    }
}

/// Open a streaming completion for `request`, reusing the provider's
/// OpenAI-compatible SSE machinery.
pub async fn open_stream(
    cfg: &Config,
    rt: &HttpRuntime,
    request: CompletionRequest,
) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError> {
    let req = build_request(cfg, &request, true)?;
    if route_for_model(&cfg.model) == CompletionRoute::Responses {
        let span = CompletionSpanBuilder::new(
            DESCRIPTOR.name,
            &cfg.model,
            CompletionOperation::ChatStreaming,
        )
        .system_instructions(
            request.preamble.as_deref(),
            request.record_telemetry_content,
        )
        .build();
        return Ok(stream_copilot_responses_from_event_source(
            rt.sse_events(req, false),
            span,
        ));
    }
    Ok(send_copilot_chat_streaming_request(
        rt.sse_events(req, false),
    ))
}

/// Send `request` to Copilot and return the normalized response.
pub async fn complete(
    cfg: &Config,
    rt: &HttpRuntime,
    request: CompletionRequest,
) -> Result<completion::CompletionResponse, CompletionError> {
    let req = build_request(cfg, &request, false)?;
    let (status, body) = rt.send(req).await?;
    match route_for_model(&cfg.model) {
        CompletionRoute::Responses => parse_responses_route_response(status, &body),
        CompletionRoute::ChatCompletions => parse_response(status, &body),
    }
}

// ================================================================
// Embeddings
// ================================================================

/// Plain-data Copilot embeddings configuration.
///
/// A sibling of [`Config`]: embeddings target their own model identifier
/// and an optional requested dimension count. Credentials follow the same
/// contract as [`Config::api_key`] — an already-resolved Copilot chat token
/// or API key (see the module docs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingConfig {
    /// API base URL (defaults to [`DEFAULT_BASE_URL`]). When left at the
    /// default, a `proxy-ep=` segment inside the resolved token overrides
    /// it, exactly like the classic client.
    pub base_url: String,
    /// Credential location; must resolve to a Copilot chat token or API key.
    pub api_key: ApiKeyLocation,
    /// Embedding model identifier requests are built for.
    pub model: String,
    /// Requested embedding dimensions, sent verbatim as `dimensions` when
    /// set (models that reject the field, like `text-embedding-ada-002`,
    /// should leave it unset).
    pub dimensions: Option<usize>,
    /// Extra headers attached to every request.
    pub extra_headers: Vec<(String, String)>,
}

impl EmbeddingConfig {
    /// Config for `model` reading `GITHUB_COPILOT_API_KEY` from the
    /// environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: ApiKeyLocation::Env("GITHUB_COPILOT_API_KEY".to_string()),
            model: model.into(),
            dimensions: None,
            extra_headers: Vec::new(),
        }
    }

    /// Config for `model` with an explicit Copilot token.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = ApiKeyLocation::Inline(key.into());
        self
    }

    /// Override the API base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Request `dimensions`-sized embeddings.
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }
}

/// Build the complete HTTP `/embeddings` request for one chunk of `texts`.
///
/// Pure up to credential resolution and the generated `x-request-id`
/// inside Copilot's default headers (matching [`build_request`]).
pub fn build_embedding_request(
    cfg: &EmbeddingConfig,
    texts: &[String],
) -> Result<http::Request<Vec<u8>>, crate::embeddings::EmbeddingError> {
    use crate::embeddings::EmbeddingError;

    let key = cfg
        .api_key
        .resolve()
        .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?
        .ok_or_else(|| {
            EmbeddingError::ProviderError("Copilot requires an API key or chat token".to_string())
        })?;

    let base_url = if cfg.base_url == DEFAULT_BASE_URL {
        base_url_from_token(&key).unwrap_or_else(|| cfg.base_url.clone())
    } else {
        cfg.base_url.clone()
    };
    let url = format!("{}/embeddings", base_url.trim_end_matches('/'));
    let body = super::build_embedding_body(&cfg.model, texts, cfg.dimensions, None, None)?;

    let headers = default_headers(&key, "user", false, CopilotIntent::Panel);
    let mut builder = apply_headers(
        http::Request::post(url).header(CONTENT_TYPE, "application/json"),
        &headers,
    );
    for (name, value) in &cfg.extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    builder
        .body(body)
        .map_err(|e| EmbeddingError::ProviderError(e.to_string()))
}

/// Parse an `/embeddings` response into the normalized
/// [`crate::embeddings::EmbeddingResponse`]. Pure.
pub fn parse_embedding_response(
    status: http::StatusCode,
    body: &str,
    documents: Vec<String>,
) -> Result<crate::embeddings::EmbeddingResponse, crate::embeddings::EmbeddingError> {
    super::parse_embedding_response(status, body, documents)
}

/// Embed `texts`, chunking to honor [`DESCRIPTOR`]'s
/// `max_embedding_documents`; embeddings are returned in input order.
pub async fn embed(
    cfg: &EmbeddingConfig,
    rt: &HttpRuntime,
    texts: Vec<String>,
) -> Result<crate::embeddings::EmbeddingResponse, crate::embeddings::EmbeddingError> {
    crate::embeddings::batching::embed_chunked(
        rt,
        texts,
        DESCRIPTOR.max_embedding_documents,
        |chunk| build_embedding_request(cfg, chunk),
        parse_embedding_response,
    )
    .await
}

/// Embed caller-defined batches, returning one order-aligned
/// [`OneOrMany`](crate::OneOrMany) group per input batch plus summed usage.
pub async fn embed_batches(
    cfg: &EmbeddingConfig,
    rt: &HttpRuntime,
    texts: Vec<Vec<String>>,
) -> Result<
    (
        Vec<crate::OneOrMany<crate::embeddings::Embedding>>,
        crate::completion::Usage,
    ),
    crate::embeddings::EmbeddingError,
> {
    let (counts, flat) = crate::embeddings::batching::split_batches(texts);
    let response = embed(cfg, rt, flat).await?;
    let groups = crate::embeddings::batching::group_batches(&counts, response.embeddings)?;
    Ok((groups, response.usage))
}

/// Build the `GET /models` request for [`list_models`].
///
/// Resolves the credential, applies token-derived base-URL routing, and
/// attaches the same Copilot editor/session headers as [`build_request`]
/// (including a generated `x-request-id`, so the request is pure up to that
/// identifier and credential resolution).
pub fn build_list_models_request(
    cfg: &Config,
) -> Result<http::Request<Vec<u8>>, crate::model::ModelListingError> {
    use crate::model::ModelListingError;

    let key = cfg
        .api_key
        .resolve()
        .map_err(|e| ModelListingError::request_error(e.to_string()))?
        .ok_or_else(|| {
            ModelListingError::request_error("Copilot requires an API key or chat token")
        })?;

    let base_url = if cfg.base_url == DEFAULT_BASE_URL {
        base_url_from_token(&key).unwrap_or_else(|| cfg.base_url.clone())
    } else {
        cfg.base_url.clone()
    };
    let url = format!("{}/models", base_url.trim_end_matches('/'));

    let headers = default_headers(&key, "user", false, CopilotIntent::Panel);
    let mut builder = apply_headers(http::Request::get(url), &headers);
    for (name, value) in &cfg.extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    builder
        .body(Vec::new())
        .map_err(|e| ModelListingError::request_error(e.to_string()))
}

/// List the models available to `cfg`'s credentials.
///
/// Parses through the pure `super::parse_list_models_response`.
pub async fn list_models(
    cfg: &Config,
    rt: &HttpRuntime,
) -> Result<crate::model::ModelList, crate::model::ModelListingError> {
    let req = build_list_models_request(cfg)?;
    let (status, body) = rt.send_bytes(req).await?;
    super::parse_list_models_response(status, &body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_list_models_request_sets_url_auth_and_copilot_headers() {
        let cfg = Config::new("gpt-4o").with_api_key("secret");
        let req = build_list_models_request(&cfg).expect("build");
        assert_eq!(req.method(), http::Method::GET);
        assert_eq!(req.uri(), "https://api.githubcopilot.com/models");
        assert_eq!(
            req.headers()
                .get(http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok()),
            Some("Bearer secret")
        );
        assert!(req.headers().get("copilot-integration-id").is_some());
        assert!(req.headers().get("x-request-id").is_some());
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
    fn build_request_sets_url_auth_and_copilot_headers() {
        let cfg = Config::new("gpt-4o").with_api_key("secret");
        let req = build_request(&cfg, &sample_request(), false).expect("build");
        assert_eq!(req.uri(), "https://api.githubcopilot.com/chat/completions");
        assert_eq!(
            req.headers()
                .get(http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok()),
            Some("Bearer secret")
        );
        assert_eq!(
            req.headers()
                .get("copilot-integration-id")
                .and_then(|v| v.to_str().ok()),
            Some("vscode-chat")
        );
        assert_eq!(
            req.headers()
                .get("editor-version")
                .and_then(|v| v.to_str().ok()),
            Some(super::super::EDITOR_VERSION)
        );
        // A fresh user turn without assistant/tool history is a "user" call.
        assert_eq!(
            req.headers()
                .get("X-Initiator")
                .and_then(|v| v.to_str().ok()),
            Some("user")
        );
        assert!(req.headers().get("x-request-id").is_some());
    }

    #[test]
    fn build_request_defaults_to_the_panel_intent() {
        let cfg = Config::new("gpt-4o").with_api_key("secret");
        let req = build_request(&cfg, &sample_request(), false).expect("build");
        assert_eq!(
            req.headers()
                .get("openai-intent")
                .and_then(|v| v.to_str().ok()),
            Some("conversation-panel")
        );
    }

    /// Restores the deleted classic
    /// `completion_model_edits_intent_sets_request_header` assertion on the
    /// functions path.
    #[test]
    fn edits_intent_sets_request_header() {
        let cfg = Config::new("gpt-4o")
            .with_api_key("secret")
            .with_intent(CopilotIntent::Edits);
        let req = build_request(&cfg, &sample_request(), false).expect("build");
        assert_eq!(
            req.headers()
                .get("openai-intent")
                .and_then(|v| v.to_str().ok()),
            Some("conversation-edits")
        );
    }

    #[test]
    fn config_intent_round_trips_through_serde() {
        let cfg = Config::new("gpt-4o").with_intent(CopilotIntent::Edits);
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: Config = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, cfg);
        // A config serialized before `intent` existed still deserializes.
        let mut value: serde_json::Value = serde_json::from_str(&json).expect("json");
        if let Some(map) = value.as_object_mut() {
            map.remove("intent");
        }
        let legacy: Config = serde_json::from_value(value).expect("deserialize legacy");
        assert_eq!(legacy.intent, CopilotIntent::Panel);
    }

    #[test]
    fn build_request_derives_base_url_from_token_proxy_ep() {
        let cfg = Config::new("gpt-4o")
            .with_api_key("tid=abc;proxy-ep=proxy.enterprise.githubcopilot.com;exp=1");
        let req = build_request(&cfg, &sample_request(), false).expect("build");
        assert_eq!(
            req.uri(),
            "https://api.enterprise.githubcopilot.com/chat/completions"
        );
    }

    #[test]
    fn build_request_body_injects_model_and_stream_options() {
        let cfg = Config::new("gpt-4o").with_api_key("k");
        let body = build_request_body(&cfg, &sample_request(), false).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["model"], "gpt-4o");
        assert_eq!(value["temperature"], 0.5);
        assert!(value.get("stream").is_none());

        let streaming = build_request_body(&cfg, &sample_request(), true).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&streaming).expect("json");
        assert_eq!(value["stream"], true);
        assert_eq!(value["stream_options"]["include_usage"], true);
    }

    #[test]
    fn parse_response_normalizes() {
        let body = serde_json::json!({
            "id": "chatcmpl-1",
            "model": "gpt-4o-2024",
            "system_fingerprint": null,
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
        assert_eq!(response.provider, "copilot");
        assert_eq!(response.model.as_deref(), Some("gpt-4o-2024"));
        assert_eq!(response.message_id.as_deref(), Some("chatcmpl-1"));
        assert_eq!(response.finish_reason, Some(completion::FinishReason::Stop));
        assert_eq!(response.usage.input_tokens, 3);
        assert_eq!(response.usage.total_tokens, 5);
    }

    #[test]
    fn parse_response_surfaces_http_errors() {
        let err = parse_response(http::StatusCode::UNAUTHORIZED, "bad token")
            .expect_err("non-success status must error");
        assert!(matches!(err, CompletionError::HttpError(_)));
    }
}
