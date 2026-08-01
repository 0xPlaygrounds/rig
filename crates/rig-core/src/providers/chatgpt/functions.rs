//! ChatGPT Codex backend as config + pure functions.
//!
//! The data-oriented face of the ChatGPT subscription provider: a serde
//! [`Config`], a [`DESCRIPTOR`] capability sheet, and free functions
//! decomposing a completion into its parts — [`build_request_body`] /
//! [`build_request`] and [`parse_response`] — plus async [`complete`] and
//! [`open_stream`] wrappers over [`HttpRuntime`].
//!
//! # Credentials
//!
//! ChatGPT authenticates via an OAuth device-code flow with a cached,
//! refreshable token file, which cannot be represented as plain data. So
//! [`Config`] carries only the *resolved* credential — a bearer access token
//! plus the optional `ChatGPT-Account-Id` — and resolution is a
//! construction-time step:
//!
//! - [`Config::from_env`] — synchronous, `CHATGPT_ACCESS_TOKEN` only;
//! - [`config_from_env`] — async, the full precedence the deleted
//!   `chatgpt::Client::from_env` had (access token, then cached OAuth with
//!   refresh and device flow) via
//!   [`auth::Authenticator`](super::auth::Authenticator);
//! - [`config_from_auth`] — async, for a caller-configured authenticator.
//!
//! Because the resolved token is stored in the config, long-lived processes
//! should rebuild the config when a token expires — the classic client
//! refreshed per request.
//!
//! # Other deviations
//!
//! - The Codex backend only speaks SSE, so the request body always carries
//!   `stream: true` — [`build_request_body`]'s `stream` flag does not change
//!   the bytes (both paths share
//!   `build_codex_responses_request`).
//! - [`build_request`] is not fully pure: it mints a fresh random
//!   `session_id` header per request, exactly as the trait path does.
//! - [`parse_response`] is the pure subset of response handling; the trait
//!   path and [`complete`] additionally apply an async streamed-text fallback
//!   for empty `response.completed` payloads (shared
//!   `parse_codex_sse_response`).

use http::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};

use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_runtime::HttpRuntime;
use crate::providers::descriptor::{
    ApiKeyLocation, ConfigError, ProviderDescriptor, optional_env_var,
};
use crate::providers::openai::responses_api;
use crate::telemetry::{CompletionOperation, completion_span};

/// Default ChatGPT Codex backend base URL.
pub const DEFAULT_BASE_URL: &str = super::CHATGPT_API_BASE_URL;

/// ChatGPT's capability sheet.
///
/// `supports_response_format` is `false`: the request build clears the
/// Responses `text` parameter (where structured output would go), so
/// `output_schema` never reaches the wire.
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    name: "chatgpt",
    supports_tools: true,
    supports_response_format: false,
    stream_include_usage: false,
    emits_complete_single_chunk_tool_calls: false,
    composes_native_output_with_tools: false,
    max_embedding_documents: None,
    verify_path: None,
};

/// Plain-data ChatGPT provider configuration (access-token auth only — see
/// the module docs for the OAuth deviation).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Config {
    /// Reusable HTTP connection data, including the bearer access token.
    #[serde(flatten)]
    pub connection: crate::providers::HttpConnectionConfig,
    /// Optional `ChatGPT-Account-Id` header value.
    pub account_id: Option<String>,
    /// Model identifier requests are built for.
    pub model: String,
    /// Default instructions merged into every request's `instructions`.
    pub default_instructions: Option<String>,
    /// Value of the required `originator` header.
    pub originator: String,
    /// Value of the `user-agent` header.
    pub user_agent: String,
    /// Default tool definitions appended to every request's `tools` (classic
    /// `ResponsesCompletionModel::with_tool` / `with_tools`).
    #[serde(default)]
    pub default_tools: Vec<responses_api::ResponsesToolDefinition>,
    /// Request strict function tool schemas (classic
    /// `ResponsesCompletionModel::with_strict_tools`).
    #[serde(default)]
    pub strict_tools: bool,
}

crate::providers::client::impl_http_connection_config!(Config);

impl Config {
    /// Config for `model` reading `CHATGPT_ACCESS_TOKEN` from the
    /// environment.
    ///
    /// Mirrors the client builder's defaults, including honoring the
    /// `CHATGPT_DEFAULT_INSTRUCTIONS` environment variable for the default
    /// instructions (so both faces produce identical request bytes).
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            connection: crate::providers::HttpConnectionConfig::new(
                DEFAULT_BASE_URL,
                ApiKeyLocation::Env("CHATGPT_ACCESS_TOKEN".to_string()),
            ),
            account_id: None,
            model: model.into(),
            default_instructions: Some(
                std::env::var("CHATGPT_DEFAULT_INSTRUCTIONS")
                    .ok()
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or_else(|| super::DEFAULT_INSTRUCTIONS.to_string()),
            ),
            originator: super::DEFAULT_ORIGINATOR.to_string(),
            user_agent: super::default_user_agent(),
            default_tools: Vec::new(),
            strict_tools: false,
        }
    }

    /// Config for `model` with an explicit access token.
    pub fn with_access_token(mut self, token: impl Into<String>) -> Self {
        self.api_key = ApiKeyLocation::Inline(token.into());
        self
    }

    /// Attach a `ChatGPT-Account-Id` header value.
    pub fn with_account_id(mut self, account_id: impl Into<String>) -> Self {
        self.account_id = Some(account_id.into());
        self
    }

    /// Override the API base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Append a default tool definition to every request.
    pub fn with_tool(mut self, tool: impl Into<responses_api::ResponsesToolDefinition>) -> Self {
        self.default_tools.push(tool.into());
        self
    }

    /// Append default tool definitions to every request.
    pub fn with_tools<I, Tool>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = Tool>,
        Tool: Into<responses_api::ResponsesToolDefinition>,
    {
        self.default_tools.extend(tools.into_iter().map(Into::into));
        self
    }

    /// Enable strict mode for function tool schemas.
    pub fn with_strict_tools(mut self) -> Self {
        self.strict_tools = true;
        self
    }

    /// Config for `model` from an explicit access token in the environment.
    ///
    /// The synchronous subset of [`config_from_env`]: it requires
    /// `CHATGPT_ACCESS_TOKEN`, reads the optional `CHATGPT_ACCOUNT_ID`, and
    /// applies the `CHATGPT_API_BASE` / `OPENAI_CHATGPT_API_BASE` override.
    /// Use [`config_from_env`] when the credential may come from a cached
    /// OAuth login, which requires an async refresh.
    ///
    /// # Errors
    /// [`ConfigError::InvalidConfiguration`] when `CHATGPT_ACCESS_TOKEN` is
    /// unset.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
        let mut cfg = Self::new(model);
        if optional_env_var("CHATGPT_ACCESS_TOKEN")?.is_none() {
            return Err(ConfigError::InvalidConfiguration(
                "`CHATGPT_ACCESS_TOKEN` must be set; use `config_from_env` for OAuth credentials",
            ));
        }
        cfg.account_id = optional_env_var("CHATGPT_ACCOUNT_ID")?;
        if let Some(base_url) = env_base_url()? {
            cfg.base_url = base_url;
        }
        Ok(cfg)
    }
}

/// The `CHATGPT_API_BASE` / `OPENAI_CHATGPT_API_BASE` override, in the classic
/// client's precedence order.
fn env_base_url() -> Result<Option<String>, ConfigError> {
    Ok(optional_env_var("CHATGPT_API_BASE")?.or(optional_env_var("OPENAI_CHATGPT_API_BASE")?))
}

/// Failure while resolving a ChatGPT configuration from the environment.
#[derive(Debug, thiserror::Error)]
pub enum ConfigFromEnvError {
    /// An environment variable was missing or unreadable.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// OAuth refresh or the device-code flow failed.
    #[error(transparent)]
    Auth(#[from] super::auth::AuthError),
}

/// Config for `model`, resolving the credential the way the deleted
/// `chatgpt::Client::from_env` did.
///
/// Precedence, unchanged from the classic client:
///
/// 1. `CHATGPT_ACCESS_TOKEN` (plus optional `CHATGPT_ACCOUNT_ID`);
/// 2. otherwise the cached OAuth login in the standard `auth.json`, refreshing
///    or running the device-code flow as needed.
///
/// The `CHATGPT_API_BASE` / `OPENAI_CHATGPT_API_BASE` override and the
/// `CHATGPT_DEFAULT_INSTRUCTIONS` default (applied by [`Config::new`]) behave
/// as before.
///
/// This is async because case 2 performs IO. Use [`Config::from_env`] for the
/// synchronous access-token path.
///
/// # Errors
/// [`ConfigFromEnvError`] when a variable is unreadable or credential
/// resolution fails.
pub async fn config_from_env(model: impl Into<String>) -> Result<Config, ConfigFromEnvError> {
    let source = match optional_env_var("CHATGPT_ACCESS_TOKEN")? {
        Some(access_token) => super::auth::AuthSource::AccessToken {
            access_token,
            account_id: optional_env_var("CHATGPT_ACCOUNT_ID")?,
        },
        None => super::auth::AuthSource::OAuth,
    };

    // The classic `ChatGPTBuilder::default()` auth-file location, so a login
    // cached by the old client is still found.
    let authenticator = super::auth::Authenticator::new(
        source,
        super::default_auth_file(),
        super::auth::DeviceCodePrompter::default(),
        true,
    );

    config_from_auth(model, &authenticator).await
}

/// Config for `model` using a caller-built `Authenticator`.
///
/// The escape hatch [`config_from_env`] does not cover: a custom `auth.json`
/// location, a device-code prompt handler, or `allow_device_flow = false` for
/// unattended services (the classic builder's `auth_file`/`token_dir`/
/// `on_device_code`/`allow_device_flow` knobs).
///
/// # Errors
/// [`ConfigFromEnvError`] when credential resolution fails.
pub async fn config_from_auth(
    model: impl Into<String>,
    authenticator: &super::auth::Authenticator,
) -> Result<Config, ConfigFromEnvError> {
    let mut cfg = Config::new(model);
    let auth = authenticator.auth_context().await?;

    cfg.api_key = ApiKeyLocation::Inline(auth.access_token);
    cfg.account_id = auth.account_id;
    if let Some(base_url) = env_base_url()? {
        cfg.base_url = base_url;
    }
    Ok(cfg)
}

/// Build the serialized Codex Responses request body for `request`.
///
/// Pure: the exact bytes the wire sees. The `stream` flag is accepted for
/// interface parity but does not change the bytes — the Codex backend only
/// speaks SSE, so the body always carries `stream: true` (see module docs).
pub fn build_request_body(
    cfg: &Config,
    request: &CompletionRequest,
    _stream: bool,
) -> Result<Vec<u8>, CompletionError> {
    let typed = super::build_codex_responses_request(
        cfg.model.clone(),
        &cfg.default_tools,
        cfg.strict_tools,
        cfg.default_instructions.as_deref(),
        request.clone(),
    )?;
    Ok(serde_json::to_vec(&typed)?)
}

/// Build the complete HTTP request (URL, headers, body) for `request`.
///
/// Not fully pure: resolves the credential (`ApiKeyLocation::Env` reads the
/// environment) and mints a fresh random `session_id` header, exactly as the
/// trait path does per request.
pub fn build_request(
    cfg: &Config,
    request: &CompletionRequest,
    stream: bool,
) -> Result<http::Request<Vec<u8>>, CompletionError> {
    let url = format!("{}/responses", cfg.base_url.trim_end_matches('/'));
    let body = build_request_body(cfg, request, stream)?;

    let mut builder = http::Request::post(url)
        .header(CONTENT_TYPE, "application/json")
        .header(ACCEPT, "text/event-stream")
        .header("originator", cfg.originator.as_str())
        .header("user-agent", cfg.user_agent.as_str())
        .header("session_id", crate::id::generate());
    if let Some(token) = cfg
        .api_key
        .resolve()
        .map_err(|e| CompletionError::RequestError(Box::new(e)))?
    {
        builder = builder.header(AUTHORIZATION, format!("Bearer {token}"));
    }
    if let Some(account_id) = &cfg.account_id {
        builder = builder.header("ChatGPT-Account-Id", account_id.as_str());
    }
    for (name, value) in &cfg.extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    builder
        .body(body)
        .map_err(|e| CompletionError::RequestError(Box::new(e)))
}

/// Parse a Codex SSE completion body into the normalized
/// [`completion::CompletionResponse`]. Pure.
///
/// This is the synchronous subset of the shared parse path: it does not
/// apply the async streamed-text fallback for empty `response.completed`
/// payloads — [`complete`] (like the trait path) does.
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
    let raw = responses_api::streaming::parse_sse_completion_body(body, "ChatGPT")?;
    raw.try_into()
}

/// Open a streaming completion for `request`, driving the same SSE machinery
/// as the trait path.
pub async fn open_stream(
    cfg: &Config,
    rt: &HttpRuntime,
    request: CompletionRequest,
) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError> {
    let model = request.model.clone().unwrap_or_else(|| cfg.model.clone());
    let span = completion_span(
        DESCRIPTOR.name,
        &model,
        CompletionOperation::ChatStreaming,
        &request,
    );
    let req = build_request(cfg, &request, true)?;
    // The ChatGPT backend omits the SSE `Content-Type` header.
    Ok(responses_api::streaming::stream_from_event_source(
        rt.sse_events(req, true),
        span,
    ))
}

/// Send `request` to the ChatGPT Codex backend and return the normalized
/// response.
///
/// Routes through the same shared parse path as the trait impl, including
/// the async streamed-text fallback for empty `response.completed` payloads.
pub async fn complete(
    cfg: &Config,
    rt: &HttpRuntime,
    request: CompletionRequest,
) -> Result<completion::CompletionResponse, CompletionError> {
    let req = build_request(cfg, &request, false)?;
    let (status, body) = rt.send(req).await?;
    super::parse_codex_sse_response(status, &body).await
}
/// Credential verification is not available for this provider.
///
/// The deleted client declared `const VERIFY_PATH: &'static str = ""`, so the
/// classic `verify()` issued a bare `GET` of the base URL — a request that
/// checked no credential. [`DESCRIPTOR`] therefore carries no `verify_path`
/// and this reports the fact rather than repeating the empty check.
///
/// # Errors
/// Always [`VerifyError::Unsupported`](crate::providers::verify::VerifyError::Unsupported).
pub async fn verify(
    cfg: &Config,
    rt: &HttpRuntime,
) -> Result<(), crate::providers::verify::VerifyError> {
    let _ = (cfg, rt);
    Err(crate::providers::verify::VerifyError::Unsupported {
        provider: DESCRIPTOR.name,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OneOrMany;
    use crate::message::Message;

    fn sample_config() -> Config {
        Config {
            connection: crate::providers::HttpConnectionConfig::new(
                DEFAULT_BASE_URL,
                ApiKeyLocation::Inline("token".to_string()),
            ),
            account_id: Some("acct_1".to_string()),
            model: "gpt-5.3-codex".to_string(),
            default_instructions: Some(super::super::DEFAULT_INSTRUCTIONS.to_string()),
            originator: "rig".to_string(),
            user_agent: "rig-test".to_string(),
            default_tools: Vec::new(),
            strict_tools: false,
        }
    }

    fn sample_request() -> CompletionRequest {
        CompletionRequest {
            model: None,
            chat_history: OneOrMany::many(vec![
                Message::system("Respond tersely.".to_string()),
                Message::user("hello"),
            ])
            .expect("non-empty"),
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
    fn build_request_body_applies_codex_shaping() {
        let cfg = sample_config();
        let body = build_request_body(&cfg, &sample_request(), false).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["model"], "gpt-5.3-codex");
        // The Codex backend only speaks SSE: stream is always set.
        assert_eq!(value["stream"], true);
        // Temperature and max tokens are dropped by the backend shaping.
        assert!(value.get("temperature").is_none());
        assert!(value.get("max_output_tokens").is_none());
        assert_eq!(value["store"], false);
        let instructions = value["instructions"].as_str().expect("instructions");
        assert!(instructions.starts_with(super::super::DEFAULT_INSTRUCTIONS));
        assert!(instructions.ends_with("Respond tersely."));
    }

    #[test]
    fn build_request_body_carries_default_tools() {
        let cfg = sample_config().with_tool(responses_api::ResponsesToolDefinition::function(
            "get_weather",
            "Look up the weather",
            serde_json::json!({"type": "object", "properties": {}}),
        ));
        let body = build_request_body(&cfg, &sample_request(), false).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        let tools = value["tools"].as_array().expect("tools");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "get_weather");
        // Strict mode is off by default, so `strict` is skipped on the wire.
        assert!(tools[0].get("strict").is_none());
    }

    #[test]
    fn build_request_body_applies_strict_tools() {
        let cfg = sample_config()
            .with_tools([responses_api::ResponsesToolDefinition::function(
                "get_weather",
                "Look up the weather",
                serde_json::json!({"type": "object", "properties": {}}),
            )])
            .with_strict_tools();
        let body = build_request_body(&cfg, &sample_request(), false).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        let tools = value["tools"].as_array().expect("tools");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["strict"], true);
    }

    #[test]
    fn config_tool_fields_round_trip_through_serde() {
        let cfg = sample_config()
            .with_tool(responses_api::ResponsesToolDefinition::function(
                "get_weather",
                "Look up the weather",
                serde_json::json!({"type": "object", "properties": {}}),
            ))
            .with_strict_tools();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: Config = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, cfg);
        // A config serialized before the fields existed still deserializes.
        let mut value: serde_json::Value = serde_json::from_str(&json).expect("json");
        if let Some(map) = value.as_object_mut() {
            map.remove("default_tools");
            map.remove("strict_tools");
        }
        let legacy: Config = serde_json::from_value(value).expect("deserialize legacy");
        assert!(legacy.default_tools.is_empty());
        assert!(!legacy.strict_tools);
    }

    #[test]
    fn build_request_sets_url_and_headers() {
        let cfg = sample_config();
        let req = build_request(&cfg, &sample_request(), false).expect("build");
        assert_eq!(req.uri(), "https://chatgpt.com/backend-api/codex/responses");
        let headers = req.headers();
        assert_eq!(
            headers
                .get(http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok()),
            Some("Bearer token")
        );
        assert_eq!(
            headers
                .get("ChatGPT-Account-Id")
                .and_then(|v| v.to_str().ok()),
            Some("acct_1")
        );
        assert_eq!(
            headers.get("originator").and_then(|v| v.to_str().ok()),
            Some("rig")
        );
        assert_eq!(
            headers
                .get(http::header::ACCEPT)
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream")
        );
        assert!(headers.get("session_id").is_some());
    }

    #[test]
    fn parse_response_normalizes_sse_body() {
        let body = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\"status\":\"completed\",\"error\":null,\"incomplete_details\":null,\"instructions\":null,\"max_output_tokens\":null,\"model\":\"gpt-5\",\"usage\":{\"input_tokens\":3,\"input_tokens_details\":{\"cached_tokens\":0},\"output_tokens\":2,\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":5},\"output\":[{\"type\":\"message\",\"id\":\"msg_1\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"annotations\":[],\"text\":\"hi\"}]}],\"tools\":[]}}\ndata: [DONE]";
        let response = parse_response(http::StatusCode::OK, body).expect("parse");
        assert_eq!(response.usage.input_tokens, 3);
        assert_eq!(response.usage.total_tokens, 5);
    }

    #[test]
    fn parse_response_preserves_error_status() {
        let error = parse_response(
            http::StatusCode::UNAUTHORIZED,
            r#"{"error":{"message":"expired"}}"#,
        )
        .expect_err("should error");
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::UNAUTHORIZED)
        );
    }
}
