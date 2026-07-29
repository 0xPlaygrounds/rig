//! ChatGPT Codex backend as config + pure functions.
//!
//! The data-oriented face of the ChatGPT subscription provider: a serde
//! [`Config`], a [`DESCRIPTOR`] capability sheet, and free functions
//! decomposing a completion into its parts — [`build_request_body`] /
//! [`build_request`] and [`parse_response`] — plus async [`complete`] and
//! [`open_stream`] wrappers over [`HttpRuntime`].
//!
//! # Auth deviation
//!
//! The full ChatGPT client authenticates via an OAuth device-code flow with
//! a cached, refreshable token file (`auth::Authenticator`). That flow is
//! interactive and stateful and cannot be represented as plain data, so
//! [`Config`] carries only the resolved credential: a bearer access token
//! (mirroring the client's `CHATGPT_ACCESS_TOKEN` path) plus the optional
//! `ChatGPT-Account-Id`. Callers using OAuth should resolve a token via the
//! client's authenticator and store it here as
//! [`ApiKeyLocation::Inline`]/`Env`.
//!
//! # Other deviations
//!
//! - The Codex backend only speaks SSE, so the request body always carries
//!   `stream: true` — [`build_request_body`]'s `stream` flag does not change
//!   the bytes (both paths share
//!   [`build_codex_responses_request`](super::build_codex_responses_request)).
//! - [`build_request`] is not fully pure: it mints a fresh random
//!   `session_id` header per request, exactly as the trait path does.
//! - [`parse_response`] is the pure subset of response handling; the trait
//!   path and [`complete`] additionally apply an async streamed-text fallback
//!   for empty `response.completed` payloads (shared
//!   [`parse_codex_sse_response`](super::parse_codex_sse_response)).

use http::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};

use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_runtime::HttpRuntime;
use crate::providers::descriptor::{ApiKeyLocation, ProviderDescriptor};
use crate::providers::openai::responses_api;
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder};

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
};

/// Plain-data ChatGPT provider configuration (access-token auth only — see
/// the module docs for the OAuth deviation).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Config {
    /// API base URL (defaults to [`DEFAULT_BASE_URL`]).
    pub base_url: String,
    /// Bearer access token location (the client's `CHATGPT_ACCESS_TOKEN`
    /// path). OAuth token files are not representable here.
    pub api_key: ApiKeyLocation,
    /// Optional `ChatGPT-Account-Id` header value.
    pub account_id: Option<String>,
    /// Model identifier requests are built for.
    pub model: String,
    /// Extra headers attached to every request.
    pub extra_headers: Vec<(String, String)>,
    /// Default instructions merged into every request's `instructions`.
    pub default_instructions: Option<String>,
    /// Value of the required `originator` header.
    pub originator: String,
    /// Value of the `user-agent` header.
    pub user_agent: String,
}

impl Config {
    /// Config for `model` reading `CHATGPT_ACCESS_TOKEN` from the
    /// environment.
    ///
    /// Mirrors the client builder's defaults, including honoring the
    /// `CHATGPT_DEFAULT_INSTRUCTIONS` environment variable for the default
    /// instructions (so both faces produce identical request bytes).
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: ApiKeyLocation::Env("CHATGPT_ACCESS_TOKEN".to_string()),
            account_id: None,
            model: model.into(),
            extra_headers: Vec::new(),
            default_instructions: Some(
                std::env::var("CHATGPT_DEFAULT_INSTRUCTIONS")
                    .ok()
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or_else(|| super::DEFAULT_INSTRUCTIONS.to_string()),
            ),
            originator: super::DEFAULT_ORIGINATOR.to_string(),
            user_agent: super::default_user_agent(),
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
        &[],
        false,
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
    use crate::http_client::sse::GenericEventSource;
    use crate::http_runtime::Transport;

    let model = request.model.clone().unwrap_or_else(|| cfg.model.clone());
    let span =
        CompletionSpanBuilder::new(DESCRIPTOR.name, &model, CompletionOperation::ChatStreaming)
            .system_instructions(
                request.preamble.as_deref(),
                request.record_telemetry_content,
            )
            .build();
    let req = build_request(cfg, &request, true)?;
    match rt.transport() {
        Transport::Reqwest(client) => {
            let event_source =
                GenericEventSource::new(client.clone(), req).allow_missing_content_type();
            Ok(responses_api::streaming::stream_from_event_source(
                event_source,
                span,
            ))
        }
        #[cfg(any(test, feature = "test-utils"))]
        Transport::Recording(client) => {
            let event_source =
                GenericEventSource::new(client.clone(), req).allow_missing_content_type();
            Ok(responses_api::streaming::stream_from_event_source(
                event_source,
                span,
            ))
        }
        #[cfg(any(test, feature = "test-utils"))]
        Transport::Sequenced(client) => {
            let event_source =
                GenericEventSource::new(client.clone(), req).allow_missing_content_type();
            Ok(responses_api::streaming::stream_from_event_source(
                event_source,
                span,
            ))
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OneOrMany;
    use crate::message::Message;

    fn sample_config() -> Config {
        Config {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: ApiKeyLocation::Inline("token".to_string()),
            account_id: Some("acct_1".to_string()),
            model: "gpt-5.3-codex".to_string(),
            extra_headers: Vec::new(),
            default_instructions: Some(super::super::DEFAULT_INSTRUCTIONS.to_string()),
            originator: "rig".to_string(),
            user_agent: "rig-test".to_string(),
        }
    }

    fn sample_request() -> CompletionRequest {
        CompletionRequest {
            model: None,
            preamble: Some("Respond tersely.".to_string()),
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
