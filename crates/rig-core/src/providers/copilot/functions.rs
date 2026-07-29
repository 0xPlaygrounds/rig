//! GitHub Copilot Chat Completions as config + pure functions.
//!
//! The data-oriented face of the Copilot provider: a serde [`Config`], a
//! [`DESCRIPTOR`] capability sheet, and free functions —
//! [`build_request`] (data → HTTP request) and [`parse_response`]
//! (bytes → normalized [`completion::CompletionResponse`], no IO) — plus the
//! async [`complete`] and [`open_stream`] wrappers over
//! [`HttpRuntime`](crate::http_runtime::HttpRuntime).
//!
//! This module covers the `/chat/completions` surface the classic
//! [`CompletionModel`](super::CompletionModel) uses for conversational
//! models. Codex-class models, which the classic client routes through
//! `/responses` (see [`route_for_model`](super::CompletionModel)), are
//! future work for this face.
//!
//! # Credentials
//!
//! [`Config::api_key`] carries an **already-resolved Copilot chat token**
//! (or a long-lived Copilot API key). The classic client's richer auth —
//! OAuth device flow, GitHub access-token exchange, cached token files, and
//! automatic refresh (`auth::Authenticator`) — is stateful and asynchronous
//! and cannot be represented as a plain
//! [`ApiKeyLocation`]; callers needing those flows should resolve a token
//! through the classic client first. Token-derived base-URL routing
//! (`proxy-ep=` inside the token) is honored, mirroring the classic client.
//!
//! Note: Copilot requests carry a generated `x-request-id` header, so
//! [`build_request`] is pure up to that identifier (and credential
//! resolution); the body bytes are fully deterministic.

use http::header::CONTENT_TYPE;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{
    ChatApiResponse, ChatCompletionResponse, CopilotIntent, apply_headers, base_url_from_token,
    default_headers, request_has_vision, request_initiator, send_copilot_chat_streaming_request,
};
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_runtime::HttpRuntime;
use crate::providers::descriptor::{ApiKeyLocation, ProviderDescriptor};
use crate::providers::openai;

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
    /// (see the module docs for why OAuth flows are out of scope).
    pub api_key: ApiKeyLocation,
    /// Model identifier requests are built for.
    pub model: String,
    /// Request strict function tool schemas (classic `with_strict_tools`).
    pub strict_tools: bool,
    /// Send tool results as content arrays (classic
    /// `with_tool_result_array_content`).
    pub tool_result_array_content: bool,
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
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let body = build_request_body(cfg, request, stream)?;

    let headers = default_headers(
        &key,
        request_initiator(request),
        request_has_vision(request),
        CopilotIntent::default(),
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
            Err(CompletionError::from_http_response(status, body.to_string()))
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
    use crate::http_runtime::Transport;

    let req = build_request(cfg, &request, true)?;
    match rt.transport() {
        Transport::Reqwest(client) => {
            send_copilot_chat_streaming_request(client.clone(), req).await
        }
        #[cfg(feature = "test-utils")]
        Transport::Recording(client) => {
            send_copilot_chat_streaming_request(client.clone(), req).await
        }
    }
}

/// Send `request` to Copilot and return the normalized response.
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
