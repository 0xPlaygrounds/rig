//! Anthropic Messages API as config + pure functions.
//!
//! The data-oriented face of the Anthropic provider: a serde [`Config`], a
//! [`DESCRIPTOR`] capability sheet, and free functions decomposing a
//! completion into pure parts — [`build_request_body`] / [`build_request`]
//! (data → HTTP request, no IO) and [`parse_response`] (bytes → normalized
//! [`completion::CompletionResponse`], no IO) — plus async [`complete`] and
//! [`open_stream`] wrappers over [`HttpRuntime`].
//!
//! During the transition the pure functions delegate to the same typed
//! conversions the [`GenericCompletionModel`](super::completion::GenericCompletionModel)
//! path uses ([`AnthropicCompletionRequest`]'s `TryFrom` for blocking bodies,
//! `create_streaming_request_body` for streaming bodies), so both paths
//! produce byte-identical request bodies. [`open_stream`] drives the exact
//! SSE machinery the trait path uses (`stream_anthropic_sse`).

use http::header::CONTENT_TYPE;
use serde::{Deserialize, Serialize};

use super::completion::{
    ANTHROPIC_VERSION_LATEST, AnthropicCompletionRequest, AnthropicRequestParams, ApiResponse,
    default_max_tokens_for_model,
};
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_runtime::HttpRuntime;
use crate::providers::descriptor::{ApiKeyLocation, ProviderDescriptor};
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder};

/// Default Anthropic API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// Anthropic's capability sheet.
///
/// Anthropic's native structured outputs (constrained decoding) are designed
/// to compose with strict tool use, so the schema constraint does not
/// suppress tool calls (see issue #1928) —
/// `composes_native_output_with_tools` is `true`, mirroring
/// `CompletionModel::composes_native_output_with_tools`.
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    name: "anthropic",
    supports_tools: true,
    supports_response_format: true,
    stream_include_usage: false,
    emits_complete_single_chunk_tool_calls: false,
    composes_native_output_with_tools: true,
    max_embedding_documents: None,
};

/// Plain-data Anthropic provider configuration.
///
/// Prompt-caching knobs (`with_prompt_caching`/`with_automatic_caching` on the
/// model type) are deliberately not carried here yet; requests are built with
/// caching off, matching a freshly constructed
/// [`GenericCompletionModel`](super::completion::GenericCompletionModel).
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
}

impl Config {
    /// Config for `model` reading `ANTHROPIC_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        let model = model.into();
        let default_max_tokens = default_max_tokens_for_model(&model);
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: ApiKeyLocation::Env("ANTHROPIC_API_KEY".to_string()),
            model,
            extra_headers: Vec::new(),
            anthropic_version: ANTHROPIC_VERSION_LATEST.to_string(),
            anthropic_betas: Vec::new(),
            default_max_tokens,
        }
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
            model, request, max_tokens, false, false, None,
        )?;
        Ok(serde_json::to_vec(&body)?)
    } else {
        request.max_tokens = Some(max_tokens);
        let typed = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: &model,
            request,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
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
        Transport::Reqwest(client) => Ok(super::streaming::stream_anthropic_sse(
            client.clone(),
            req,
            DESCRIPTOR.name,
            span,
        )),
        #[cfg(any(test, feature = "test-utils"))]
        Transport::Recording(client) => Ok(super::streaming::stream_anthropic_sse(
            client.clone(),
            req,
            DESCRIPTOR.name,
            span,
        )),
        #[cfg(any(test, feature = "test-utils"))]
        Transport::Sequenced(client) => Ok(super::streaming::stream_anthropic_sse(
            client.clone(),
            req,
            DESCRIPTOR.name,
            span,
        )),
    }
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
    let mut builder =
        http::Request::get(url).header("anthropic-version", cfg.anthropic_version.as_str());
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
/// The classic `ModelListingClient` path parses through the same pure
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

#[cfg(test)]
mod tests {
    use super::*;

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
