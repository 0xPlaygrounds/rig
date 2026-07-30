//! Cohere v2 Chat API as config + pure functions.
//!
//! The data-oriented face of the Cohere provider: a serde [`Config`], a
//! [`DESCRIPTOR`] capability sheet, and free functions decomposing a
//! completion into pure parts — [`build_request_body`] / [`build_request`]
//! (data → HTTP request, no IO) and [`parse_response`] (bytes → normalized
//! [`completion::CompletionResponse`], no IO) — plus async [`complete`] and
//! [`open_stream`] wrappers over [`HttpRuntime`].
//!
//! The pure functions delegate to the same typed conversion the
//! [`CompletionModel`](super::CompletionModel) trait path uses
//! (`CohereCompletionRequest`'s `TryFrom` plus the shared stream-flag
//! helper), so both paths produce byte-identical request bodies.
//! [`open_stream`] drives the exact SSE machinery the trait path uses.

use http::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};

use super::completion::{CohereCompletionRequest, apply_stream_flag};
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_runtime::HttpRuntime;
use crate::providers::descriptor::{
    ApiKeyLocation, ConfigError, ProviderDescriptor, required_env_var,
};
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder};

/// Default Cohere API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.cohere.ai";

/// Cohere's capability sheet.
///
/// `supports_response_format` is `false`: the request conversion warns and
/// drops `output_schema` ("Structured outputs currently not supported for
/// Cohere").
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    name: "cohere",
    supports_tools: true,
    supports_response_format: false,
    stream_include_usage: false,
    emits_complete_single_chunk_tool_calls: false,
    composes_native_output_with_tools: false,
    max_embedding_documents: Some(96),
};

/// Plain-data Cohere provider configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Config {
    /// API base URL (defaults to [`DEFAULT_BASE_URL`]).
    pub base_url: String,
    /// Credential location; sent as a bearer `Authorization` header.
    pub api_key: ApiKeyLocation,
    /// Model identifier requests are built for.
    pub model: String,
    /// Extra headers attached to every request.
    pub extra_headers: Vec<(String, String)>,
}

impl Config {
    /// Config for `model` reading `COHERE_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: ApiKeyLocation::Env("COHERE_API_KEY".to_string()),
            model: model.into(),
            extra_headers: Vec::new(),
        }
    }

    /// Config for `model` built from the process environment.
    ///
    /// Reads `COHERE_API_KEY` (required) — the same variable the deleted
    /// `cohere::Client::from_env` read. There is no base-URL override: the
    /// classic client always targeted [`DEFAULT_BASE_URL`]. The credential is
    /// validated eagerly but stored as [`ApiKeyLocation::Env`], so the secret is
    /// read at request time rather than held inside the config.
    ///
    /// # Errors
    /// [`ConfigError`] when a required variable is missing or invalid.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
        let cfg = Self::new(model);
        required_env_var("COHERE_API_KEY")?;
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

/// Build the serialized v2 Chat request body for `request`.
///
/// Pure: the exact bytes the wire sees. Delegates to the same typed
/// conversion as the trait path; `stream` merges the top-level
/// `stream: true` flag exactly as the trait streaming path does.
pub fn build_request_body(
    cfg: &Config,
    request: &CompletionRequest,
    stream: bool,
) -> Result<Vec<u8>, CompletionError> {
    let mut typed = CohereCompletionRequest::try_from((cfg.model.as_str(), request.clone()))?;
    if stream {
        apply_stream_flag(&mut typed);
    }
    Ok(serde_json::to_vec(&typed)?)
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
    let url = format!("{}/v2/chat", cfg.base_url.trim_end_matches('/'));
    let body = build_request_body(cfg, request, stream)?;

    let mut builder = http::Request::post(url).header(CONTENT_TYPE, "application/json");
    if let Some(key) = cfg
        .api_key
        .resolve()
        .map_err(|e| CompletionError::RequestError(Box::new(e)))?
    {
        builder = builder.header(AUTHORIZATION, format!("Bearer {key}"));
    }
    for (name, value) in &cfg.extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    builder
        .body(body)
        .map_err(|e| CompletionError::RequestError(Box::new(e)))
}

/// Parse a v2 Chat response body into the normalized
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
    let response: super::completion::CompletionResponse = serde_json::from_str(body)?;
    response.try_into()
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
    Ok(super::streaming::stream_cohere_sse(
        rt.sse_events(req, false),
        span,
    ))
}

/// Send `request` to Cohere and return the normalized response.
pub async fn complete(
    cfg: &Config,
    rt: &HttpRuntime,
    request: CompletionRequest,
) -> Result<completion::CompletionResponse, CompletionError> {
    let req = build_request(cfg, &request, false)?;
    let (status, body) = rt.send(req).await?;
    parse_response(status, &body)
}

// ================================================================
// Embeddings
// ================================================================

/// Plain-data Cohere embeddings configuration.
///
/// A sibling of [`Config`]: embeddings carry their own model plus the
/// Cohere-specific `input_type`, which do not belong on the completion
/// configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingConfig {
    /// API base URL (defaults to [`DEFAULT_BASE_URL`]).
    pub base_url: String,
    /// Credential location; sent as a bearer `Authorization` header.
    pub api_key: ApiKeyLocation,
    /// Embedding model identifier requests are built for.
    pub model: String,
    /// Cohere embedding `input_type` (defaults to `search_document`).
    pub input_type: String,
    /// Extra headers attached to every request.
    pub extra_headers: Vec<(String, String)>,
}

impl EmbeddingConfig {
    /// Config for `model` reading `COHERE_API_KEY` from the environment,
    /// embedding with `input_type: search_document`.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: ApiKeyLocation::Env("COHERE_API_KEY".to_string()),
            model: model.into(),
            input_type: "search_document".to_string(),
            extra_headers: Vec::new(),
        }
    }

    /// Config for `model` built from the process environment.
    ///
    /// Same variable as [`Config::from_env`]: `COHERE_API_KEY` (required).
    ///
    /// # Errors
    /// [`ConfigError`] when a required variable is missing or invalid.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
        let cfg = Self::new(model);
        required_env_var("COHERE_API_KEY")?;
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

    /// Override the Cohere `input_type`.
    pub fn with_input_type(mut self, input_type: impl Into<String>) -> Self {
        self.input_type = input_type.into();
        self
    }
}

/// Build the complete HTTP `/v1/embed` request for one chunk of `texts`.
///
/// Pure except for credential resolution.
pub fn build_embedding_request(
    cfg: &EmbeddingConfig,
    texts: &[String],
) -> Result<http::Request<Vec<u8>>, crate::embeddings::EmbeddingError> {
    use crate::embeddings::EmbeddingError;

    let body = super::embeddings::build_embedding_body(&cfg.model, &cfg.input_type, texts)?;
    let url = format!("{}/v1/embed", cfg.base_url.trim_end_matches('/'));
    let mut builder = http::Request::post(url).header(CONTENT_TYPE, "application/json");
    if let Some(key) = cfg
        .api_key
        .resolve()
        .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?
    {
        builder = builder.header(AUTHORIZATION, format!("Bearer {key}"));
    }
    for (name, value) in &cfg.extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    builder
        .body(body)
        .map_err(|e| EmbeddingError::ProviderError(e.to_string()))
}

/// Parse a `/v1/embed` response into the normalized
/// [`crate::embeddings::EmbeddingResponse`]. Pure.
pub fn parse_embedding_response(
    status: http::StatusCode,
    body: &str,
    documents: Vec<String>,
) -> Result<crate::embeddings::EmbeddingResponse, crate::embeddings::EmbeddingError> {
    super::embeddings::parse_embedding_response(status, body, documents)
}

/// Embed `texts`, chunking to honor [`DESCRIPTOR`]'s
/// `max_embedding_documents` (Cohere caps requests at 96 documents);
/// embeddings are returned in input order.
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
            temperature: Some(0.3),
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    #[test]
    fn build_request_body_matches_typed_conversion() {
        let cfg = Config::new("command-r-plus").with_api_key("k");
        let body = build_request_body(&cfg, &sample_request(), false).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["model"], "command-r-plus");
        assert_eq!(value["temperature"], 0.3);
        assert!(value.get("stream").is_none());

        let streaming = build_request_body(&cfg, &sample_request(), true).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&streaming).expect("json");
        assert_eq!(value["stream"], true);
    }

    #[test]
    fn build_request_honors_model_override() {
        let cfg = Config::new("command-r-plus").with_api_key("k");
        let mut request = sample_request();
        request.model = Some("command-r".to_string());
        let body = build_request_body(&cfg, &request, false).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["model"], "command-r");
    }

    #[test]
    fn build_request_sets_url_and_auth() {
        let cfg = Config::new("command-r-plus").with_api_key("secret");
        let req = build_request(&cfg, &sample_request(), false).expect("build");
        assert_eq!(req.uri(), "https://api.cohere.ai/v2/chat");
        assert_eq!(
            req.headers()
                .get(http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok()),
            Some("Bearer secret")
        );
    }

    #[test]
    fn parse_response_normalizes() {
        let body = serde_json::json!({
            "id": "abc123",
            "finish_reason": "COMPLETE",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hi"}]
            },
            "usage": {
                "tokens": {"input_tokens": 3.0, "output_tokens": 2.0}
            }
        })
        .to_string();
        let response = parse_response(http::StatusCode::OK, &body).expect("parse");
        assert_eq!(response.provider, "cohere");
        assert_eq!(response.usage.input_tokens, 3);
        assert_eq!(response.usage.total_tokens, 5);
    }

    #[test]
    fn parse_response_preserves_error_status() {
        let error = parse_response(http::StatusCode::SERVICE_UNAVAILABLE, r#"{"error":"boom"}"#)
            .expect_err("should error");
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
    }
}
