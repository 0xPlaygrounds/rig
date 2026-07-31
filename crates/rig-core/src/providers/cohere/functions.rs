//! Cohere v2 Chat API as config + pure functions.
//!
//! The data-oriented face of the Cohere provider: a serde [`Config`], a
//! [`DESCRIPTOR`] capability sheet, and free functions decomposing a
//! completion into pure parts — [`build_request_body`] / [`build_request`]
//! (data → HTTP request, no IO) and [`parse_response`] (bytes → normalized
//! [`completion::CompletionResponse`], no IO) — plus async [`complete`] and
//! [`open_stream`] wrappers over [`HttpRuntime`].
//!
//! The pure functions delegate to the typed `CohereCompletionRequest`
//! conversion plus the shared stream-flag helper in
//! [`super::completion`]; [`open_stream`] drives the SSE machinery in
//! [`super::streaming`].

use http::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};

use super::completion::{CohereCompletionRequest, apply_stream_flag};
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_runtime::HttpRuntime;
use crate::providers::descriptor::{
    ApiKeyLocation, ConfigError, ProviderDescriptor, required_env_var,
};
use crate::telemetry::{CompletionOperation, completion_span};

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
    verify_path: Some("/models"),
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
    let span = completion_span(
        DESCRIPTOR.name,
        &model,
        CompletionOperation::ChatStreaming,
        &request,
    );
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
    use crate::message::Message;

    fn sample_request() -> CompletionRequest {
        CompletionRequest {
            model: None,
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
#[cfg(test)]
mod usage_tests {
    use super::*;

    /// A Cohere v2 `/v2/chat` success body carrying **both** usage blocks, as
    /// the wire does.
    const RESPONSE_WITH_BOTH_USAGE_BLOCKS: &str = r#"{
        "id": "abc123",
        "finish_reason": "COMPLETE",
        "message": {
            "role": "assistant",
            "content": [{ "type": "text", "text": "hello" }]
        },
        "usage": {
            "billed_units": { "input_tokens": 78, "output_tokens": 27 },
            "tokens": { "input_tokens": 1028, "output_tokens": 63 }
        }
    }"#;

    /// Cohere reports two usage blocks: `usage.tokens` is the actual token
    /// count the model processed, and `usage.billed_units` is the (smaller)
    /// billable subset. The normalized [`crate::completion::Usage`] fields are
    /// token counts, so `usage.tokens` is the correct source — and it is what
    /// the streaming path's `streamed_token_usage` has always read.
    ///
    /// The deleted classic model's `Usage::token_usage` read `billed_units`
    /// for its telemetry span while its own `TryFrom` read `tokens`, so the
    /// two halves of one response disagreed. This pins the surviving,
    /// consistent answer.
    #[test]
    fn parse_response_reads_usage_tokens_not_billed_units() {
        let response =
            parse_response(http::StatusCode::OK, RESPONSE_WITH_BOTH_USAGE_BLOCKS).expect("parse");

        assert_eq!(response.usage.input_tokens, 1028);
        assert_eq!(response.usage.output_tokens, 63);
        assert_eq!(response.usage.total_tokens, 1091);

        // The billed-unit counts must not leak into any usage field.
        assert_ne!(response.usage.input_tokens, 78);
        assert_ne!(response.usage.output_tokens, 27);
    }

    /// A response with only `billed_units` reports zero usage rather than
    /// silently substituting billed units for token counts.
    #[test]
    fn parse_response_reports_zero_usage_without_a_tokens_block() {
        let body = r#"{
            "id": "abc123",
            "finish_reason": "COMPLETE",
            "message": {
                "role": "assistant",
                "content": [{ "type": "text", "text": "hello" }]
            },
            "usage": {
                "billed_units": { "input_tokens": 78, "output_tokens": 27 }
            }
        }"#;

        let response = parse_response(http::StatusCode::OK, body).expect("parse");
        assert_eq!(response.usage.input_tokens, 0);
        assert_eq!(response.usage.output_tokens, 0);
        assert_eq!(response.usage.total_tokens, 0);
    }

    #[test]
    fn verify_path_matches_the_deleted_provider_const() {
        assert_eq!(DESCRIPTOR.verify_path, Some("/models"));
    }

    /// Cohere renders system instructions as `system`-role messages in the
    /// array (it has no dedicated system field), so canonical `Message::System`
    /// entries must survive conversion in order. Cohere has no cassette
    /// coverage, so this is its only replay-equivalent proof.
    #[test]
    fn system_messages_become_ordered_system_role_messages() {
        let cfg = Config::new("command-r");
        let request = crate::completion::CompletionRequest::builder("now")
            .preamble("first")
            .message(crate::message::Message::system("second"))
            .message(crate::message::Message::user("earlier"))
            .build();

        let body = build_request_body(&cfg, &request, false).expect("request builds");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        let messages = value["messages"].as_array().expect("messages array");

        let systems: Vec<String> = messages
            .iter()
            .filter(|m| m["role"] == "system")
            .map(|m| {
                m["content"]
                    .as_str()
                    .map(str::to_string)
                    .or_else(|| m["content"][0]["text"].as_str().map(str::to_string))
                    .expect("system content")
            })
            .collect();
        assert_eq!(systems, ["first", "second"], "order must be preserved");
    }
}
