//! Gemini `generateContent` as config + pure functions.
//!
//! The data-oriented face of the Gemini provider: a serde [`Config`], a
//! [`DESCRIPTOR`] capability sheet, and free functions —
//! [`build_request`] (data → HTTP request, no IO) and [`parse_response`]
//! (bytes → normalized [`completion::CompletionResponse`], no IO) — plus the
//! async [`complete`] and [`open_stream`] wrappers over
//! [`HttpRuntime`](crate::http_runtime::HttpRuntime).
//!
//! The pure functions delegate to the same typed conversion
//! (`super::completion::create_request_body` /
//! `GenerateContentResponse::try_into`) that the
//! [`CompletionModel`](super::completion::CompletionModel) trait path uses,
//! so both paths produce byte-identical request bodies.
//!
//! Gemini's URL embeds the model and the operation
//! (`:generateContent` / `:streamGenerateContent`) and the credential rides
//! as a `key` query parameter (with `alt=sse` for streaming), mirroring
//! [`GeminiExt`](super::client::GeminiExt)'s `build_uri`.
//!
//! Future work: the same treatment for the Gemini Interactions API surface
//! (`super::interactions_api`), which this module deliberately does not cover.

use http::header::CONTENT_TYPE;
use serde::{Deserialize, Serialize};

use super::completion::gemini_api_types::GenerateContentResponse;
use super::completion::{
    completion_endpoint, create_request_body, resolve_request_model, streaming_endpoint,
};
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_runtime::HttpRuntime;
use crate::providers::descriptor::{ApiKeyLocation, ProviderDescriptor};

/// Default Gemini API base URL.
pub const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";

/// Gemini's capability sheet.
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    name: "gemini",
    supports_tools: true,
    // `output_schema` maps to `generationConfig.responseJsonSchema`.
    supports_response_format: true,
    // Usage arrives in `usageMetadata` on streaming chunks; there is no
    // OpenAI-style `stream_options` opt-in.
    stream_include_usage: false,
    // Gemini emits whole `functionCall` parts in a single streaming chunk.
    emits_complete_single_chunk_tool_calls: true,
    // The request builder sends `responseJsonSchema` and `tools` together
    // without gating.
    composes_native_output_with_tools: true,
    max_embedding_documents: Some(1024),
};

/// Plain-data Gemini provider configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Config {
    /// API base URL (defaults to [`DEFAULT_BASE_URL`]).
    pub base_url: String,
    /// Credential location. Gemini sends the key as a `key` query parameter,
    /// not a header.
    pub api_key: ApiKeyLocation,
    /// Model identifier requests are built for.
    pub model: String,
    /// Extra headers attached to every request.
    pub extra_headers: Vec<(String, String)>,
}

impl Config {
    /// Config for `model` reading `GEMINI_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: ApiKeyLocation::Env("GEMINI_API_KEY".to_string()),
            model: model.into(),
            extra_headers: Vec::new(),
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

/// Build the serialized `generateContent` request body for `request`.
///
/// Pure: the exact bytes the wire sees. Gemini's streaming/non-streaming
/// choice lives in the URL, not the body, so there is no `stream` parameter.
pub fn build_request_body(
    cfg: &Config,
    request: &CompletionRequest,
) -> Result<Vec<u8>, CompletionError> {
    let _ = cfg;
    let typed = create_request_body(request.clone())?;
    Ok(serde_json::to_vec(&typed)?)
}

/// Build the complete HTTP request (URL, headers, body) for `request`.
///
/// Pure except for credential resolution (`ApiKeyLocation::Env` reads the
/// environment). `stream` selects `:streamGenerateContent?alt=sse` over
/// `:generateContent`; the resolved key rides as `key=` in the query string,
/// matching the classic client's URL shape.
pub fn build_request(
    cfg: &Config,
    request: &CompletionRequest,
    stream: bool,
) -> Result<http::Request<Vec<u8>>, CompletionError> {
    let model = resolve_request_model(&cfg.model, request);
    let path = if stream {
        streaming_endpoint(&model)
    } else {
        completion_endpoint(&model)
    };
    let key = cfg
        .api_key
        .resolve()
        .map_err(|e| CompletionError::RequestError(Box::new(e)))?;
    let query = match (stream, key) {
        (true, Some(key)) => format!("?alt=sse&key={key}"),
        (true, None) => "?alt=sse".to_string(),
        (false, Some(key)) => format!("?key={key}"),
        (false, None) => String::new(),
    };
    let url = format!(
        "{}/{}{}",
        cfg.base_url.trim_end_matches('/'),
        path.trim_start_matches('/'),
        query
    );
    let body = build_request_body(cfg, request)?;

    let mut builder = http::Request::post(url).header(CONTENT_TYPE, "application/json");
    for (name, value) in &cfg.extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    builder
        .body(body)
        .map_err(|e| CompletionError::RequestError(Box::new(e)))
}

/// Parse a `generateContent` response body into the normalized
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
    let response: GenerateContentResponse = serde_json::from_str(body)?;
    response.try_into()
}

/// Open a streaming completion for `request` over
/// `:streamGenerateContent?alt=sse`, reusing the provider's SSE machinery.
pub async fn open_stream(
    cfg: &Config,
    rt: &HttpRuntime,
    request: CompletionRequest,
) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError> {
    use crate::http_runtime::Transport;

    let req = build_request(cfg, &request, true)?;
    match rt.transport() {
        Transport::Reqwest(client) => Ok(crate::streaming::StreamingCompletionResponse::stream(
            Box::pin(super::streaming::generate_content_stream(
                client.clone(),
                req,
            )),
        )),
        #[cfg(any(test, feature = "test-utils"))]
        Transport::Recording(client) => Ok(crate::streaming::StreamingCompletionResponse::stream(
            Box::pin(super::streaming::generate_content_stream(
                client.clone(),
                req,
            )),
        )),
        #[cfg(any(test, feature = "test-utils"))]
        Transport::Sequenced(client) => Ok(crate::streaming::StreamingCompletionResponse::stream(
            Box::pin(super::streaming::generate_content_stream(
                client.clone(),
                req,
            )),
        )),
    }
}

/// The keyed `generateContent` URL for `cfg`'s model.
fn generate_content_url(cfg: &Config) -> Result<String, crate::http_client::Error> {
    let path = format!("/v1beta/models/{}:generateContent", cfg.model);
    let key = cfg
        .api_key
        .resolve()
        .map_err(|e| crate::http_client::Error::Instance(e.into()))?;
    let query = match key {
        Some(key) => format!("?key={key}"),
        None => String::new(),
    };
    Ok(format!(
        "{}/{}{}",
        cfg.base_url.trim_end_matches('/'),
        path.trim_start_matches('/'),
        query
    ))
}

fn modality_request(
    cfg: &Config,
    body: Vec<u8>,
) -> Result<http::Request<Vec<u8>>, crate::http_client::Error> {
    let url = generate_content_url(cfg)?;
    let mut builder = http::Request::post(url).header(CONTENT_TYPE, "application/json");
    for (name, value) in &cfg.extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    builder.body(body).map_err(crate::http_client::Error::from)
}

/// Transcribe `request` via Gemini's `generateContent` endpoint (audio rides
/// inline as base64).
pub async fn transcribe(
    cfg: &Config,
    rt: &HttpRuntime,
    request: crate::transcription::TranscriptionRequest,
) -> Result<
    crate::transcription::TranscriptionResponse<GenerateContentResponse>,
    crate::transcription::TranscriptionError,
> {
    let body = super::transcription::build_transcription_body(request)?;
    let req = modality_request(cfg, body)?;
    let (status, body) = rt.send_bytes(req).await?;
    super::transcription::parse_transcription_response(status, &body)
}

/// Generate an image via Gemini's `generateContent` endpoint.
#[cfg(feature = "image")]
pub async fn generate_image(
    cfg: &Config,
    rt: &HttpRuntime,
    request: crate::image_generation::ImageGenerationRequest,
) -> Result<
    crate::image_generation::ImageGenerationResponse<GenerateContentResponse>,
    crate::image_generation::ImageGenerationError,
> {
    let typed = super::image_generation::create_request_body(request)?;
    let body = serde_json::to_vec(&typed)?;
    let req = modality_request(cfg, body)?;
    let (status, body) = rt.send_bytes(req).await?;
    super::image_generation::parse_image_generation_response(
        status,
        &String::from_utf8_lossy(&body),
    )
}

/// Send `request` to Gemini and return the normalized response.
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

/// Plain-data Gemini embeddings configuration.
///
/// A sibling of [`Config`]: embeddings target their own model identifier
/// plus an optional `output_dimensionality`, which do not belong on the
/// completion configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingConfig {
    /// API base URL (defaults to [`DEFAULT_BASE_URL`]).
    pub base_url: String,
    /// Credential location; sent as a `key` query parameter.
    pub api_key: ApiKeyLocation,
    /// Embedding model identifier requests are built for.
    pub model: String,
    /// Requested embedding dimensions, sent per entry as
    /// `output_dimensionality` when set.
    pub dimensions: Option<usize>,
    /// Extra headers attached to every request.
    pub extra_headers: Vec<(String, String)>,
}

impl EmbeddingConfig {
    /// Config for `model` reading `GEMINI_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: ApiKeyLocation::Env("GEMINI_API_KEY".to_string()),
            model: model.into(),
            dimensions: None,
            extra_headers: Vec::new(),
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

    /// Request `dimensions`-sized embeddings.
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }
}

/// Build the complete HTTP `batchEmbedContents` request for one chunk of
/// `texts`.
///
/// Pure except for credential resolution; the resolved key rides as `key=`
/// in the query string, matching the classic client's URL shape.
pub fn build_embedding_request(
    cfg: &EmbeddingConfig,
    texts: &[String],
) -> Result<http::Request<Vec<u8>>, crate::embeddings::EmbeddingError> {
    use crate::embeddings::EmbeddingError;

    let body = super::embedding::build_embedding_body(&cfg.model, texts, cfg.dimensions)?;
    let key = cfg
        .api_key
        .resolve()
        .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?;
    let query = match key {
        Some(key) => format!("?key={key}"),
        None => String::new(),
    };
    let url = format!(
        "{}/v1beta/models/{}:batchEmbedContents{}",
        cfg.base_url.trim_end_matches('/'),
        cfg.model,
        query
    );
    let mut builder = http::Request::post(url).header(CONTENT_TYPE, "application/json");
    for (name, value) in &cfg.extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    builder
        .body(body)
        .map_err(|e| EmbeddingError::ProviderError(e.to_string()))
}

/// Parse a `batchEmbedContents` response into the normalized
/// [`crate::embeddings::EmbeddingResponse`]. Pure.
pub fn parse_embedding_response(
    status: http::StatusCode,
    body: &str,
    documents: Vec<String>,
) -> Result<crate::embeddings::EmbeddingResponse, crate::embeddings::EmbeddingError> {
    super::embedding::parse_embedding_response(status, body, documents)
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
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    #[test]
    fn build_request_sets_url_with_key_query_param() {
        let cfg = Config::new("gemini-2.0-flash").with_api_key("secret");
        let req = build_request(&cfg, &sample_request(), false).expect("build");
        assert_eq!(
            req.uri(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=secret"
        );
        // The key rides in the query string; there is no Authorization header.
        assert!(req.headers().get(http::header::AUTHORIZATION).is_none());
        assert_eq!(
            req.headers()
                .get(http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
    }

    #[test]
    fn build_request_streaming_uses_sse_endpoint() {
        let cfg = Config::new("gemini-2.0-flash").with_api_key("secret");
        let req = build_request(&cfg, &sample_request(), true).expect("build");
        assert_eq!(
            req.uri(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse&key=secret"
        );
    }

    #[test]
    fn build_request_prefers_per_request_model() {
        let cfg = Config::new("gemini-2.0-flash").with_api_key("k");
        let mut request = sample_request();
        request.model = Some("gemini-2.5-flash".to_string());
        let req = build_request(&cfg, &request, false).expect("build");
        assert!(
            req.uri()
                .to_string()
                .contains("/models/gemini-2.5-flash:generateContent")
        );
    }

    #[test]
    fn build_request_body_matches_typed_conversion() {
        let cfg = Config::new("gemini-2.0-flash").with_api_key("k");
        let body = build_request_body(&cfg, &sample_request()).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        // Gemini carries the model in the URL, never the body.
        assert!(value.get("model").is_none());
        assert_eq!(value["contents"][0]["role"], "user");
        assert_eq!(value["contents"][0]["parts"][0]["text"], "hello");
    }

    #[test]
    fn parse_response_normalizes() {
        let body = serde_json::json!({
            "responseId": "resp-1",
            "modelVersion": "gemini-2.0-flash-001",
            "candidates": [{
                "content": {
                    "parts": [{"text": "hi"}],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0
            }],
            "usageMetadata": {
                "promptTokenCount": 3,
                "candidatesTokenCount": 2,
                "totalTokenCount": 5
            }
        })
        .to_string();
        let response = parse_response(http::StatusCode::OK, &body).expect("parse");
        assert_eq!(response.provider, "gemini");
        assert_eq!(response.model.as_deref(), Some("gemini-2.0-flash-001"));
        assert_eq!(response.message_id.as_deref(), Some("resp-1"));
        assert_eq!(response.finish_reason, Some(completion::FinishReason::Stop));
        assert_eq!(response.usage.input_tokens, 3);
        assert_eq!(response.usage.total_tokens, 5);
    }

    #[test]
    fn parse_response_surfaces_http_errors() {
        let err = parse_response(http::StatusCode::SERVICE_UNAVAILABLE, "boom")
            .expect_err("non-success status must error");
        assert!(matches!(err, CompletionError::HttpError(_)));
    }
}
