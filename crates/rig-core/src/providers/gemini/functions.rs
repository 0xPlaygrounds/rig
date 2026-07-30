//! Gemini `generateContent` as config + pure functions.
//!
//! The data-oriented face of the Gemini provider: a serde [`Config`], a
//! [`DESCRIPTOR`] capability sheet, and free functions —
//! [`build_request`] (data → HTTP request, no IO) and [`parse_response`]
//! (bytes → normalized [`completion::CompletionResponse`], no IO) — plus the
//! async [`complete`] and [`open_stream`] wrappers over
//! [`HttpRuntime`].
//!
//! The pure functions delegate to the typed wire conversions in
//! [`super::completion`] (`create_request_body` /
//! `GenerateContentResponse::try_into`).
//!
//! Gemini's URL embeds the model and the operation
//! (`:generateContent` / `:streamGenerateContent`) and the credential rides
//! as a `key` query parameter (with `alt=sse` for streaming).
//!
//! This module covers `generateContent` only. The Interactions API is a
//! separate surface with its own credential placement (an `x-goog-api-key`
//! header rather than `?key=`); its face is
//! [`super::interactions_api::functions`].

use http::header::CONTENT_TYPE;
use serde::{Deserialize, Serialize};

use super::completion::gemini_api_types::GenerateContentResponse;
use super::completion::{
    completion_endpoint, create_request_body, resolve_request_model, streaming_endpoint,
};
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_runtime::HttpRuntime;
use crate::providers::descriptor::{
    ApiKeyLocation, ConfigError, ProviderDescriptor, required_env_var,
};
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder};

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
    verify_path: Some("/v1beta/models"),
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

    /// Config for `model` built from the process environment.
    ///
    /// Reads `GEMINI_API_KEY` (required) — the same variable the deleted
    /// deleted `gemini::Client::from_env` read. There is no base-URL override:
    /// the classic client always targeted [`DEFAULT_BASE_URL`]. The credential is
    /// validated eagerly but stored as [`ApiKeyLocation::Env`], so the secret is
    /// read at request time rather than held inside the config (Gemini sends it
    /// as a `key` query parameter).
    ///
    /// # Errors
    /// [`ConfigError`] when a required variable is missing or invalid.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
        let cfg = Self::new(model);
        required_env_var("GEMINI_API_KEY")?;
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
/// as Gemini requires.
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
    let model = resolve_request_model(&cfg.model, &request);
    // `gcp.gemini` is the OTel `gen_ai.provider.name` value the deleted
    // `CompletionModel::stream` recorded; `DESCRIPTOR.name` ("gemini") is the
    // normalized-response provider field, a different vocabulary.
    let span = CompletionSpanBuilder::new("gcp.gemini", &model, CompletionOperation::ChatStreaming)
        .system_instructions(
            request.preamble.as_deref(),
            request.record_telemetry_content,
        )
        .build();
    let req = build_request(cfg, &request, true)?;
    Ok(crate::streaming::StreamingCompletionResponse::stream(
        Box::pin(super::streaming::generate_content_stream(
            rt.sse_events(req, false),
            span,
        )),
    ))
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

    /// Config for `model` built from the process environment.
    ///
    /// Same variable as [`Config::from_env`]: `GEMINI_API_KEY` (required).
    ///
    /// # Errors
    /// [`ConfigError`] when a required variable is missing or invalid.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
        let cfg = Self::new(model);
        required_env_var("GEMINI_API_KEY")?;
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
/// in the query string.
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

/// Build one `GET /v1beta/models` page request for [`list_models`].
///
/// Pure except for credential resolution; the resolved key rides as `key=`
/// in the query string.
pub fn build_list_models_request(
    cfg: &Config,
    page_token: Option<&str>,
) -> Result<http::Request<Vec<u8>>, crate::model::ModelListingError> {
    use crate::model::ModelListingError;

    let path = super::model_listing::list_models_path(page_token);
    let key = cfg
        .api_key
        .resolve()
        .map_err(|e| ModelListingError::request_error(e.to_string()))?;
    let url = match key {
        Some(key) => format!("{}{}&key={}", cfg.base_url.trim_end_matches('/'), path, key),
        None => format!("{}{}", cfg.base_url.trim_end_matches('/'), path),
    };
    // Classic parity: the deleted client stamped `accept`/`content-type` on
    // every request, bodyless GETs included, and the recordings match on them.
    let mut builder = http::Request::get(url)
        .header(http::header::ACCEPT, "*/*")
        .header(http::header::CONTENT_TYPE, "application/json");
    for (name, value) in &cfg.extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    builder
        .body(Vec::new())
        .map_err(|e| ModelListingError::request_error(e.to_string()))
}

/// List the models available to `cfg`'s credentials, following page-token
/// pagination through all pages.
///
/// Parses through the pure page parser
/// (`model_listing::parse_models_page`).
pub async fn list_models(
    cfg: &Config,
    rt: &HttpRuntime,
) -> Result<crate::model::ModelList, crate::model::ModelListingError> {
    use crate::model::{ModelList, ModelListingError};

    let mut all_models = Vec::new();
    let mut page_token: Option<String> = None;

    loop {
        let path = super::model_listing::list_models_path(page_token.as_deref());
        let req = build_list_models_request(cfg, page_token.as_deref())?;
        let (status, body) = rt.send_bytes(req).await?;

        if !status.is_success() {
            return Err(ModelListingError::api_error_with_context(
                "Gemini",
                &path,
                status.as_u16(),
                &body,
            ));
        }

        let (models, next_page_token) = super::model_listing::parse_models_page(&body, &path)?;
        all_models.extend(models);

        match next_page_token {
            Some(token) => page_token = Some(token),
            None => break,
        }
    }

    Ok(ModelList::new(all_models))
}
/// Build the `GET /v1beta/models` credential-verification request.
///
/// Gemini authenticates with a `key` query parameter, not a header, so this
/// builds the request itself rather than using
/// [`crate::providers::verify::verify_bearer`].
///
/// # Errors
/// [`VerifyError`](crate::providers::verify::VerifyError) when the
/// credential cannot be resolved.
pub fn build_verify_request(
    cfg: &Config,
) -> Result<http::Request<Vec<u8>>, crate::providers::verify::VerifyError> {
    use crate::providers::verify::{VerifyError, verify_url};

    let base = verify_url(&DESCRIPTOR, &cfg.base_url)?;
    let key = cfg
        .api_key
        .resolve()
        .map_err(|e| VerifyError::ProviderError(e.to_string()))?;
    let url = match key {
        Some(key) => format!("{base}?key={key}"),
        None => base,
    };
    let mut builder = http::Request::get(url);
    for (name, value) in &cfg.extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    builder
        .body(Vec::new())
        .map_err(|e| VerifyError::ProviderError(e.to_string()))
}

/// Verify that `cfg`'s credential is accepted by Gemini.
///
/// The data-oriented replacement for the deleted `VerifyClient::verify`: the
/// endpoint is [`DESCRIPTOR`]'s `verify_path` (`/v1beta/models`, the value
/// the deleted `Provider::VERIFY_PATH` carried) and the status mapping is
/// the classic one — see [`crate::providers::verify`].
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

    #[test]
    fn build_list_models_request_puts_key_and_page_token_in_query() {
        let cfg = Config::new("gemini-2.0-flash").with_api_key("secret");
        let req = build_list_models_request(&cfg, None).expect("build");
        assert_eq!(req.method(), http::Method::GET);
        assert_eq!(
            req.uri(),
            "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000&key=secret"
        );

        let paged = build_list_models_request(&cfg, Some("tok")).expect("build");
        assert_eq!(
            paged.uri(),
            "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000&pageToken=tok&key=secret"
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
#[cfg(test)]
mod telemetry_tests {
    use super::*;
    use futures::StreamExt;
    use std::io;
    use std::sync::{Arc, Mutex};

    use crate::OneOrMany;
    use crate::message::Message;
    use crate::test_utils::MockStreamingClient;

    #[derive(Clone)]
    struct SharedWriter(Arc<Mutex<Vec<u8>>>);

    impl io::Write for SharedWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if let Ok(mut sink) = self.0.lock() {
                sink.extend_from_slice(buf);
            }
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

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

    /// The deleted `gemini::completion::CompletionModel::stream` built a
    /// `CompletionSpanBuilder` span (`gcp.gemini`, `ChatStreaming`) and
    /// instrumented the SSE stream with it; `functions::open_stream` had lost
    /// it, so `record_token_usage` wrote to whatever ambient span existed.
    #[tokio::test]
    async fn open_stream_emits_the_chat_streaming_completion_span() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let captured = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_ansi(false)
            .without_time()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::NEW)
            .with_writer({
                let captured = captured.clone();
                move || SharedWriter(captured.clone())
            })
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);

        let sse = format!(
            "data: {}\n\n",
            serde_json::json!({
                "responseId": "resp-1",
                "modelVersion": "gemini-2.0-flash-001",
                "candidates": [{
                    "content": { "parts": [{"text": "hi"}], "role": "model" },
                    "finishReason": "STOP",
                    "index": 0
                }],
                "usageMetadata": {
                    "promptTokenCount": 3,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 5
                }
            })
        );
        let rt = HttpRuntime::mock_streaming(MockStreamingClient {
            sse_bytes: bytes::Bytes::from(sse),
        });
        let cfg = Config::new("gemini-2.0-flash").with_api_key("secret");

        let mut stream = open_stream(&cfg, &rt, sample_request())
            .await
            .expect("stream opens");
        while stream.next().await.is_some() {}

        let logs = String::from_utf8(captured.lock().map(|sink| sink.clone()).unwrap_or_default())
            .expect("utf8 logs");
        assert!(
            logs.contains("chat_streaming"),
            "no chat_streaming span was created: {logs}"
        );
        assert!(
            logs.contains("gcp.gemini"),
            "span did not carry the classic `gcp.gemini` provider name: {logs}"
        );
        assert!(
            logs.contains("gemini-2.0-flash"),
            "span did not carry the request model: {logs}"
        );
    }
}
