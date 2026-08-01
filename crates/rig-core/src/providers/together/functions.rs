//! Together AI chat completions as config + pure functions.
//!
//! The data-oriented face of the Together AI provider, mirroring
//! [`crate::providers::openai::functions`]: a serde [`Config`], a
//! [`DESCRIPTOR`] capability sheet, and pure
//! [`build_request`]/[`parse_response`] free functions plus the async
//! [`complete`]/[`open_stream`] wrappers over
//! [`HttpRuntime`]. The request/parse
//! mechanics are shared with the other OpenAI-compatible providers via
//! `openai::functions`' stage functions; this module is the source of truth for
//! Together AI's path, body assembly, and streaming dialect.

use serde::{Deserialize, Serialize};

use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_runtime::HttpRuntime;
use crate::providers::descriptor::ChatCompletionsDialect;
use crate::providers::descriptor::{
    ApiKeyLocation, ConfigError, ProviderDescriptor, required_env_var,
};
use crate::providers::openai::completion::CompletionModelOptions;
use crate::providers::openai::functions as openai_functions;

/// Default Together AI API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.together.xyz";

/// Together AI's Chat Completions streaming dialect.
pub(crate) const STREAM_DIALECT: ChatCompletionsDialect =
    ChatCompletionsDialect::from_descriptor(&DESCRIPTOR);

/// Together AI's capability sheet.
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    name: "together",
    supports_tools: true,
    supports_response_format: false,
    stream_include_usage: true,
    emits_complete_single_chunk_tool_calls: false,
    composes_native_output_with_tools: false,
    max_embedding_documents: Some(1024),
    verify_path: Some("/models"),
};

/// Plain-data Together AI provider configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Config {
    /// Reusable HTTP connection data.
    #[serde(flatten)]
    pub connection: crate::providers::HttpConnectionConfig,
    /// Model identifier requests are built for.
    pub model: String,
}

crate::providers::client::impl_http_connection_config!(Config);

impl Config {
    /// Config for `model` reading `TOGETHER_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            connection: crate::providers::HttpConnectionConfig::new(
                DEFAULT_BASE_URL.to_string(),
                ApiKeyLocation::Env("TOGETHER_API_KEY".to_string()),
            ),
            model: model.into(),
        }
    }

    /// Config for `model` built from the process environment.
    ///
    /// Reads `TOGETHER_API_KEY` (required) — the same variable the deleted
    /// `together::Client::from_env` read, which had no base-URL override. The
    /// credential is validated eagerly but stored as [`ApiKeyLocation::Env`], so
    /// the secret is read at request time rather than held inside the config.
    ///
    /// # Errors
    /// [`ConfigError`] when a required variable is missing or invalid.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
        let cfg = Self::new(model);
        required_env_var("TOGETHER_API_KEY")?;
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

/// Build the serialized chat-completions request body for `request`. Pure.
pub fn build_request_body(
    cfg: &Config,
    request: &CompletionRequest,
    stream: bool,
) -> Result<Vec<u8>, CompletionError> {
    build_body(
        &cfg.model,
        request,
        CompletionModelOptions::default(),
        stream,
    )
}

/// Together AI's straight-line chat-completions body assembly.
///
/// No wire-level quirks beyond the descriptor's
/// `supports_response_format: false` (Together's structured-output support is
/// model-dependent, so `output_schema` is dropped with a warning), so the body
/// is the shared typed conversion serialized as-is.
pub(crate) fn build_body(
    model: &str,
    request: &CompletionRequest,
    options: CompletionModelOptions,
    stream: bool,
) -> Result<Vec<u8>, CompletionError> {
    let typed = openai_functions::compatible_typed_request(model, request, &DESCRIPTOR, options)?;
    let body = openai_functions::compatible_body_value(&typed, &DESCRIPTOR, stream)?;
    Ok(serde_json::to_vec(&body)?)
}

/// The chat-completions request path for `model`.
///
/// The client base URL is the bare host; embeddings build their own v1 path.
pub(crate) fn completion_path(_model: &str) -> String {
    "/v1/chat/completions".to_string()
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
    openai_functions::compatible_http_request(
        &cfg.base_url,
        &completion_path(&cfg.model),
        &cfg.api_key,
        &cfg.extra_headers,
        build_request_body(cfg, request, stream)?,
    )
}

/// Parse a chat-completions response body into the normalized
/// [`completion::CompletionResponse`]. Pure.
pub fn parse_response(
    status: http::StatusCode,
    body: &str,
) -> Result<completion::CompletionResponse, CompletionError> {
    openai_functions::compatible_parse_response::<crate::providers::openai::CompletionResponse>(
        status,
        body,
        DESCRIPTOR.name,
    )
}

/// Open a streaming completion for `request`.
pub async fn open_stream(
    cfg: &Config,
    rt: &HttpRuntime,
    request: CompletionRequest,
) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError> {
    let req = build_request(cfg, &request, true)?;
    Ok(openai_functions::compatible_open_stream(
        rt,
        req,
        STREAM_DIALECT,
    ))
}

/// Send `request` to Together AI and return the normalized response.
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

/// Plain-data Together AI embeddings configuration.
///
/// A sibling of [`Config`]: embeddings target their own model identifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingConfig {
    /// Reusable HTTP connection data.
    #[serde(flatten)]
    pub connection: crate::providers::HttpConnectionConfig,
    /// Embedding model identifier requests are built for.
    pub model: String,
    /// Requested embedding dimensions, sent verbatim as `dimensions`.
    pub dimensions: Option<usize>,
}

crate::providers::client::impl_http_connection_config!(EmbeddingConfig);

impl EmbeddingConfig {
    /// Config for `model` reading `TOGETHER_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            connection: crate::providers::HttpConnectionConfig::new(
                DEFAULT_BASE_URL.to_string(),
                ApiKeyLocation::Env("TOGETHER_API_KEY".to_string()),
            ),
            model: model.into(),
            dimensions: None,
        }
    }

    /// Embedding config for `model` built from the process environment.
    ///
    /// Same variable as [`Config::from_env`]: `TOGETHER_API_KEY` (required).
    ///
    /// # Errors
    /// [`ConfigError`] when a required variable is missing or invalid.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
        let cfg = Self::new(model);
        required_env_var("TOGETHER_API_KEY")?;
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

/// Build the complete HTTP `/v1/embeddings` request for one chunk of `texts`.
///
/// Pure except for credential resolution. Together AI accepts neither the
/// OpenAI-compatible `encoding_format` nor `user` field, so neither is ever
/// sent.
pub fn build_embedding_request(
    cfg: &EmbeddingConfig,
    texts: &[String],
) -> Result<http::Request<Vec<u8>>, crate::embeddings::EmbeddingError> {
    use crate::embeddings::EmbeddingError;
    use http::header::{AUTHORIZATION, CONTENT_TYPE};

    let body = crate::providers::openai::embedding::build_embedding_body(
        &cfg.model,
        texts,
        cfg.dimensions
            .map(crate::providers::openai::embedding::EmbeddingDimensions::Dimensions),
        None,
        None,
    )?;
    let url = format!("{}/v1/embeddings", cfg.base_url.trim_end_matches('/'));
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

/// Parse a `/v1/embeddings` response into the normalized
/// [`crate::embeddings::EmbeddingResponse`]. Pure.
///
/// Together AI omits `usage` on some models, so a missing usage object is
/// tolerated and reported as zero.
pub fn parse_embedding_response(
    status: http::StatusCode,
    body: &str,
    documents: Vec<String>,
) -> Result<crate::embeddings::EmbeddingResponse, crate::embeddings::EmbeddingError> {
    crate::providers::openai::embedding::parse_embedding_response(
        status,
        body,
        documents,
        DESCRIPTOR.name,
        false,
    )
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

    mod embeddings {
        use super::super::{EmbeddingConfig, embed};
        use crate::http_runtime::HttpRuntime;
        use crate::providers::together::BGE_BASE_EN_V1_5;
        use crate::test_utils::RecordingHttpClient;

        const RESPONSE_BODY: &str = r#"{
            "object": "list",
            "model": "BAAI/bge-base-en-v1.5",
            "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3] }]
        }"#;

        /// Retargets the classic `together_embeddings_send_dimensions_to_v1_path`:
        /// dimensions ride the OpenAI-compatible `dimensions` field, the request
        /// goes to `/v1/embeddings`, and Together's usage-less payload reports
        /// zero rather than failing.
        #[tokio::test]
        async fn together_embeddings_send_dimensions_to_v1_path() {
            let http_client = RecordingHttpClient::new(RESPONSE_BODY);
            let rt = HttpRuntime::recording(http_client.clone());
            let cfg = EmbeddingConfig::new(BGE_BASE_EN_V1_5)
                .with_api_key("dummy-key")
                .with_dimensions(3);

            let response = embed(&cfg, &rt, vec!["hello".to_string()])
                .await
                .expect("embedding request should succeed");

            assert_eq!(response.usage.total_tokens, 0);
            let requests = http_client.requests();
            assert_eq!(requests.len(), 1);
            let request = requests.first().expect("one recorded request");
            assert!(request.uri.ends_with("/v1/embeddings"));
            let body: serde_json::Value =
                serde_json::from_slice(&request.body).expect("request body should be JSON");
            assert_eq!(body["dimensions"], serde_json::json!(3));
            assert_eq!(body["model"], BGE_BASE_EN_V1_5);
        }

        /// Retargets the classic `together_embeddings_omit_dimensions_when_unset`,
        /// extended to cover the two OpenAI fields Together rejects: with no
        /// dimension override the body carries neither `dimensions`,
        /// `encoding_format`, nor `user`.
        #[tokio::test]
        async fn together_omits_dimensions_encoding_format_and_user() {
            let http_client = RecordingHttpClient::new(RESPONSE_BODY);
            let rt = HttpRuntime::recording(http_client.clone());
            let cfg = EmbeddingConfig::new(BGE_BASE_EN_V1_5).with_api_key("dummy-key");

            embed(&cfg, &rt, vec!["hello".to_string()])
                .await
                .expect("embedding request should succeed");

            let requests = http_client.requests();
            let request = requests.first().expect("one recorded request");
            let body: serde_json::Value =
                serde_json::from_slice(&request.body).expect("request body should be JSON");
            assert!(body.get("dimensions").is_none());
            assert!(body.get("encoding_format").is_none());
            assert!(body.get("user").is_none());
        }

        /// Retargets the classic `together_error_envelope_preserves_raw_response`:
        /// an error envelope returned with a non-error status still surfaces the
        /// raw body and status.
        #[tokio::test]
        async fn together_error_envelope_preserves_raw_response() {
            let body = r#"{"error":{"message":"invalid model"},"code":"invalid_request"}"#;
            let http_client =
                RecordingHttpClient::with_error_response(http::StatusCode::ACCEPTED, body);
            let rt = HttpRuntime::recording(http_client);
            let cfg = EmbeddingConfig::new(BGE_BASE_EN_V1_5).with_api_key("dummy-key");

            let error = embed(&cfg, &rt, vec!["hello".to_string()])
                .await
                .expect_err("provider error envelope should fail");

            assert_eq!(error.provider_response_body(), Some(body));
            assert_eq!(
                error.provider_response_status(),
                Some(http::StatusCode::ACCEPTED)
            );
        }

        /// Retargets the classic `together_non_success_preserves_raw_response`.
        #[tokio::test]
        async fn together_non_success_preserves_raw_response() {
            let body = r#"{"error":{"message":"invalid api key"}}"#;
            let http_client =
                RecordingHttpClient::with_error_response(http::StatusCode::UNAUTHORIZED, body);
            let rt = HttpRuntime::recording(http_client);
            let cfg = EmbeddingConfig::new(BGE_BASE_EN_V1_5).with_api_key("dummy-key");

            let error = embed(&cfg, &rt, vec!["hello".to_string()])
                .await
                .expect_err("non-success response should fail");

            assert_eq!(error.provider_response_body(), Some(body));
            assert_eq!(
                error.provider_response_status(),
                Some(http::StatusCode::UNAUTHORIZED)
            );
        }
    }

    fn sample_request() -> CompletionRequest {
        CompletionRequest {
            model: None,
            chat_history: OneOrMany::one(crate::message::Message::user("hello")),
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
    fn build_request_sets_url_and_model() {
        let cfg = Config::new("test-model").with_api_key("secret");
        let req = build_request(&cfg, &sample_request(), false).expect("build");
        assert_eq!(req.uri(), "https://api.together.xyz/v1/chat/completions");
        let value: serde_json::Value = serde_json::from_slice(req.body()).expect("json");
        assert_eq!(value["model"], "test-model");
    }

    #[test]
    fn parse_response_normalizes() {
        let body = serde_json::json!({
            "id": "chatcmpl-1",
            "model": "test-model",
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
        assert_eq!(response.provider, "together");
        assert_eq!(response.usage.input_tokens, 3);
        assert_eq!(response.usage.total_tokens, 5);
    }
}
