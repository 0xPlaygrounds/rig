//! Mistral chat completions as config + pure functions.
//!
//! The data-oriented face of the Mistral provider, mirroring
//! [`crate::providers::openai::functions`]: a serde [`Config`], a
//! [`DESCRIPTOR`] capability sheet, and pure
//! [`build_request`]/[`parse_response`] free functions plus the async
//! [`complete`]/[`open_stream`] wrappers over
//! [`HttpRuntime`](crate::http_runtime::HttpRuntime). The request/parse
//! mechanics are shared with the other OpenAI-compatible providers via
//! `openai::functions`; this module supplies Mistral's own paths, wire
//! dialect, and provider name.

use serde::{Deserialize, Serialize};

use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_runtime::HttpRuntime;
use crate::providers::descriptor::{
    ApiKeyLocation, ConfigError, ProviderDescriptor, required_env_var,
};
use crate::providers::internal::openai_chat_completions_compatible::{
    ChatCompletionsDialect, ChatCompletionsUsageDialect,
};
use crate::providers::openai::completion::CompletionModelOptions;
use crate::providers::openai::embedding::EmbeddingDimensions;
use crate::providers::openai::functions as openai_functions;

/// Default Mistral API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.mistral.ai";

/// Mistral's Chat Completions streaming dialect: OpenAI-shaped chunks with
/// Mistral's own usage payload (cached-token fallbacks).
pub(crate) const STREAM_DIALECT: ChatCompletionsDialect =
    ChatCompletionsDialect::from_descriptor(&DESCRIPTOR)
        .with_usage(ChatCompletionsUsageDialect::Mistral);

/// Mistral's capability sheet.
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    name: "mistral",
    supports_tools: true,
    supports_response_format: true,
    stream_include_usage: false,
    emits_complete_single_chunk_tool_calls: true,
    composes_native_output_with_tools: true,
    max_embedding_documents: Some(1024),
};

/// Plain-data Mistral provider configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Config {
    /// API base URL (defaults to [`DEFAULT_BASE_URL`]).
    pub base_url: String,
    /// Credential location.
    pub api_key: ApiKeyLocation,
    /// Model identifier requests are built for.
    pub model: String,
    /// Extra headers attached to every request.
    pub extra_headers: Vec<(String, String)>,
}

impl Config {
    /// Config for `model` reading `MISTRAL_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: ApiKeyLocation::Env("MISTRAL_API_KEY".to_string()),
            model: model.into(),
            extra_headers: Vec::new(),
        }
    }

    /// Config for `model` built from the process environment.
    ///
    /// Reads `MISTRAL_API_KEY` (required) — the same variable the deleted
    /// `mistral::Client::from_env` read, which had no base-URL override. The
    /// credential is validated eagerly but stored as [`ApiKeyLocation::Env`], so
    /// the secret is read at request time rather than held inside the config.
    ///
    /// # Errors
    /// [`ConfigError`] when a required variable is missing or invalid.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
        let cfg = Self::new(model);
        required_env_var("MISTRAL_API_KEY")?;
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

/// The chat-completions request path for `model`.
///
/// The client base URL is the bare host; other Mistral capabilities
/// (embeddings, transcription, model listing) build their own v1 paths.
pub(crate) fn completion_path(_model: &str) -> String {
    "/v1/chat/completions".to_string()
}

/// Mistral's straight-line chat-completions body assembly.
///
/// Mistral's dialect quirks are all wire-level: the "must call some tool" mode is
/// spelled `any` rather than `required`, message `content` is a plain string
/// rather than an array of content parts, assistant messages carry the `prefix`
/// flag from Mistral's schema and must not echo hidden `reasoning_content` (the
/// API rejects unknown assistant fields). Mistral is also strict about unknown
/// parameters and reports usage on the final stream chunk without
/// `stream_options` — hence `stream_include_usage: false` on [`DESCRIPTOR`].
pub(crate) fn build_body(
    model: &str,
    request: &CompletionRequest,
    options: CompletionModelOptions,
    stream: bool,
) -> Result<Vec<u8>, CompletionError> {
    let typed = openai_functions::compatible_typed_request(model, request, &DESCRIPTOR, options)?;
    let mut body = openai_functions::compatible_body_value(&typed, &DESCRIPTOR, stream)?;
    apply_wire_dialect(&mut body);
    Ok(serde_json::to_vec(&body)?)
}

/// Rewrite a serialized chat-completions `body` into Mistral's wire dialect in
/// place: `any` tool choice, string message `content`, `prefix`-stamped
/// assistant turns, and no echoed `reasoning_content`.
pub(crate) fn apply_wire_dialect(body: &mut serde_json::Value) {
    if let Some(map) = body.as_object_mut() {
        // Mistral spells the "must call some tool" mode `any`, not `required`.
        if let Some(tool_choice) = map.get_mut("tool_choice")
            && tool_choice.as_str() == Some("required")
        {
            *tool_choice = serde_json::Value::String("any".to_string());
        }

        if let Some(messages) = map
            .get_mut("messages")
            .and_then(serde_json::Value::as_array_mut)
        {
            for message in messages {
                let Some(message) = message.as_object_mut() else {
                    continue;
                };
                let is_assistant =
                    message.get("role").and_then(serde_json::Value::as_str) == Some("assistant");

                // Mistral takes message `content` as a plain string.
                if let Some(content) = message.get_mut("content") {
                    crate::providers::openai::completion::flatten_text_content_parts(
                        content, "", false,
                    );
                }

                if is_assistant {
                    if !message.contains_key("content") {
                        message.insert(
                            "content".to_string(),
                            serde_json::Value::String(String::new()),
                        );
                    }
                    // `prefix` is part of Mistral's assistant message schema.
                    message
                        .entry("prefix")
                        .or_insert(serde_json::Value::Bool(false));
                    // Mistral rejects unknown assistant fields; hidden
                    // reasoning cannot be echoed back.
                    message.remove("reasoning_content");
                }
            }
        }
    }
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
    openai_functions::compatible_parse_response::<super::CompletionResponse>(
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

/// Send `request` to Mistral and return the normalized response.
/// Transcribe `request` with Mistral's `/v1/audio/transcriptions` endpoint.
pub async fn transcribe(
    cfg: &Config,
    rt: &HttpRuntime,
    request: crate::transcription::TranscriptionRequest,
) -> Result<
    crate::transcription::TranscriptionResponse<super::transcription::MistralTranscriptionResponse>,
    crate::transcription::TranscriptionError,
> {
    use crate::transcription::TranscriptionError;

    let form = super::transcription::build_transcription_form(&cfg.model, request)?;
    let url = format!(
        "{}/v1/audio/transcriptions",
        cfg.base_url.trim_end_matches('/')
    );
    let req = openai_functions::bearer_post(url, &cfg.api_key, &cfg.extra_headers, false)?
        .body(form)
        .map_err(|e| TranscriptionError::RequestError(Box::new(e)))?;
    let (status, body) = rt.send_multipart(req).await?;
    super::transcription::parse_transcription_response(status, &body)
}

pub async fn complete(
    cfg: &Config,
    rt: &HttpRuntime,
    request: CompletionRequest,
) -> Result<completion::CompletionResponse, CompletionError> {
    let req = build_request(cfg, &request, false)?;
    let (status, body) = rt.send(req).await?;
    parse_response(status, &body)
}

/// Build the `GET /v1/models` request for [`list_models`].
///
/// Pure except for credential resolution (`ApiKeyLocation::Env` reads the
/// environment).
pub fn build_list_models_request(
    cfg: &Config,
) -> Result<http::Request<Vec<u8>>, crate::model::ModelListingError> {
    let url = format!(
        "{}{}",
        cfg.base_url.trim_end_matches('/'),
        super::model_listing::LIST_MODELS_PATH
    );
    openai_functions::bearer_get(url, &cfg.api_key, &cfg.extra_headers)
}

/// Parse a `GET /v1/models` response body into a
/// [`ModelList`](crate::model::ModelList). Pure.
pub fn parse_list_models_response(
    status: http::StatusCode,
    body: &[u8],
) -> Result<crate::model::ModelList, crate::model::ModelListingError> {
    super::model_listing::parse_list_models_response(status, body)
}

/// List the models available to `cfg`'s credentials.
///
/// The classic `ModelListingClient` path parses through the same pure
/// [`parse_list_models_response`].
pub async fn list_models(
    cfg: &Config,
    rt: &HttpRuntime,
) -> Result<crate::model::ModelList, crate::model::ModelListingError> {
    let req = build_list_models_request(cfg)?;
    let (status, body) = rt.send_bytes(req).await?;
    parse_list_models_response(status, &body)
}

// ================================================================
// Embeddings
// ================================================================

/// Plain-data Mistral embeddings configuration.
///
/// A sibling of [`Config`]: embeddings target their own model identifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingConfig {
    /// API base URL (defaults to [`DEFAULT_BASE_URL`]).
    pub base_url: String,
    /// Credential location.
    pub api_key: ApiKeyLocation,
    /// Embedding model identifier requests are built for.
    pub model: String,
    /// Requested embedding dimensions.
    ///
    /// Mistral only accepts a dimension override on the Codestral Embed
    /// models, where it is sent as `output_dimension`; see
    /// [`embedding_dimensions`].
    pub dimensions: Option<usize>,
    /// Extra headers attached to every request.
    pub extra_headers: Vec<(String, String)>,
}

impl EmbeddingConfig {
    /// Config for `model` reading `MISTRAL_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: ApiKeyLocation::Env("MISTRAL_API_KEY".to_string()),
            model: model.into(),
            dimensions: None,
            extra_headers: Vec::new(),
        }
    }

    /// Embedding config for `model` built from the process environment.
    ///
    /// Same variable as [`Config::from_env`]: `MISTRAL_API_KEY` (required).
    ///
    /// # Errors
    /// [`ConfigError`] when a required variable is missing or invalid.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
        let cfg = Self::new(model);
        required_env_var("MISTRAL_API_KEY")?;
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

    /// Request `dimensions`-sized embeddings (Codestral Embed only).
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }
}

/// Validate and place Mistral's dimension override. Pure.
///
/// Only the Codestral Embed models accept a dimension override, and they cap
/// it at 3072; the value rides Mistral's own `output_dimension` field rather
/// than the OpenAI-compatible `dimensions`.
///
/// # Errors
/// `EmbeddingError::UnsupportedParameter` for a fixed-size model, and
/// `EmbeddingError::InvalidParameterValue` above the 3072 cap.
pub fn embedding_dimensions(
    model: &str,
    dimensions: Option<usize>,
) -> Result<Option<EmbeddingDimensions>, crate::embeddings::EmbeddingError> {
    use crate::embeddings::EmbeddingError;

    let Some(dimensions) = dimensions else {
        return Ok(None);
    };

    if !matches!(model, "codestral-embed" | "codestral-embed-2505") {
        return Err(EmbeddingError::UnsupportedParameter {
            provider: DESCRIPTOR.name,
            parameter: "dimensions",
        });
    }

    if dimensions > 3_072 {
        return Err(EmbeddingError::InvalidParameterValue {
            provider: DESCRIPTOR.name,
            parameter: "dimensions",
            requirement: "to be at most 3072 for Codestral Embed",
        });
    }

    Ok(Some(EmbeddingDimensions::OutputDimension(dimensions)))
}

/// Build the complete HTTP `/v1/embeddings` request for one chunk of `texts`.
///
/// Pure except for credential resolution. Mistral rejects the OpenAI-compatible
/// `user` field, so it is never sent.
pub fn build_embedding_request(
    cfg: &EmbeddingConfig,
    texts: &[String],
) -> Result<http::Request<Vec<u8>>, crate::embeddings::EmbeddingError> {
    use crate::embeddings::EmbeddingError;
    use http::header::{AUTHORIZATION, CONTENT_TYPE};

    let body = crate::providers::openai::embedding::build_embedding_body(
        &cfg.model,
        texts,
        embedding_dimensions(&cfg.model, cfg.dimensions)?,
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
        true,
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

#[cfg(test)]
mod tests {
    use super::*;

    mod embeddings {
        use super::super::{EmbeddingConfig, embed};
        use crate::embeddings::EmbeddingError;
        use crate::http_runtime::HttpRuntime;
        use crate::providers::mistral::{CODESTRAL_EMBED, MISTRAL_EMBED};
        use crate::test_utils::RecordingHttpClient;

        const RESPONSE_BODY: &str = r#"{
            "id": "emb-1",
            "object": "list",
            "model": "mistral-embed",
            "usage": { "prompt_tokens": 5, "total_tokens": 5 },
            "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3] }]
        }"#;

        /// Retargets the classic
        /// `codestral_embeddings_map_dimensions_and_mistral_usage`: Codestral's
        /// dimension override rides `output_dimension` (never `dimensions`),
        /// the request goes to `/v1/embeddings`, `user` is never sent, and
        /// Mistral's usage object is surfaced.
        #[tokio::test]
        async fn codestral_embeddings_map_dimensions_and_mistral_usage() {
            let http_client = RecordingHttpClient::new(RESPONSE_BODY);
            let rt = HttpRuntime::recording(http_client.clone());
            let cfg = EmbeddingConfig::new(CODESTRAL_EMBED)
                .with_api_key("dummy-key")
                .with_dimensions(512);

            let response = embed(&cfg, &rt, vec!["hello".to_string()])
                .await
                .expect("embedding request should succeed");

            assert_eq!(
                response.embeddings.first().map(|e| e.vec.clone()),
                Some(vec![0.1, 0.2, 0.3])
            );
            assert_eq!(response.usage.input_tokens, 5);
            assert_eq!(response.usage.total_tokens, 5);

            let requests = http_client.requests();
            assert_eq!(requests.len(), 1);
            let request = requests.first().expect("one recorded request");
            assert!(request.uri.ends_with("/v1/embeddings"));
            let body: serde_json::Value =
                serde_json::from_slice(&request.body).expect("request body should be JSON");
            assert_eq!(body["output_dimension"], serde_json::json!(512));
            assert!(body.get("dimensions").is_none());
            assert!(body.get("user").is_none());
            assert!(body.get("encoding_format").is_none());
        }

        /// Retargets the classic `mistral_embed_rejects_dimensions_before_sending`:
        /// a fixed-size model rejects a dimension override without an HTTP call.
        #[tokio::test]
        async fn mistral_embed_rejects_dimensions_before_sending() {
            let http_client = RecordingHttpClient::new(RESPONSE_BODY);
            let rt = HttpRuntime::recording(http_client.clone());
            let cfg = EmbeddingConfig::new(MISTRAL_EMBED)
                .with_api_key("dummy-key")
                .with_dimensions(512);

            let error = embed(&cfg, &rt, vec!["hello".to_string()])
                .await
                .expect_err("fixed-size model should reject dimensions");

            assert!(matches!(
                error,
                EmbeddingError::UnsupportedParameter {
                    provider: "mistral",
                    parameter: "dimensions"
                }
            ));
            assert!(http_client.requests().is_empty());
        }

        /// Retargets the classic
        /// `codestral_embed_rejects_dimensions_above_maximum_before_sending`:
        /// the 3072 cap is enforced before any HTTP call.
        #[tokio::test]
        async fn codestral_embed_rejects_dimensions_above_maximum_before_sending() {
            let http_client = RecordingHttpClient::new(RESPONSE_BODY);
            let rt = HttpRuntime::recording(http_client.clone());
            let cfg = EmbeddingConfig::new(CODESTRAL_EMBED)
                .with_api_key("dummy-key")
                .with_dimensions(3_073);

            let error = embed(&cfg, &rt, vec!["hello".to_string()])
                .await
                .expect_err("out-of-range dimensions should fail");

            assert!(matches!(
                error,
                EmbeddingError::InvalidParameterValue {
                    provider: "mistral",
                    parameter: "dimensions",
                    ..
                }
            ));
            assert!(http_client.requests().is_empty());
        }
    }

    #[test]
    fn build_list_models_request_sets_url_and_bearer_auth() {
        let cfg = Config::new("test-model").with_api_key("secret");
        let req = build_list_models_request(&cfg).expect("build");
        assert_eq!(req.method(), http::Method::GET);
        assert_eq!(req.uri(), "https://api.mistral.ai/v1/models");
        assert_eq!(
            req.headers()
                .get(http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok()),
            Some("Bearer secret")
        );
    }
    use crate::OneOrMany;

    fn sample_request() -> CompletionRequest {
        CompletionRequest {
            model: None,
            preamble: None,
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
        assert_eq!(req.uri(), "https://api.mistral.ai/v1/chat/completions");
        let value: serde_json::Value = serde_json::from_slice(req.body()).expect("json");
        assert_eq!(value["model"], "test-model");
    }

    #[test]
    fn parse_response_normalizes() {
        let body = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
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
        assert_eq!(response.provider, "mistral");
        assert_eq!(response.usage.input_tokens, 3);
        assert_eq!(response.usage.total_tokens, 5);
    }
}
