//! OpenRouter chat completions as config + pure functions.
//!
//! The data-oriented face of the OpenRouter provider, mirroring
//! [`crate::providers::openai::functions`]: a serde [`Config`], a
//! [`DESCRIPTOR`] capability sheet, and pure
//! [`build_request`]/[`parse_response`] free functions plus the async
//! [`complete`]/[`open_stream`] wrappers over
//! [`HttpRuntime`](crate::http_runtime::HttpRuntime). The request/parse
//! mechanics reuse the shared OpenAI-compatible stages in
//! `openai::functions`, but OpenRouter has its own typed request
//! (`OpenrouterCompletionRequest`)
//! and its own body finalization (provider routing preferences and prompt
//! caching), both applied by `build_body`.

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
use crate::providers::openai::embedding::EncodingFormat;
use crate::providers::openai::functions as openai_functions;

/// Default OpenRouter API base URL.
pub const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// OpenRouter's Chat Completions streaming dialect.
///
/// OpenRouter never receives `stream_options.include_usage` (its usage rides
/// the final chunk under its own [`Usage`](super::completion::Usage) shape), and
/// its `reasoning_details` deltas decorate accumulated tool calls.
pub(crate) const STREAM_DIALECT: ChatCompletionsDialect =
    ChatCompletionsDialect::from_descriptor(&DESCRIPTOR)
        .with_usage(ChatCompletionsUsageDialect::OpenRouter)
        .with_reasoning_detail_decoration();

/// OpenRouter's capability sheet.
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    name: "openrouter",
    supports_tools: true,
    supports_response_format: true,
    stream_include_usage: false,
    emits_complete_single_chunk_tool_calls: false,
    composes_native_output_with_tools: true,
    max_embedding_documents: Some(1024),
    verify_path: Some("/key"),
};

/// Plain-data OpenRouter provider configuration.
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
    /// Config for `model` reading `OPENROUTER_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: ApiKeyLocation::Env("OPENROUTER_API_KEY".to_string()),
            model: model.into(),
            extra_headers: Vec::new(),
        }
    }

    /// Config for `model` built from the process environment.
    ///
    /// Reads `OPENROUTER_API_KEY` (required) — the same variable the deleted
    /// `openrouter::Client::from_env` read, which had no base-URL override. The
    /// credential is validated eagerly but stored as [`ApiKeyLocation::Env`], so
    /// the secret is read at request time rather than held inside the config.
    ///
    /// # Errors
    /// [`ConfigError`] when a required variable is missing or invalid.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
        let cfg = Self::new(model);
        required_env_var("OPENROUTER_API_KEY")?;
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

/// The chat-completions request path for `model`.
pub(crate) fn completion_path(_model: &str) -> String {
    "/chat/completions".to_string()
}

/// OpenRouter's chat-completions body assembly.
///
/// Two dialect steps over OpenAI's: the typed request is OpenRouter's own
/// (provider routing preferences, reasoning details), and the serialized body
/// is finalized with OpenRouter's routing/prompt-caching fields.
pub(crate) fn build_body(
    model: &str,
    request: &CompletionRequest,
    options: CompletionModelOptions,
    stream: bool,
) -> Result<Vec<u8>, CompletionError> {
    let typed = super::completion::OpenrouterCompletionRequest::try_from(
        super::completion::OpenRouterRequestParams {
            model,
            request: request.clone(),
            strict_tools: options.strict_tools,
        },
    )?;
    let mut body = openai_functions::compatible_body_value(&typed, &DESCRIPTOR, stream)?;
    super::completion::finalize_openrouter_request_body(&mut body, options.prompt_caching);
    Ok(serde_json::to_vec(&body)?)
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
    openai_functions::compatible_parse_response::<super::completion::CompletionResponse>(
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

/// Transcribe `request` with OpenRouter's `/audio/transcriptions` endpoint
/// (JSON body carrying base64 audio).
pub async fn transcribe(
    cfg: &Config,
    rt: &HttpRuntime,
    request: crate::transcription::TranscriptionRequest,
) -> Result<
    crate::transcription::TranscriptionResponse<super::transcription::TranscriptionResponse>,
    crate::transcription::TranscriptionError,
> {
    use crate::transcription::TranscriptionError;

    let body = super::transcription::build_transcription_body(&cfg.model, request)?;
    let url = format!(
        "{}/audio/transcriptions",
        cfg.base_url.trim_end_matches('/')
    );
    let req = openai_functions::bearer_post(url, &cfg.api_key, &cfg.extra_headers, true)?
        .body(body)
        .map_err(|e| TranscriptionError::RequestError(Box::new(e)))?;
    let (status, body) = rt.send_bytes(req).await?;
    super::transcription::parse_transcription_response(status, &body)
}

/// Generate speech with OpenRouter's `/audio/speech` endpoint.
#[cfg(feature = "audio")]
pub async fn generate_audio(
    cfg: &Config,
    rt: &HttpRuntime,
    request: crate::audio_generation::AudioGenerationRequest,
) -> Result<
    crate::audio_generation::AudioGenerationResponse<bytes::Bytes>,
    crate::audio_generation::AudioGenerationError,
> {
    use crate::audio_generation::AudioGenerationError;

    let body = super::audio_generation::build_audio_generation_body(&cfg.model, &request)?;
    let url = format!("{}/audio/speech", cfg.base_url.trim_end_matches('/'));
    let req = openai_functions::bearer_post(url, &cfg.api_key, &cfg.extra_headers, true)?
        .body(body)
        .map_err(|e| AudioGenerationError::RequestError(Box::new(e)))?;
    let (status, body) = rt.send_bytes(req).await?;
    super::audio_generation::parse_audio_generation_response(status, body)
}

/// Send `request` to OpenRouter and return the normalized response.
pub async fn complete(
    cfg: &Config,
    rt: &HttpRuntime,
    request: CompletionRequest,
) -> Result<completion::CompletionResponse, CompletionError> {
    let req = build_request(cfg, &request, false)?;
    let (status, body) = rt.send(req).await?;
    parse_response(status, &body)
}

/// Build the `GET /models` request for [`list_models`].
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

/// Parse a `GET /models` response body into a
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

/// Plain-data OpenRouter embeddings configuration.
///
/// A sibling of [`Config`]: embeddings target their own model identifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingConfig {
    /// API base URL (defaults to [`DEFAULT_BASE_URL`], including `/api/v1`).
    pub base_url: String,
    /// Credential location.
    pub api_key: ApiKeyLocation,
    /// Embedding model identifier requests are built for.
    pub model: String,
    /// Requested embedding dimensions, sent verbatim as `dimensions`.
    pub dimensions: Option<usize>,
    /// Requested response encoding, sent as `encoding_format`.
    ///
    /// OpenRouter's embeddings API accepts it (the deleted
    /// `OpenAIEmbeddingsCompatible` default was
    /// `SUPPORTS_ENCODING_FORMAT = true`).
    /// [`EncodingFormat::Base64`] is rejected at request-build time, as the
    /// deleted model did: Rig's parser only decodes float vectors.
    pub encoding_format: Option<EncodingFormat>,
    /// Opaque end-user identifier, sent as `user`.
    ///
    /// OpenRouter accepts it (the deleted default was
    /// `SUPPORTS_USER = true`).
    pub user: Option<String>,
    /// Extra headers attached to every request.
    pub extra_headers: Vec<(String, String)>,
}

impl EmbeddingConfig {
    /// Config for `model` reading `OPENROUTER_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: ApiKeyLocation::Env("OPENROUTER_API_KEY".to_string()),
            model: model.into(),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra_headers: Vec::new(),
        }
    }

    /// Embedding config for `model` built from the process environment.
    ///
    /// Same variable as [`Config::from_env`]: `OPENROUTER_API_KEY` (required).
    ///
    /// # Errors
    /// [`ConfigError`] when a required variable is missing or invalid.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
        let cfg = Self::new(model);
        required_env_var("OPENROUTER_API_KEY")?;
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

    /// Request a wire encoding for the returned vectors.
    ///
    /// [`EncodingFormat::Base64`] is accepted here but rejected when the
    /// request is built — Rig cannot decode base64 vectors, and the deleted
    /// model raised the same error.
    pub fn with_encoding_format(mut self, encoding_format: EncodingFormat) -> Self {
        self.encoding_format = Some(encoding_format);
        self
    }

    /// Attach an opaque end-user identifier to embedding requests.
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }
}

/// Build the complete HTTP `/embeddings` request for one chunk of `texts`.
///
/// Pure except for credential resolution. The path is resolved against
/// [`DEFAULT_BASE_URL`], which already carries OpenRouter's `/api/v1` prefix.
pub fn build_embedding_request(
    cfg: &EmbeddingConfig,
    texts: &[String],
) -> Result<http::Request<Vec<u8>>, crate::embeddings::EmbeddingError> {
    use crate::embeddings::EmbeddingError;
    use http::header::{AUTHORIZATION, CONTENT_TYPE};

    // The guard the deleted `GenericEmbeddingModel::embed_texts_with_usage`
    // applied before sending: Rig's response parser reads float vectors, so a
    // base64 request would produce embeddings it cannot decode.
    if cfg.encoding_format == Some(EncodingFormat::Base64) {
        return Err(EmbeddingError::UnsupportedResponseEncoding {
            provider: DESCRIPTOR.name,
            encoding_format: "base64",
        });
    }

    let body = crate::providers::openai::embedding::build_embedding_body(
        &cfg.model,
        texts,
        cfg.dimensions
            .map(crate::providers::openai::embedding::EmbeddingDimensions::Dimensions),
        cfg.encoding_format,
        cfg.user.as_deref(),
    )?;
    let url = format!("{}/embeddings", cfg.base_url.trim_end_matches('/'));
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

/// Parse an `/embeddings` response into the normalized
/// [`crate::embeddings::EmbeddingResponse`]. Pure.
///
/// OpenRouter's embeddings payloads may omit `usage`, so a missing usage
/// object is tolerated and reported as zero.
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
/// endpoint is [`DESCRIPTOR`]'s `verify_path` (`/key`, the value the
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

    mod embeddings {
        use super::super::{EmbeddingConfig, embed};
        use crate::embeddings::EmbeddingError;
        use crate::http_runtime::HttpRuntime;
        use crate::test_utils::RecordingHttpClient;

        const MODEL: &str = "openai/text-embedding-3-small";

        const RESPONSE_BODY: &str = r#"{
            "id": "gen-1",
            "object": "list",
            "model": "openai/text-embedding-3-small",
            "data": [{ "object": "embedding", "index": 0, "embedding": [0.5, 0.6] }]
        }"#;

        /// Retargets the classic
        /// `openrouter_embeddings_preserve_supported_parameters_and_zero_absent_usage`:
        /// `dimensions` rides the OpenAI-compatible field, the URL keeps
        /// OpenRouter's `/api/v1` prefix, and a payload without `usage` reports
        /// zero rather than failing.
        #[tokio::test]
        async fn openrouter_embeddings_send_dimensions_and_zero_absent_usage() {
            let http_client = RecordingHttpClient::new(RESPONSE_BODY);
            let rt = HttpRuntime::recording(http_client.clone());
            let cfg = EmbeddingConfig::new(MODEL)
                .with_api_key("dummy-key")
                .with_dimensions(2);

            let response = embed(&cfg, &rt, vec!["hello".to_string()])
                .await
                .expect("embedding request should succeed");

            assert_eq!(response.embeddings.len(), 1);
            assert_eq!(response.usage.total_tokens, 0);
            let requests = http_client.requests();
            let request = requests.first().expect("one recorded request");
            assert_eq!(request.uri, "https://openrouter.ai/api/v1/embeddings");
            let body: serde_json::Value =
                serde_json::from_slice(&request.body).expect("request body should be JSON");
            assert_eq!(body["dimensions"], serde_json::json!(2));
        }

        /// Retargets the classic `openrouter_rejects_response_length_mismatch`:
        /// a response with fewer vectors than inputs is a parse error.
        #[tokio::test]
        async fn openrouter_rejects_response_length_mismatch() {
            let http_client = RecordingHttpClient::new(RESPONSE_BODY);
            let rt = HttpRuntime::recording(http_client);
            let cfg = EmbeddingConfig::new(MODEL).with_api_key("dummy-key");

            let error = embed(&cfg, &rt, vec!["one".to_string(), "two".to_string()])
                .await
                .expect_err("response length mismatch should fail");

            assert!(matches!(error, EmbeddingError::ResponseError(_)));
        }
    }

    #[test]
    fn build_list_models_request_sets_url_and_bearer_auth() {
        let cfg = Config::new("test-model").with_api_key("secret");
        let req = build_list_models_request(&cfg).expect("build");
        assert_eq!(req.method(), http::Method::GET);
        assert!(req.uri().to_string().ends_with("/models"));
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
        assert_eq!(req.uri(), "https://openrouter.ai/api/v1/chat/completions");
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
            "choices": [{
                "index": 0,
                "native_finish_reason": null,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop"
            }],
            "system_fingerprint": null,
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        })
        .to_string();
        let response = parse_response(http::StatusCode::OK, &body).expect("parse");
        assert_eq!(response.provider, "openrouter");
        assert_eq!(response.usage.input_tokens, 3);
        assert_eq!(response.usage.total_tokens, 5);
    }
}
#[cfg(test)]
mod embedding_parameter_tests {
    use super::*;

    /// The deleted `GenericEmbeddingModel` exposed `encoding_format` and
    /// `user` builders and OpenRouter accepted both
    /// (`SUPPORTS_ENCODING_FORMAT` / `SUPPORTS_USER` defaulted to `true`);
    /// `EmbeddingConfig` had lost the two knobs entirely.
    #[test]
    fn embedding_body_carries_encoding_format_and_user() {
        let cfg = EmbeddingConfig::new("text-embedding-3-small")
            .with_api_key("secret")
            .with_dimensions(1_536)
            .with_encoding_format(EncodingFormat::Float)
            .with_user("user-123");

        let req = build_embedding_request(&cfg, &["hello".to_string()]).expect("build");
        let value: serde_json::Value = serde_json::from_slice(req.body()).expect("json");

        assert_eq!(value["model"], "text-embedding-3-small");
        assert_eq!(value["input"], serde_json::json!(["hello"]));
        assert_eq!(value["dimensions"], serde_json::json!(1_536));
        assert_eq!(value["encoding_format"], serde_json::json!("float"));
        assert_eq!(value["user"], serde_json::json!("user-123"));
    }

    /// Unset knobs stay off the wire, so the default body is byte-identical to
    /// the one this face produced before the fields existed.
    #[test]
    fn embedding_body_omits_unset_parameters() {
        let cfg = EmbeddingConfig::new("text-embedding-3-small").with_api_key("secret");
        let req = build_embedding_request(&cfg, &["hello".to_string()]).expect("build");
        let value: serde_json::Value = serde_json::from_slice(req.body()).expect("json");

        assert!(value.get("encoding_format").is_none());
        assert!(value.get("user").is_none());
        assert!(value.get("dimensions").is_none());
    }

    /// Rig's response parser reads float vectors, so a base64 request would
    /// yield embeddings it cannot decode. The deleted model rejected it with
    /// `UnsupportedResponseEncoding`; so does this one, before sending.
    #[test]
    fn base64_encoding_format_is_rejected_before_sending() {
        let cfg = EmbeddingConfig::new("text-embedding-3-small")
            .with_api_key("secret")
            .with_encoding_format(EncodingFormat::Base64);

        let error = build_embedding_request(&cfg, &["hello".to_string()])
            .expect_err("base64 must be rejected");
        assert!(matches!(
            error,
            crate::embeddings::EmbeddingError::UnsupportedResponseEncoding {
                provider: "openrouter",
                encoding_format: "base64",
            }
        ));
    }
}
