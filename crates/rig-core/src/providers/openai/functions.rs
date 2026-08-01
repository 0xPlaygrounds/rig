//! OpenAI chat completions as config + pure functions (provider pilot).
//!
//! This module is the data-oriented face of the OpenAI provider: a serde
//! [`Config`], a [`DESCRIPTOR`] capability sheet, and free functions that
//! decompose a completion into its pure parts —
//! [`build_request`] (data → HTTP request, no IO) and [`parse_response`]
//! (bytes → normalized [`completion::CompletionResponse`], no IO) — plus the
//! async [`complete`] wrapper over [`HttpRuntime`].
//!
//! The typed conversion these functions delegate to
//! ([`super::completion::CompletionRequest`]`::try_from(`[`OpenAIRequestParams`](super::completion::OpenAIRequestParams)`)`)
//! is shared with every OpenAI-compatible provider through the
//! `compatible_*` stages at the bottom of this module.

use http::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::api::ApiResponse;
use super::completion::CompletionModelOptions;
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::embeddings::{self, EmbeddingError};
use crate::http_runtime::HttpRuntime;
use crate::json_utils::merge;
use crate::model::{ModelList, ModelListingError};
use crate::providers::descriptor::ChatCompletionsDialect;
use crate::providers::descriptor::{
    ApiKeyLocation, ConfigError, ProviderDescriptor, optional_env_var, required_env_var,
};

/// Default OpenAI API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// OpenAI's Chat Completions streaming dialect.
pub(crate) const STREAM_DIALECT: ChatCompletionsDialect =
    ChatCompletionsDialect::from_descriptor(&DESCRIPTOR);

/// OpenAI's capability sheet.
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    name: "openai",
    supports_tools: true,
    supports_response_format: true,
    stream_include_usage: true,
    emits_complete_single_chunk_tool_calls: false,
    composes_native_output_with_tools: true,
    max_embedding_documents: Some(1024),
    verify_path: Some("/models"),
};

/// Plain-data OpenAI provider configuration.
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
    /// Config for `model` reading `OPENAI_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            connection: crate::providers::HttpConnectionConfig::new(
                DEFAULT_BASE_URL.to_string(),
                ApiKeyLocation::Env("OPENAI_API_KEY".to_string()),
            ),
            model: model.into(),
        }
    }

    /// Config for `model` built from the process environment.
    ///
    /// Reads `OPENAI_API_KEY` (required) and `OPENAI_BASE_URL` (optional
    /// override of [`DEFAULT_BASE_URL`]). The credential is validated eagerly
    /// but stored as [`ApiKeyLocation::Env`], so the secret is read at request
    /// time rather than held inside the config.
    ///
    /// # Errors
    /// [`ConfigError`] when a required variable is missing or invalid.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
        let mut cfg = Self::new(model);
        required_env_var("OPENAI_API_KEY")?;
        if let Some(base_url) = optional_env_var("OPENAI_BASE_URL")? {
            cfg.base_url = base_url;
        }
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

/// Build the serialized chat-completions request body for `request`.
///
/// Pure: the exact bytes the wire sees, unit-testable against recorded
/// cassette request bodies. `stream` adds the streaming parameters
/// (`stream: true` and, per [`DESCRIPTOR`], `stream_options.include_usage`).
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

/// OpenAI's straight-line chat-completions body assembly.
///
/// OpenAI is the reference dialect: no wire-level quirks, so the body is the
/// shared typed conversion serialized as-is. Compatible providers add their own
/// dialect steps between the two shared stages (see e.g.
/// [`crate::providers::mistral::functions::build_body`]).
pub(crate) fn build_body(
    model: &str,
    request: &CompletionRequest,
    options: CompletionModelOptions,
    stream: bool,
) -> Result<Vec<u8>, CompletionError> {
    let typed = compatible_typed_request(model, request, &DESCRIPTOR, options)?;
    let body = compatible_body_value(&typed, &DESCRIPTOR, stream)?;
    Ok(serde_json::to_vec(&body)?)
}

/// The chat-completions request path for `model`.
pub(crate) fn completion_path(_model: &str) -> String {
    "/chat/completions".to_string()
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
    compatible_http_request(
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
    compatible_parse_response::<super::completion::CompletionResponse>(
        status,
        body,
        DESCRIPTOR.name,
    )
}

/// Open a streaming completion for `request`.
///
/// Returns the concrete [`crate::streaming::StreamingCompletionResponse`];
/// items terminate with the normalized
/// [`StreamFinal`](crate::streaming::StreamFinal).
pub async fn open_stream(
    cfg: &Config,
    rt: &HttpRuntime,
    request: CompletionRequest,
) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError> {
    let req = build_request(cfg, &request, true)?;
    Ok(compatible_open_stream(rt, req, STREAM_DIALECT))
}

/// Send `request` to OpenAI and return the normalized response.
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
// Model listing
// ================================================================

/// A GET request with Bearer auth and extra headers, shared by the
/// model-listing free functions of Bearer-authenticated providers.
pub(crate) fn bearer_get(
    url: String,
    api_key: &ApiKeyLocation,
    extra_headers: &[(String, String)],
) -> Result<http::Request<Vec<u8>>, ModelListingError> {
    // Classic parity: the deleted reqwest-backed client stamped `accept` and
    // `content-type` on every request, including bodyless GETs, and the
    // recorded model-listing exchanges match on them.
    let mut builder = http::Request::get(url)
        .header(http::header::ACCEPT, "*/*")
        .header(CONTENT_TYPE, "application/json");
    if let Some(key) = api_key
        .resolve()
        .map_err(|e| ModelListingError::request_error(e.to_string()))?
    {
        builder = builder.header(AUTHORIZATION, format!("Bearer {key}"));
    }
    for (name, value) in extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    builder
        .body(Vec::new())
        .map_err(|e| ModelListingError::request_error(e.to_string()))
}

/// Build the `GET /models` request for [`list_models`].
///
/// Pure except for credential resolution (`ApiKeyLocation::Env` reads the
/// environment).
pub fn build_list_models_request(
    cfg: &Config,
) -> Result<http::Request<Vec<u8>>, ModelListingError> {
    let url = format!(
        "{}{}",
        cfg.base_url.trim_end_matches('/'),
        super::model_listing::LIST_MODELS_PATH
    );
    bearer_get(url, &cfg.api_key, &cfg.extra_headers)
}

/// Parse a `GET /models` response body into a [`ModelList`]. Pure.
pub fn parse_list_models_response(
    status: http::StatusCode,
    body: &[u8],
) -> Result<ModelList, ModelListingError> {
    super::model_listing::parse_list_models_response(status, body)
}

/// List the models available to `cfg`'s credentials.
pub async fn list_models(cfg: &Config, rt: &HttpRuntime) -> Result<ModelList, ModelListingError> {
    let req = build_list_models_request(cfg)?;
    let (status, body) = rt.send_bytes(req).await?;
    parse_list_models_response(status, &body)
}

// ================================================================
// Minor modalities: transcription, image generation, audio generation
// ================================================================

/// A POST request builder with Bearer auth and optional JSON content type,
/// shared by the modality free functions of Bearer-authenticated providers.
pub(crate) fn bearer_post(
    url: String,
    api_key: &ApiKeyLocation,
    extra_headers: &[(String, String)],
    json_content_type: bool,
) -> Result<http::request::Builder, crate::http_client::Error> {
    let mut builder = http::Request::post(url);
    if json_content_type {
        builder = builder.header(CONTENT_TYPE, "application/json");
    }
    if let Some(key) = api_key
        .resolve()
        .map_err(|e| crate::http_client::Error::Instance(e.into()))?
    {
        builder = builder.header(AUTHORIZATION, format!("Bearer {key}"));
    }
    for (name, value) in extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    Ok(builder)
}

/// Build the multipart form for a transcription `request`. Pure.
pub fn build_transcription_form(
    model: &str,
    request: crate::transcription::TranscriptionRequest,
) -> Result<crate::http_client::MultipartForm, crate::transcription::TranscriptionError> {
    use crate::http_client::{MultipartForm, multipart::Part};
    use crate::transcription::TranscriptionError;

    let mut body = MultipartForm::new()
        .text("model", model.to_string())
        .part(Part::bytes("file", request.data).filename(request.filename));

    if let Some(language) = request.language {
        body = body.text("language", language);
    }
    if let Some(prompt) = request.prompt {
        body = body.text("prompt", prompt);
    }
    if let Some(ref temperature) = request.temperature {
        body = body.text("temperature", temperature.to_string());
    }
    if let Some(ref additional_params) = request.additional_params {
        let params = additional_params.as_object().ok_or_else(|| {
            TranscriptionError::RequestError(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "additional transcription parameters must be a JSON object",
            )))
        })?;
        for (key, value) in params {
            body = body.text(key.to_owned(), value.to_string());
        }
    }
    Ok(body)
}

/// Parse a transcription response body. Pure.
pub fn parse_transcription_response(
    status: http::StatusCode,
    body: &[u8],
) -> Result<
    crate::transcription::TranscriptionResponse<super::transcription::TranscriptionResponse>,
    crate::transcription::TranscriptionError,
> {
    use crate::transcription::TranscriptionError;

    if !status.is_success() {
        return Err(TranscriptionError::from_http_response(
            status,
            String::from_utf8_lossy(body).into_owned(),
        ));
    }
    match serde_json::from_slice::<ApiResponse<super::transcription::TranscriptionResponse>>(body)?
    {
        ApiResponse::Ok(response) => response.try_into(),
        ApiResponse::Err(err) => {
            tracing::warn!(message = %err.message, "provider returned an error response");
            Err(TranscriptionError::from_http_response(
                status,
                String::from_utf8_lossy(body).into_owned(),
            ))
        }
    }
}

/// Transcribe `request` with OpenAI's `/audio/transcriptions` endpoint.
pub async fn transcribe(
    cfg: &Config,
    rt: &HttpRuntime,
    request: crate::transcription::TranscriptionRequest,
) -> Result<
    crate::transcription::TranscriptionResponse<super::transcription::TranscriptionResponse>,
    crate::transcription::TranscriptionError,
> {
    use crate::transcription::TranscriptionError;

    let form = build_transcription_form(&cfg.model, request)?;
    let url = format!(
        "{}/audio/transcriptions",
        cfg.base_url.trim_end_matches('/')
    );
    let mut builder = http::Request::post(url);
    if let Some(key) = cfg
        .api_key
        .resolve()
        .map_err(|e| TranscriptionError::RequestError(Box::new(e)))?
    {
        builder = builder.header(AUTHORIZATION, format!("Bearer {key}"));
    }
    for (name, value) in &cfg.extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    let req = builder
        .body(form)
        .map_err(|e| TranscriptionError::RequestError(Box::new(e)))?;
    let (status, body) = rt.send_multipart(req).await?;
    parse_transcription_response(status, &body)
}

/// Build the serialized image-generation request body. Pure.
#[cfg(feature = "image")]
pub fn build_image_generation_body(
    model: &str,
    request: &crate::image_generation::ImageGenerationRequest,
) -> Result<Vec<u8>, crate::image_generation::ImageGenerationError> {
    use super::image_generation::{GPT_IMAGE_1, GPT_IMAGE_1_5, GPT_IMAGE_2};

    let mut body = json!({
        "model": model,
        "prompt": request.prompt,
        "size": format!("{}x{}", request.width, request.height),
    });
    if !matches!(model, GPT_IMAGE_1 | GPT_IMAGE_1_5 | GPT_IMAGE_2) {
        body = merge(body, json!({"response_format": "b64_json"}));
    }
    Ok(serde_json::to_vec(&body)?)
}

/// Parse an image-generation response body. Pure.
#[cfg(feature = "image")]
pub fn parse_image_generation_response(
    status: http::StatusCode,
    body: &[u8],
) -> Result<
    crate::image_generation::ImageGenerationResponse<
        super::image_generation::ImageGenerationResponse,
    >,
    crate::image_generation::ImageGenerationError,
> {
    use crate::image_generation::ImageGenerationError;

    if !status.is_success() {
        return Err(ImageGenerationError::from_http_response(
            status,
            String::from_utf8_lossy(body).into_owned(),
        ));
    }
    match serde_json::from_slice::<ApiResponse<super::image_generation::ImageGenerationResponse>>(
        body,
    )? {
        ApiResponse::Ok(response) => response.try_into(),
        ApiResponse::Err(err) => {
            tracing::warn!(message = %err.message, "provider returned an error response");
            Err(ImageGenerationError::from_http_response(
                status,
                String::from_utf8_lossy(body).into_owned(),
            ))
        }
    }
}

/// Generate an image with OpenAI's `/images/generations` endpoint.
#[cfg(feature = "image")]
pub async fn generate_image(
    cfg: &Config,
    rt: &HttpRuntime,
    request: crate::image_generation::ImageGenerationRequest,
) -> Result<
    crate::image_generation::ImageGenerationResponse<
        super::image_generation::ImageGenerationResponse,
    >,
    crate::image_generation::ImageGenerationError,
> {
    use crate::image_generation::ImageGenerationError;

    let body = build_image_generation_body(&cfg.model, &request)?;
    let url = format!("{}/images/generations", cfg.base_url.trim_end_matches('/'));
    let mut builder = http::Request::post(url).header(CONTENT_TYPE, "application/json");
    if let Some(key) = cfg
        .api_key
        .resolve()
        .map_err(|e| ImageGenerationError::RequestError(Box::new(e)))?
    {
        builder = builder.header(AUTHORIZATION, format!("Bearer {key}"));
    }
    for (name, value) in &cfg.extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    let req = builder
        .body(body)
        .map_err(|e| ImageGenerationError::RequestError(Box::new(e)))?;
    let (status, body) = rt.send_bytes(req).await?;
    parse_image_generation_response(status, &body)
}

/// Build the serialized audio-generation (TTS) request body. Pure.
#[cfg(feature = "audio")]
pub fn build_audio_generation_body(
    model: &str,
    request: &crate::audio_generation::AudioGenerationRequest,
) -> Result<Vec<u8>, crate::audio_generation::AudioGenerationError> {
    Ok(serde_json::to_vec(&json!({
        "model": model,
        "input": request.text,
        "voice": request.voice,
        "speed": request.speed,
    }))?)
}

/// Parse an audio-generation response: success bodies are raw audio bytes. Pure.
#[cfg(feature = "audio")]
pub fn parse_audio_generation_response(
    status: http::StatusCode,
    body: Vec<u8>,
) -> Result<
    crate::audio_generation::AudioGenerationResponse<bytes::Bytes>,
    crate::audio_generation::AudioGenerationError,
> {
    use crate::audio_generation::{AudioGenerationError, AudioGenerationResponse};

    if !status.is_success() {
        return Err(AudioGenerationError::from_http_response(
            status,
            String::from_utf8_lossy(&body).into_owned(),
        ));
    }
    Ok(AudioGenerationResponse {
        audio: body.clone(),
        response: bytes::Bytes::from(body),
    })
}

/// Generate speech with OpenAI's `/audio/speech` endpoint.
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

    let body = build_audio_generation_body(&cfg.model, &request)?;
    let url = format!("{}/audio/speech", cfg.base_url.trim_end_matches('/'));
    let mut builder = http::Request::post(url).header(CONTENT_TYPE, "application/json");
    if let Some(key) = cfg
        .api_key
        .resolve()
        .map_err(|e| AudioGenerationError::RequestError(Box::new(e)))?
    {
        builder = builder.header(AUTHORIZATION, format!("Bearer {key}"));
    }
    for (name, value) in &cfg.extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    let req = builder
        .body(body)
        .map_err(|e| AudioGenerationError::RequestError(Box::new(e)))?;
    let (status, body) = rt.send_bytes(req).await?;
    parse_audio_generation_response(status, body)
}

// ================================================================
// Embeddings
// ================================================================

/// Plain-data OpenAI embeddings configuration.
///
/// A sibling of [`Config`] rather than an overload of it: embeddings carry
/// their own model identifier and an optional requested dimension count,
/// which do not belong on the completion configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingConfig {
    /// Reusable HTTP connection data.
    #[serde(flatten)]
    pub connection: crate::providers::HttpConnectionConfig,
    /// Embedding model identifier requests are built for.
    pub model: String,
    /// Requested embedding dimensions, sent verbatim as the `dimensions`
    /// field when set (models that reject the field, like
    /// `text-embedding-ada-002`, should leave it unset).
    pub dimensions: Option<usize>,
}

crate::providers::client::impl_http_connection_config!(EmbeddingConfig);

impl EmbeddingConfig {
    /// Config for `model` reading `OPENAI_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            connection: crate::providers::HttpConnectionConfig::new(
                DEFAULT_BASE_URL.to_string(),
                ApiKeyLocation::Env("OPENAI_API_KEY".to_string()),
            ),
            model: model.into(),
            dimensions: None,
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

    /// The width of the vectors this config produces, when it is known.
    ///
    /// Resolves an explicit [`dimensions`](Self::dimensions) request override
    /// first, then the model's native width from
    /// [`model_dimensions_from_identifier`](super::embedding::model_dimensions_from_identifier).
    /// Returns [`None`] for an unrecognised model with no override, since the
    /// width is then only knowable from a response.
    ///
    /// Callers sizing a vector-store index should prefer this over hardcoding a
    /// literal — the config knows its own model.
    pub fn ndims(&self) -> Option<usize> {
        self.dimensions
            .or_else(|| super::embedding::model_dimensions_from_identifier(&self.model))
    }

    /// Config for `model` built from the process environment.
    ///
    /// Same variables as [`Config::from_env`]: `OPENAI_API_KEY` (required) and
    /// `OPENAI_BASE_URL` (optional).
    ///
    /// # Errors
    /// [`ConfigError`] when a required variable is missing or invalid.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
        let mut cfg = Self::new(model);
        required_env_var("OPENAI_API_KEY")?;
        if let Some(base_url) = optional_env_var("OPENAI_BASE_URL")? {
            cfg.base_url = base_url;
        }
        Ok(cfg)
    }

    /// Request `dimensions`-sized embeddings.
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }
}

/// Build the complete HTTP embeddings request for one chunk of `texts`.
///
/// Pure except for credential resolution.
pub fn build_embedding_request(
    cfg: &EmbeddingConfig,
    texts: &[String],
) -> Result<http::Request<Vec<u8>>, EmbeddingError> {
    let body = super::embedding::build_embedding_body(
        &cfg.model,
        texts,
        cfg.dimensions
            .map(super::embedding::EmbeddingDimensions::Dimensions),
        None,
        None,
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

/// Parse an embeddings response into the normalized
/// [`embeddings::EmbeddingResponse`]. Pure.
pub fn parse_embedding_response(
    status: http::StatusCode,
    body: &str,
    documents: Vec<String>,
) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
    super::embedding::parse_embedding_response(status, body, documents, DESCRIPTOR.name, true)
}

/// Embed `texts`, chunking to honor [`DESCRIPTOR`]'s
/// `max_embedding_documents`; embeddings are returned in input order.
pub async fn embed(
    cfg: &EmbeddingConfig,
    rt: &HttpRuntime,
    texts: Vec<String>,
) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
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
        Vec<crate::OneOrMany<embeddings::Embedding>>,
        crate::completion::Usage,
    ),
    EmbeddingError,
> {
    let (counts, flat) = crate::embeddings::batching::split_batches(texts);
    let response = embed(cfg, rt, flat).await?;
    let groups = crate::embeddings::batching::group_batches(&counts, response.embeddings)?;
    Ok((groups, response.usage))
}

// ================================================================
// Shared OpenAI-compatible request/response stages
//
// The per-provider `functions` modules (groq, deepseek, mistral, …) are
// straight-line code built out of these stages: each calls
// `compatible_typed_request`, applies its own dialect steps to the typed
// request and/or the serialized body, and finishes with
// `compatible_body_value`. Every stage is plain data in, plain data out —
// no provider trait, no hooks.
// ================================================================

/// Stage 1 (pure): a core completion request becomes the typed OpenAI
/// chat-completions request, honoring `descriptor`'s tool and
/// structured-output capabilities.
pub(crate) fn compatible_typed_request(
    model: &str,
    request: &CompletionRequest,
    descriptor: &ProviderDescriptor,
    options: CompletionModelOptions,
) -> Result<super::completion::CompletionRequest, CompletionError> {
    super::completion::CompletionRequest::try_from(super::completion::OpenAIRequestParams {
        model: model.to_string(),
        request: request.clone(),
        strict_tools: options.strict_tools,
        tool_result_array_content: options.tool_result_array_content,
        supports_response_format: descriptor.supports_response_format,
        supports_tools: descriptor.supports_tools,
    })
}

/// Stage 2 (pure): serialize a typed request and merge the streaming
/// parameters (`stream: true` and, when `descriptor.stream_include_usage`,
/// `stream_options.include_usage`).
///
/// `merge` is shallow, so `include_usage` is inserted *into* any
/// caller-supplied `stream_options` rather than merged over it: the caller's
/// keys survive and the usage chunk is still requested.
pub(crate) fn compatible_body_value<T>(
    typed: &T,
    descriptor: &ProviderDescriptor,
    stream: bool,
) -> Result<serde_json::Value, CompletionError>
where
    T: Serialize,
{
    let mut body = serde_json::to_value(typed)?;
    if stream {
        if descriptor.stream_include_usage {
            match body.get_mut("stream_options") {
                Some(serde_json::Value::Object(options)) => {
                    options
                        .entry("include_usage")
                        .or_insert(serde_json::Value::Bool(true));
                }
                Some(_) => {}
                None => {
                    body = merge(body, json!({"stream_options": {"include_usage": true}}));
                }
            }
        }
        body = merge(body, json!({"stream": true}));
    }
    Ok(body)
}

/// Stage 3: the complete HTTP request for a Bearer-authenticated
/// OpenAI-compatible provider.
///
/// Pure except for credential resolution (`ApiKeyLocation::Env` reads the
/// environment).
pub(crate) fn compatible_http_request(
    base_url: &str,
    path: &str,
    api_key: &ApiKeyLocation,
    extra_headers: &[(String, String)],
    body: Vec<u8>,
) -> Result<http::Request<Vec<u8>>, CompletionError> {
    let url = format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        path.trim_start_matches('/')
    );

    let mut builder = http::Request::post(url).header(CONTENT_TYPE, "application/json");
    if let Some(key) = api_key
        .resolve()
        .map_err(|e| CompletionError::RequestError(Box::new(e)))?
    {
        builder = builder.header(AUTHORIZATION, format!("Bearer {key}"));
    }
    for (name, value) in extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    builder
        .body(body)
        .map_err(|e| CompletionError::RequestError(Box::new(e)))
}

/// Parse an OpenAI-compatible response body into the normalized
/// [`completion::CompletionResponse`], stamping `provider`. Pure.
///
/// `R` is the provider's own concrete chat-completions payload type.
pub(crate) fn compatible_parse_response<R>(
    status: http::StatusCode,
    body: &str,
    provider: &str,
) -> Result<completion::CompletionResponse, CompletionError>
where
    R: serde::de::DeserializeOwned
        + TryInto<completion::CompletionResponse, Error = CompletionError>,
{
    if !status.is_success() {
        return Err(CompletionError::from_http_response(
            status,
            body.to_string(),
        ));
    }
    match serde_json::from_str::<ApiResponse<R>>(body)? {
        ApiResponse::Ok(response) => {
            let mut normalized: completion::CompletionResponse = response.try_into()?;
            normalized.provider = provider.to_string();
            Ok(normalized)
        }
        ApiResponse::Err(err) => {
            tracing::warn!(message = %err.message, "provider returned an error response");
            Err(CompletionError::from_http_response(
                status,
                body.to_string(),
            ))
        }
    }
}

/// Open a chat-completions stream for `req` and drive it through the shared
/// sans-IO state machine with `dialect`'s chunk normalizer.
pub(crate) fn compatible_open_stream(
    rt: &HttpRuntime,
    req: http::Request<Vec<u8>>,
    dialect: ChatCompletionsDialect,
) -> crate::streaming::StreamingCompletionResponse {
    use crate::providers::internal::openai_chat_completions_compatible as compat;

    compat::drive_compatible_stream(
        rt.sse_events(req, false),
        compat::ChunkNormalizer::ChatCompletions(dialect),
    )
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

    #[test]
    fn build_list_models_request_sets_url_and_bearer_auth() {
        let cfg = Config::new("gpt-4o").with_api_key("secret");
        let req = build_list_models_request(&cfg).expect("build");
        assert_eq!(req.method(), http::Method::GET);
        assert_eq!(req.uri(), "https://api.openai.com/v1/models");
        assert_eq!(
            req.headers()
                .get(http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok()),
            Some("Bearer secret")
        );
    }

    #[test]
    fn parse_list_models_response_maps_entries() {
        let body = serde_json::json!({
            "data": [{"id": "gpt-4o", "created": 1, "owned_by": "openai"}]
        })
        .to_string();
        let models =
            parse_list_models_response(http::StatusCode::OK, body.as_bytes()).expect("parse");
        assert_eq!(models.len(), 1);
    }
    use crate::OneOrMany;
    use crate::message::Message;

    fn sample_request() -> CompletionRequest {
        CompletionRequest {
            model: None,
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
    fn build_request_body_matches_generic_typed_conversion() {
        // Byte-equality with the generic path's assembly: same typed
        // conversion, same serialization, same finalize hooks.
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
    fn build_request_sets_url_and_auth() {
        let cfg = Config::new("gpt-4o").with_api_key("secret");
        let req = build_request(&cfg, &sample_request(), false).expect("build");
        assert_eq!(req.uri(), "https://api.openai.com/v1/chat/completions");
        assert_eq!(
            req.headers()
                .get(http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok()),
            Some("Bearer secret")
        );
    }

    #[test]
    fn build_embedding_request_sets_url_body_and_auth() {
        let cfg = EmbeddingConfig::new("text-embedding-3-small")
            .with_api_key("secret")
            .with_dimensions(256);
        let texts = vec!["a".to_string(), "b".to_string()];
        let req = build_embedding_request(&cfg, &texts).expect("build");
        assert_eq!(req.uri(), "https://api.openai.com/v1/embeddings");
        assert_eq!(
            req.headers()
                .get(http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok()),
            Some("Bearer secret")
        );
        let body: serde_json::Value = serde_json::from_slice(req.body()).expect("json");
        assert_eq!(body["model"], "text-embedding-3-small");
        assert_eq!(body["input"], serde_json::json!(["a", "b"]));
        assert_eq!(body["dimensions"], 256);
    }

    #[tokio::test]
    async fn embed_and_embed_batches_align_documents_and_usage() {
        let response_body = serde_json::json!({
            "object": "list",
            "model": "text-embedding-3-small",
            "usage": { "prompt_tokens": 4, "total_tokens": 4 },
            "data": [
                { "object": "embedding", "index": 0, "embedding": [0.1] },
                { "object": "embedding", "index": 1, "embedding": [0.2] },
                { "object": "embedding", "index": 2, "embedding": [0.3] }
            ]
        })
        .to_string();
        let http_client = crate::test_utils::RecordingHttpClient::new(response_body);
        let rt = HttpRuntime::recording(http_client.clone());
        let cfg = EmbeddingConfig::new("text-embedding-3-small").with_api_key("k");

        let batches = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["c".to_string()],
        ];
        let (groups, usage) = embed_batches(&cfg, &rt, batches).await.expect("embed");

        // One wire request (3 documents fit one chunk), regrouped 2 + 1 in
        // input order with the response usage carried through.
        assert_eq!(http_client.requests().len(), 1);
        assert_eq!(usage.input_tokens, 4);
        assert_eq!(groups.len(), 2);
        let first: Vec<_> = groups
            .first()
            .expect("first group")
            .iter()
            .map(|e| e.document.clone())
            .collect();
        assert_eq!(first, ["a", "b"]);
        let second: Vec<_> = groups
            .get(1)
            .expect("second group")
            .iter()
            .map(|e| e.document.clone())
            .collect();
        assert_eq!(second, ["c"]);
    }

    #[test]
    fn parse_response_normalizes() {
        let body = serde_json::json!({
            "id": "chatcmpl-1",
            "model": "gpt-4o-2024",
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
        assert_eq!(response.provider, "openai");
        assert_eq!(response.model.as_deref(), Some("gpt-4o-2024"));
        assert_eq!(response.message_id.as_deref(), Some("chatcmpl-1"));
        assert_eq!(response.finish_reason, Some(completion::FinishReason::Stop));
        assert_eq!(response.usage.input_tokens, 3);
        assert_eq!(response.usage.total_tokens, 5);
    }

    #[test]
    fn transcription_form_carries_model_file_and_optional_fields() {
        let request = crate::transcription::TranscriptionRequest {
            data: vec![1, 2, 3],
            filename: "audio.mp3".to_string(),
            language: Some("en".to_string()),
            prompt: Some("context".to_string()),
            temperature: Some(0.2),
            additional_params: None,
        };
        let form = build_transcription_form("whisper-1", request).expect("build");
        let names: Vec<&str> = form.parts().iter().map(|p| p.name()).collect();
        assert_eq!(
            names,
            vec!["model", "file", "language", "prompt", "temperature"]
        );
        let file = form
            .parts()
            .iter()
            .find(|p| p.name() == "file")
            .expect("file part");
        assert_eq!(file.get_filename(), Some("audio.mp3"));
    }

    #[test]
    fn transcription_form_rejects_non_object_additional_params() {
        let request = crate::transcription::TranscriptionRequest {
            data: Vec::new(),
            filename: "a.wav".to_string(),
            language: None,
            prompt: None,
            temperature: None,
            additional_params: Some(serde_json::json!(["not-an-object"])),
        };
        assert!(build_transcription_form("whisper-1", request).is_err());
    }

    #[cfg(feature = "image")]
    #[test]
    fn image_generation_body_gates_response_format_by_model() {
        let request = crate::image_generation::ImageGenerationRequest {
            prompt: "draw a cat".to_string(),
            width: 256,
            height: 512,
            additional_params: None,
        };
        let body = build_image_generation_body("dall-e-3", &request).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["model"], "dall-e-3");
        assert_eq!(value["size"], "256x512");
        assert_eq!(value["response_format"], "b64_json");

        let body = build_image_generation_body("gpt-image-1", &request).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert!(value.get("response_format").is_none());
    }

    #[cfg(feature = "audio")]
    #[test]
    fn audio_generation_body_carries_model_input_voice_and_speed() {
        let request = crate::audio_generation::AudioGenerationRequest {
            text: "hello".to_string(),
            voice: "alloy".to_string(),
            speed: 1.5,
            additional_params: None,
        };
        let body = build_audio_generation_body("tts-1", &request).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["model"], "tts-1");
        assert_eq!(value["input"], "hello");
        assert_eq!(value["voice"], "alloy");
        assert_eq!(value["speed"], 1.5);
    }

    /// System instructions are canonical ordered `Message::System` entries.
    /// This pins that the builder's `.preamble(..)` and an explicitly supplied
    /// system message are the *same* thing, and that multiple system messages
    /// keep their order on the wire.
    #[test]
    fn canonical_system_messages_reach_the_wire_in_order() {
        let cfg = Config::new("gpt-4o");

        let via_preamble = crate::completion::CompletionRequest::builder("now")
            .preamble("You are terse.")
            .messages([crate::message::Message::user("earlier")])
            .build();
        let via_message = crate::completion::CompletionRequest::builder("now")
            .message(crate::message::Message::system("You are terse."))
            .message(crate::message::Message::user("earlier"))
            .build();

        let a = build_request_body(&cfg, &via_preamble, false).expect("builds");
        let b = build_request_body(&cfg, &via_message, false).expect("builds");
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&a).expect("json"),
            serde_json::from_slice::<serde_json::Value>(&b).expect("json"),
            "`.preamble(..)` is sugar for a leading system message, nothing more",
        );

        // Multiple system messages keep their relative order.
        let ordered = crate::completion::CompletionRequest::builder("now")
            .message(crate::message::Message::system("first"))
            .message(crate::message::Message::system("second"))
            .build();
        let body = build_request_body(&cfg, &ordered, false).expect("builds");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        let systems: Vec<&str> = value["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .filter(|m| m["role"] == "system")
            // Content is an array of typed parts, not a bare string.
            .filter_map(|m| m["content"][0]["text"].as_str())
            .collect();
        assert_eq!(systems, ["first", "second"]);
    }
}
