//! OpenAI chat completions as config + pure functions (provider pilot).
//!
//! This module is the data-oriented face of the OpenAI provider: a serde
//! [`Config`], a [`DESCRIPTOR`] capability sheet, and free functions that
//! decompose a completion into its pure parts —
//! [`build_request`] (data → HTTP request, no IO) and [`parse_response`]
//! (bytes → normalized [`completion::CompletionResponse`], no IO) — plus the
//! async [`complete`] wrapper over [`HttpRuntime`](crate::http_runtime::HttpRuntime).
//!
//! During the transition the pure functions delegate to the same typed
//! conversion the [`GenericCompletionModel`](super::completion::GenericCompletionModel)
//! path uses, so both paths produce byte-identical request bodies; the
//! generic path is retired later in the migration.

use http::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::client::ApiResponse;
use super::client::OpenAICompletionsExt;
use super::completion::{CompletionModelOptions, OpenAICompatibleProvider as _};
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::embeddings::{self, EmbeddingError};
use crate::http_runtime::HttpRuntime;
use crate::json_utils::merge;
use crate::providers::descriptor::{ApiKeyLocation, ProviderDescriptor};

/// Default OpenAI API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// OpenAI's capability sheet.
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    name: "openai",
    supports_tools: true,
    supports_response_format: true,
    stream_include_usage: true,
    emits_complete_single_chunk_tool_calls: false,
    composes_native_output_with_tools: true,
    max_embedding_documents: Some(1024),
};

/// Plain-data OpenAI provider configuration.
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
    /// Config for `model` reading `OPENAI_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: ApiKeyLocation::Env("OPENAI_API_KEY".to_string()),
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
    let ext = OpenAICompletionsExt::default();
    let options = CompletionModelOptions::default();
    let mut typed = ext.build_completion_request(cfg.model.clone(), request.clone(), options)?;
    ext.prepare_request(&mut typed)?;
    let mut body = serde_json::to_value(&typed)?;
    if stream {
        if DESCRIPTOR.stream_include_usage {
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
    ext.finalize_request_body_with_options(&mut body, options)?;
    Ok(serde_json::to_vec(&body)?)
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
    let ext = OpenAICompletionsExt::default();
    let path = ext.completion_path(&cfg.model);
    let url = format!("{}{}", cfg.base_url.trim_end_matches('/'), path);
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
    match serde_json::from_str::<ApiResponse<super::completion::CompletionResponse>>(body)? {
        ApiResponse::Ok(response) => response.try_into(),
        ApiResponse::Err(err) => {
            tracing::warn!(message = %err.message, "provider returned an error response");
            Err(CompletionError::from_http_response(
                status,
                body.to_string(),
            ))
        }
    }
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
    use crate::http_runtime::Transport;
    use crate::providers::internal::openai_chat_completions_compatible as compat;

    let req = build_request(cfg, &request, true)?;
    let profile = super::completion::streaming::openai_stream_profile();
    match rt.transport() {
        Transport::Reqwest(client) => {
            compat::send_compatible_streaming_request(client.clone(), req, profile).await
        }
        #[cfg(feature = "test-utils")]
        Transport::Recording(client) => {
            compat::send_compatible_streaming_request(client.clone(), req, profile).await
        }
        #[cfg(feature = "test-utils")]
        Transport::Sequenced(client) => {
            compat::send_compatible_streaming_request(client.clone(), req, profile).await
        }
    }
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
    crate::image_generation::ImageGenerationResponse<super::image_generation::ImageGenerationResponse>,
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
    crate::image_generation::ImageGenerationResponse<super::image_generation::ImageGenerationResponse>,
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
    /// API base URL (defaults to [`DEFAULT_BASE_URL`]).
    pub base_url: String,
    /// Credential location.
    pub api_key: ApiKeyLocation,
    /// Embedding model identifier requests are built for.
    pub model: String,
    /// Requested embedding dimensions, sent verbatim as the `dimensions`
    /// field when set (models that reject the field, like
    /// `text-embedding-ada-002`, should leave it unset).
    pub dimensions: Option<usize>,
    /// Extra headers attached to every request.
    pub extra_headers: Vec<(String, String)>,
}

impl EmbeddingConfig {
    /// Config for `model` reading `OPENAI_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: ApiKeyLocation::Env("OPENAI_API_KEY".to_string()),
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
    super::embedding::parse_embedding_response::<super::OpenAIResponsesExt>(
        status, body, documents,
    )
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
) -> Result<(Vec<crate::OneOrMany<embeddings::Embedding>>, crate::completion::Usage), EmbeddingError>
{
    let (counts, flat) = crate::embeddings::batching::split_batches(texts);
    let response = embed(cfg, rt, flat).await?;
    let groups = crate::embeddings::batching::group_batches(&counts, response.embeddings)?;
    Ok((groups, response.usage))
}

// ================================================================
// Generic OpenAI-compatible helpers
//
// The per-provider `functions` modules (groq, deepseek, together, …) are
// thin wrappers over these: each instantiates its own `Ext` so the
// provider's completion_path/prepare_request/finalize hooks and
// PROVIDER_NAME apply, while the request/parse mechanics stay in one
// place. Transitional compile-time plumbing; retired with the generic
// path later in the migration.
// ================================================================

/// Build the serialized chat-completions body for an OpenAI-compatible `ext`.
pub(crate) fn compatible_request_body<Ext>(
    ext: &Ext,
    model: &str,
    request: &CompletionRequest,
    stream: bool,
) -> Result<Vec<u8>, CompletionError>
where
    Ext: super::completion::OpenAICompatibleProvider,
{
    let options = CompletionModelOptions::default();
    let mut typed = ext.build_completion_request(model.to_string(), request.clone(), options)?;
    ext.prepare_request(&mut typed)?;
    let mut body = serde_json::to_value(&typed)?;
    if stream {
        if Ext::STREAM_INCLUDE_USAGE {
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
    ext.finalize_request_body_with_options(&mut body, options)?;
    Ok(serde_json::to_vec(&body)?)
}

/// Build the complete HTTP request for an OpenAI-compatible `ext` with
/// Bearer authentication.
pub(crate) fn compatible_request<Ext>(
    ext: &Ext,
    base_url: &str,
    api_key: &ApiKeyLocation,
    extra_headers: &[(String, String)],
    model: &str,
    request: &CompletionRequest,
    stream: bool,
) -> Result<http::Request<Vec<u8>>, CompletionError>
where
    Ext: super::completion::OpenAICompatibleProvider,
{
    let path = ext.completion_path(model);
    let url = format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        path.trim_start_matches('/')
    );
    let body = compatible_request_body(ext, model, request, stream)?;

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
/// [`completion::CompletionResponse`], stamping `Ext::PROVIDER_NAME`
/// (mirroring `GenericCompletionModel`). Pure.
pub(crate) fn compatible_parse_response<Ext>(
    status: http::StatusCode,
    body: &str,
) -> Result<completion::CompletionResponse, CompletionError>
where
    Ext: super::completion::OpenAICompatibleProvider,
{
    if !status.is_success() {
        return Err(CompletionError::from_http_response(
            status,
            body.to_string(),
        ));
    }
    match serde_json::from_str::<ApiResponse<Ext::Response>>(body)? {
        ApiResponse::Ok(response) => {
            let mut normalized: completion::CompletionResponse = response.try_into()?;
            normalized.provider = Ext::PROVIDER_NAME.to_string();
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

/// Drive `req` through the shared OpenAI-compatible streaming path with
/// `ext`'s streaming profile.
pub(crate) async fn compatible_open_stream<Ext>(
    ext: Ext,
    rt: &HttpRuntime,
    req: http::Request<Vec<u8>>,
) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError>
where
    Ext: super::completion::OpenAICompatibleProvider
        + Clone
        + crate::wasm_compat::WasmCompatSend
        + 'static,
{
    use crate::http_runtime::Transport;
    use crate::providers::internal::openai_chat_completions_compatible as compat;

    let profile = super::completion::streaming::stream_profile_for(ext);
    match rt.transport() {
        Transport::Reqwest(client) => {
            compat::send_compatible_streaming_request(client.clone(), req, profile).await
        }
        #[cfg(feature = "test-utils")]
        Transport::Recording(client) => {
            compat::send_compatible_streaming_request(client.clone(), req, profile).await
        }
        #[cfg(feature = "test-utils")]
        Transport::Sequenced(client) => {
            compat::send_compatible_streaming_request(client.clone(), req, profile).await
        }
    }
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
}
