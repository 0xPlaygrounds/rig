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
//! ([`OpenrouterCompletionRequest`](super::completion::OpenrouterCompletionRequest))
//! and its own body finalization (provider routing preferences and prompt
//! caching), both applied by [`build_body`].

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
use crate::providers::openai::functions as openai_functions;

/// Default OpenRouter API base URL.
pub const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// OpenRouter's Chat Completions streaming dialect.
///
/// OpenRouter never receives `stream_options.include_usage` (its usage rides
/// the final chunk under its own [`Usage`](super::client::Usage) shape), and
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

#[cfg(test)]
mod tests {
    use super::*;

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
