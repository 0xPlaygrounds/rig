//! OpenRouter chat completions as config + pure functions.
//!
//! The data-oriented face of the OpenRouter provider, mirroring
//! [`crate::providers::openai::functions`]: a serde [`Config`], a
//! [`DESCRIPTOR`] capability sheet, and pure
//! [`build_request`]/[`parse_response`] free functions plus the async
//! [`complete`]/[`open_stream`] wrappers over
//! [`HttpRuntime`](crate::http_runtime::HttpRuntime). The request/parse
//! mechanics are shared with the other OpenAI-compatible providers via
//! `openai::functions`; this module instantiates them with
//! [`OpenRouterExt`](super::client::OpenRouterExt) so OpenRouter's paths, hooks, and
//! provider name apply.

use serde::{Deserialize, Serialize};

use super::client::OpenRouterExt as Ext;
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_runtime::HttpRuntime;
use crate::providers::descriptor::{ApiKeyLocation, ProviderDescriptor};
use crate::providers::openai::functions as openai_functions;

/// Default OpenRouter API base URL.
pub const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";

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
    openai_functions::compatible_request_body(&Ext, &cfg.model, request, stream)
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
    openai_functions::compatible_request(
        &Ext,
        &cfg.base_url,
        &cfg.api_key,
        &cfg.extra_headers,
        &cfg.model,
        request,
        stream,
    )
}

/// Parse a chat-completions response body into the normalized
/// [`completion::CompletionResponse`]. Pure.
pub fn parse_response(
    status: http::StatusCode,
    body: &str,
) -> Result<completion::CompletionResponse, CompletionError> {
    openai_functions::compatible_parse_response::<Ext>(status, body)
}

/// Open a streaming completion for `request`.
pub async fn open_stream(
    cfg: &Config,
    rt: &HttpRuntime,
    request: CompletionRequest,
) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError> {
    let req = build_request(cfg, &request, true)?;
    openai_functions::compatible_open_stream(Ext, rt, req).await
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

#[cfg(test)]
mod tests {
    use super::*;
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
