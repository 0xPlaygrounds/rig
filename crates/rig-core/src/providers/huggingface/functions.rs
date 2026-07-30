//! Hugging Face chat completions as config + pure functions.
//!
//! The data-oriented face of the Hugging Face provider, mirroring
//! [`crate::providers::openai::functions`]: a serde [`Config`], a
//! [`DESCRIPTOR`] capability sheet, and pure
//! [`build_request`]/[`parse_response`] free functions plus the async
//! [`complete`]/[`open_stream`] wrappers over
//! [`HttpRuntime`](crate::http_runtime::HttpRuntime). The request/parse
//! mechanics reuse the shared OpenAI-compatible stages in
//! `openai::functions`; the Hugging Face dialect steps ([`completion_path`],
//! [`build_body`]) live here and are what the classic
//! [`HuggingFaceExt`](super::client::HuggingFaceExt) trait impl forwards to.
//! The [`Config`] path targets the default sub-provider
//! ([`SubProvider::default`]), matching the classic default client.

use serde::{Deserialize, Serialize};

use super::client::SubProvider;
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_runtime::HttpRuntime;
use crate::providers::descriptor::ChatCompletionsDialect;
use crate::providers::descriptor::{ApiKeyLocation, ProviderDescriptor};
use crate::providers::openai::completion::CompletionModelOptions;
use crate::providers::openai::functions as openai_functions;

/// Default Hugging Face API base URL.
pub const DEFAULT_BASE_URL: &str = "https://router.huggingface.co";

/// Hugging Face's Chat Completions streaming dialect (OpenAI's own).
pub(crate) const STREAM_DIALECT: ChatCompletionsDialect =
    ChatCompletionsDialect::from_descriptor(&DESCRIPTOR);

/// Hugging Face's capability sheet.
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    name: "huggingface",
    supports_tools: true,
    supports_response_format: false,
    stream_include_usage: true,
    emits_complete_single_chunk_tool_calls: false,
    composes_native_output_with_tools: false,
    max_embedding_documents: None,
};

/// Plain-data Hugging Face provider configuration.
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
    /// Config for `model` reading `HUGGINGFACE_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: ApiKeyLocation::Env("HUGGINGFACE_API_KEY".to_string()),
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

/// The chat-completions request path for `model` on `subprovider`.
///
/// Chat completions live under the router's `/v1`, while the minor modalities
/// use root-relative paths, so the prefix cannot live in the base URL.
pub(crate) fn completion_path(subprovider: &SubProvider, model: &str) -> String {
    subprovider.completion_endpoint(model)
}

/// Hugging Face's chat-completions body assembly.
///
/// One dialect step over OpenAI's: some sub-providers (Fireworks) address
/// models through a fully-qualified identifier in the request body.
pub(crate) fn build_body(
    subprovider: &SubProvider,
    model: &str,
    request: &CompletionRequest,
    options: CompletionModelOptions,
    stream: bool,
) -> Result<Vec<u8>, CompletionError> {
    let mut typed =
        openai_functions::compatible_typed_request(model, request, &DESCRIPTOR, options)?;
    // Some sub-providers (Fireworks) address models through a qualified
    // identifier in the request body.
    typed.model = subprovider.model_identifier(&typed.model);
    let body = openai_functions::compatible_body_value(&typed, &DESCRIPTOR, stream)?;
    Ok(serde_json::to_vec(&body)?)
}

/// Build the serialized chat-completions request body for `request`. Pure.
pub fn build_request_body(
    cfg: &Config,
    request: &CompletionRequest,
    stream: bool,
) -> Result<Vec<u8>, CompletionError> {
    build_body(
        &SubProvider::default(),
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
        &completion_path(&SubProvider::default(), &cfg.model),
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

/// The HF Inference modality URL: `{base}/hf-inference/models/{model}`.
///
/// The classic client routes minor modalities through the `hf-inference`
/// sub-provider only; the other sub-providers do not support them.
fn hf_inference_url(cfg: &Config) -> String {
    format!(
        "{}/hf-inference/models/{}",
        cfg.base_url.trim_end_matches('/'),
        cfg.model.trim_start_matches('/')
    )
}

/// Transcribe `request` with the HF Inference sub-provider.
pub async fn transcribe(
    cfg: &Config,
    rt: &HttpRuntime,
    request: crate::transcription::TranscriptionRequest,
) -> Result<
    crate::transcription::TranscriptionResponse<super::transcription::TranscriptionResponse>,
    crate::transcription::TranscriptionError,
> {
    use crate::transcription::TranscriptionError;

    let body = super::transcription::build_transcription_body(&request.data)?;
    let req = openai_functions::bearer_post(
        hf_inference_url(cfg),
        &cfg.api_key,
        &cfg.extra_headers,
        true,
    )?
    .body(body)
    .map_err(|e| TranscriptionError::RequestError(Box::new(e)))?;
    let (status, body) = rt.send_bytes(req).await?;
    super::transcription::parse_transcription_response(status, &body)
}

/// Generate an image with the HF Inference sub-provider.
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

    let body = super::image_generation::build_image_generation_body(&request)?;
    let req = openai_functions::bearer_post(
        hf_inference_url(cfg),
        &cfg.api_key,
        &cfg.extra_headers,
        true,
    )?
    .body(body)
    .map_err(|e| ImageGenerationError::RequestError(Box::new(e)))?;
    let (status, body) = rt.send_bytes(req).await?;
    super::image_generation::parse_image_generation_response(status, body)
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

/// Send `request` to Hugging Face and return the normalized response.
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
        assert_eq!(
            req.uri(),
            "https://router.huggingface.co/v1/chat/completions"
        );
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
        assert_eq!(response.provider, "huggingface");
        assert_eq!(response.usage.input_tokens, 3);
        assert_eq!(response.usage.total_tokens, 5);
    }
}
