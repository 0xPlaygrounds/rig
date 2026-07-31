//! Hugging Face chat completions as config + pure functions.
//!
//! The data-oriented face of the Hugging Face provider, mirroring
//! [`crate::providers::openai::functions`]: a serde [`Config`], a
//! [`DESCRIPTOR`] capability sheet, and pure
//! [`build_request`]/[`parse_response`] free functions plus the async
//! [`complete`]/[`open_stream`] wrappers over
//! [`HttpRuntime`]. The request/parse
//! mechanics reuse the shared OpenAI-compatible stages in
//! `openai::functions`; the Hugging Face dialect steps (`completion_path`,
//! `build_body`) live here.
//!
//! Routing is carried by [`Config::sub_provider`]: it selects the
//! chat-completions path, the model identifier written into the request body
//! (Fireworks qualifies ids), and the transcription / image-generation
//! endpoints (which only the default `hf-inference` sub-provider supports).

use serde::{Deserialize, Serialize};

use super::SubProvider;
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_runtime::HttpRuntime;
use crate::providers::descriptor::ChatCompletionsDialect;
use crate::providers::descriptor::{
    ApiKeyLocation, ConfigError, ProviderDescriptor, required_env_var,
};
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
    verify_path: Some("/api/whoami-v2"),
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
    /// Inference-router sub-provider requests are routed through (classic
    /// `ClientBuilder::subprovider`).
    #[serde(default)]
    pub sub_provider: SubProvider,
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
            sub_provider: SubProvider::default(),
            extra_headers: Vec::new(),
        }
    }

    /// Config for `model` built from the process environment.
    ///
    /// Reads `HUGGINGFACE_API_KEY` (required) — the same variable the deleted
    /// `huggingface::Client::from_env` read. There is no base-URL override: the
    /// classic client always targeted the router at [`DEFAULT_BASE_URL`]. The
    /// credential is validated eagerly but stored as [`ApiKeyLocation::Env`], so
    /// the secret is read at request time rather than held inside the config.
    ///
    /// The sub-provider is not environment-derived: it defaults to
    /// [`SubProvider::default`] and is selected with
    /// [`Config::with_sub_provider`], exactly as the classic client's
    /// `ClientBuilder::subprovider` did.
    ///
    /// # Errors
    /// [`ConfigError`] when a required variable is missing or invalid.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
        let cfg = Self::new(model);
        required_env_var("HUGGINGFACE_API_KEY")?;
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

    /// Route requests through `sub_provider`.
    pub fn with_sub_provider(mut self, sub_provider: SubProvider) -> Self {
        self.sub_provider = sub_provider;
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
        &cfg.sub_provider,
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
        &completion_path(&cfg.sub_provider, &cfg.model),
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

/// A minor-modality URL: `{base}/{sub_provider}{endpoint}`.
///
/// `endpoint` comes from [`SubProvider::transcription_endpoint`] /
/// [`SubProvider::image_generation_endpoint`], which only the default
/// `hf-inference` sub-provider implements — the others reject the modality,
/// exactly as the classic transcription/image-generation models did.
fn modality_url(cfg: &Config, endpoint: &str) -> String {
    format!(
        "{}/{}{}",
        cfg.base_url.trim_end_matches('/'),
        cfg.sub_provider,
        endpoint
    )
}

/// Transcribe `request` through the configured sub-provider.
///
/// Only `hf-inference` supports transcription; the other sub-providers
/// return [`TranscriptionError::ProviderError`](crate::transcription::TranscriptionError::ProviderError).
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
    let endpoint = cfg.sub_provider.transcription_endpoint(&cfg.model)?;
    let req = openai_functions::bearer_post(
        modality_url(cfg, &endpoint),
        &cfg.api_key,
        &cfg.extra_headers,
        true,
    )?
    .body(body)
    .map_err(|e| TranscriptionError::RequestError(Box::new(e)))?;
    let (status, body) = rt.send_bytes(req).await?;
    super::transcription::parse_transcription_response(status, &body)
}

/// Generate an image through the configured sub-provider.
///
/// Only `hf-inference` supports image generation; the other sub-providers
/// return [`ImageGenerationError::ProviderError`](crate::image_generation::ImageGenerationError::ProviderError).
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
    let endpoint = cfg.sub_provider.image_generation_endpoint(&cfg.model)?;
    let req = openai_functions::bearer_post(
        modality_url(cfg, &endpoint),
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
/// Verify that `cfg`'s credential is accepted by the provider.
///
/// The data-oriented replacement for the deleted `VerifyClient::verify`: the
/// endpoint is [`DESCRIPTOR`]'s `verify_path` (`/api/whoami-v2`, the value the
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
        assert_eq!(
            req.uri(),
            "https://router.huggingface.co/v1/chat/completions"
        );
        let value: serde_json::Value = serde_json::from_slice(req.body()).expect("json");
        assert_eq!(value["model"], "test-model");
    }

    #[test]
    fn sub_provider_reaches_the_request_body_model() {
        let cfg = Config::new("deepseek-v3")
            .with_api_key("secret")
            .with_sub_provider(SubProvider::Fireworks);
        let req = build_request(&cfg, &sample_request(), false).expect("build");
        let value: serde_json::Value = serde_json::from_slice(req.body()).expect("json");
        assert_eq!(value["model"], "accounts/fireworks/models/deepseek-v3");
        // The router addresses every sub-provider through the same
        // chat-completions path; only the body model identifier changes.
        assert_eq!(
            req.uri(),
            "https://router.huggingface.co/v1/chat/completions"
        );
    }

    #[test]
    fn sub_provider_routes_the_modality_url() {
        let cfg = Config::new("openai/whisper-large-v3").with_api_key("secret");
        let endpoint = cfg
            .sub_provider
            .transcription_endpoint(&cfg.model)
            .expect("hf-inference supports transcription");
        assert_eq!(
            modality_url(&cfg, &endpoint),
            "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3"
        );

        // Non-default sub-providers reject the minor modalities, exactly as
        // the classic transcription/image-generation models did.
        let together = cfg.clone().with_sub_provider(SubProvider::Together);
        assert!(
            together
                .sub_provider
                .transcription_endpoint(&together.model)
                .is_err()
        );
    }

    #[test]
    fn config_sub_provider_round_trips_through_serde() {
        for sub_provider in [
            SubProvider::Together,
            SubProvider::Fireworks,
            SubProvider::Custom("my-route".to_string()),
        ] {
            let cfg = Config::new("m").with_sub_provider(sub_provider);
            let json = serde_json::to_string(&cfg).expect("serialize");
            let back: Config = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, cfg);
        }
        // A config serialized before the field existed still deserializes.
        let json = serde_json::to_string(&Config::new("m")).expect("serialize");
        let mut value: serde_json::Value = serde_json::from_str(&json).expect("json");
        if let Some(map) = value.as_object_mut() {
            map.remove("sub_provider");
        }
        let legacy: Config = serde_json::from_value(value).expect("deserialize legacy");
        assert_eq!(legacy.sub_provider, SubProvider::HFInference);
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
