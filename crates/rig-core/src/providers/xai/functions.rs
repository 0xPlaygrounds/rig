//! xAI Responses API as config + pure functions.
//!
//! The data-oriented face of the xAI provider: a serde `Config`, a
//! [`DESCRIPTOR`] capability sheet, and free functions decomposing a
//! completion into pure parts — [`build_request_body`] / [`build_request`]
//! (data → HTTP request, no IO) and [`parse_response`] (bytes → normalized
//! [`completion::CompletionResponse`], no IO) — plus async [`complete`] and
//! [`open_stream`] wrappers over [`HttpRuntime`].
//!
//! The pure functions delegate to the typed conversion in
//! [`super::completion`] (`XAICompletionRequest`'s `TryFrom` plus the shared
//! stream-flag helper), so every request body is produced in exactly one
//! place. [`open_stream`] drives the SSE state machine in
//! the crate-internal `xai::streaming` module (`send_xai_streaming_request`).

use http::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use tracing_futures::Instrument;

use super::api::ApiResponse;
use super::completion::{XAICompletionRequest, apply_stream_flag};
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_runtime::HttpRuntime;
use crate::providers::descriptor::{
    ApiKeyLocation, ConfigError, ProviderDescriptor, required_env_var,
};
use crate::telemetry::{CompletionOperation, completion_span};

/// Default xAI API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.x.ai";

/// xAI's capability sheet.
///
/// `supports_response_format` is `false`: the request conversion warns and
/// drops `output_schema` ("Structured outputs currently not supported for
/// xAI").
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    name: "xai",
    supports_tools: true,
    supports_response_format: false,
    stream_include_usage: false,
    emits_complete_single_chunk_tool_calls: false,
    composes_native_output_with_tools: false,
    max_embedding_documents: None,
    verify_path: Some("/v1/api-key"),
};

/// Plain-data xAI provider configuration.
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
    /// Config for `model` reading `XAI_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            connection: crate::providers::HttpConnectionConfig::new(
                DEFAULT_BASE_URL.to_string(),
                ApiKeyLocation::Env("XAI_API_KEY".to_string()),
            ),
            model: model.into(),
        }
    }

    /// Config for `model` built from the process environment.
    ///
    /// Reads `XAI_API_KEY` (required) — the same variable the deleted
    /// `xai::Client::from_env` read. There is no base-URL override: the classic
    /// client always targeted [`DEFAULT_BASE_URL`]. The credential is validated
    /// eagerly but stored as [`ApiKeyLocation::Env`], so the secret is read at
    /// request time rather than held inside the config.
    ///
    /// # Errors
    /// [`ConfigError`] when a required variable is missing or invalid.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
        let cfg = Self::new(model);
        required_env_var("XAI_API_KEY")?;
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

/// Build the serialized Responses API request body for `request`.
///
/// Pure: the exact bytes the wire sees. Delegates to the same typed
/// conversion as the trait path; `stream` merges the top-level
/// `stream: true` flag exactly as the trait streaming path does.
pub fn build_request_body(
    cfg: &Config,
    request: &CompletionRequest,
    stream: bool,
) -> Result<Vec<u8>, CompletionError> {
    let mut typed = XAICompletionRequest::try_from((cfg.model.as_str(), request.clone()))?;
    if stream {
        apply_stream_flag(&mut typed);
    }
    Ok(serde_json::to_vec(&typed)?)
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
    let url = format!("{}/v1/responses", cfg.base_url.trim_end_matches('/'));
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

/// Parse a Responses API response body into the normalized
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
        ApiResponse::Error(error) => {
            tracing::warn!(message = %error.message(), "provider returned an error response");
            Err(CompletionError::from_http_response(
                status,
                body.to_string(),
            ))
        }
    }
}

/// Open a streaming completion for `request`, driving the same SSE machinery
/// as the trait path.
pub async fn open_stream(
    cfg: &Config,
    rt: &HttpRuntime,
    request: CompletionRequest,
) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError> {
    let model = request.model.clone().unwrap_or_else(|| cfg.model.clone());
    let span = completion_span(
        DESCRIPTOR.name,
        &model,
        CompletionOperation::ChatStreaming,
        &request,
    );
    let req = build_request(cfg, &request, true)?;
    super::streaming::send_xai_streaming_request(rt.sse_events(req, false))
        .instrument(span)
        .await
}

/// Generate an image with xAI's `/v1/images/generations` endpoint.
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

    let body = super::image_generation::build_image_generation_body(&cfg.model, &request)?;
    let url = format!(
        "{}/v1/images/generations",
        cfg.base_url.trim_end_matches('/')
    );
    let req = crate::providers::openai::functions::bearer_post(
        url,
        &cfg.api_key,
        &cfg.extra_headers,
        true,
    )?
    .body(body)
    .map_err(|e| ImageGenerationError::RequestError(Box::new(e)))?;
    let (status, body) = rt.send_bytes(req).await?;
    super::image_generation::parse_image_generation_response(
        status,
        &String::from_utf8_lossy(&body),
    )
}

/// Generate speech with xAI's `/v1/tts` endpoint. The route has no model
/// field, so `cfg.model` is unused beyond parity with the other modalities.
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

    let body = super::audio_generation::build_audio_generation_body(&request)?;
    let url = format!("{}/v1/tts", cfg.base_url.trim_end_matches('/'));
    let req = crate::providers::openai::functions::bearer_post(
        url,
        &cfg.api_key,
        &cfg.extra_headers,
        true,
    )?
    .body(body)
    .map_err(|e| AudioGenerationError::RequestError(Box::new(e)))?;
    let (status, body) = rt.send_bytes(req).await?;
    super::audio_generation::parse_audio_generation_response(status, body)
}

/// Send `request` to xAI and return the normalized response.
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
/// endpoint is [`DESCRIPTOR`]'s `verify_path` (`/v1/api-key`, the value the
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
    use crate::message::Message;

    fn sample_request() -> CompletionRequest {
        CompletionRequest {
            model: None,
            chat_history: OneOrMany::one(Message::user("hello")),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: Some(0.7),
            max_tokens: Some(32),
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    #[test]
    fn build_request_body_matches_typed_conversion() {
        let cfg = Config::new("grok-4-0709").with_api_key("k");
        let body = build_request_body(&cfg, &sample_request(), false).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["model"], "grok-4-0709");
        assert_eq!(value["temperature"], 0.7);
        assert_eq!(value["max_output_tokens"], 32);
        assert!(value.get("stream").is_none());

        let streaming = build_request_body(&cfg, &sample_request(), true).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&streaming).expect("json");
        assert_eq!(value["stream"], true);
    }

    #[test]
    fn build_request_honors_model_override() {
        let cfg = Config::new("grok-4-0709").with_api_key("k");
        let mut request = sample_request();
        request.model = Some("grok-3".to_string());
        let body = build_request_body(&cfg, &request, false).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["model"], "grok-3");
    }

    #[test]
    fn build_request_sets_url_and_auth() {
        let cfg = Config::new("grok-4-0709").with_api_key("secret");
        let req = build_request(&cfg, &sample_request(), false).expect("build");
        assert_eq!(req.uri(), "https://api.x.ai/v1/responses");
        assert_eq!(
            req.headers()
                .get(http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok()),
            Some("Bearer secret")
        );
    }

    #[test]
    fn parse_response_normalizes() {
        let body = serde_json::json!({
            "id": "resp_1",
            "model": "grok-4-0709",
            "output": [{
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "hi", "annotations": []}
                ]
            }],
            "usage": {
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5
            }
        })
        .to_string();
        let response = parse_response(http::StatusCode::OK, &body).expect("parse");
        assert_eq!(response.provider, "xai");
        assert_eq!(response.model.as_deref(), Some("grok-4-0709"));
        assert_eq!(response.message_id.as_deref(), Some("msg_1"));
        assert_eq!(response.usage.input_tokens, 3);
        assert_eq!(response.usage.total_tokens, 5);
    }

    #[test]
    fn parse_response_surfaces_provider_error_envelope() {
        let body = r#"{"error":"boom","code":"503"}"#;
        let error = parse_response(http::StatusCode::OK, body).expect_err("should error");
        assert!(error.to_string().contains("boom"));
    }
}
