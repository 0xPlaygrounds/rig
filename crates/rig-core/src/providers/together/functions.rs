//! Together AI chat completions as config + pure functions.
//!
//! The data-oriented face of the Together AI provider, mirroring
//! [`crate::providers::openai::functions`]: a serde [`Config`], a
//! [`DESCRIPTOR`] capability sheet, and pure
//! [`build_request`]/[`parse_response`] free functions plus the async
//! [`complete`]/[`open_stream`] wrappers over
//! [`HttpRuntime`](crate::http_runtime::HttpRuntime). The request/parse
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
};

/// Plain-data Together AI provider configuration.
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
    /// Config for `model` reading `TOGETHER_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: ApiKeyLocation::Env("TOGETHER_API_KEY".to_string()),
            model: model.into(),
            extra_headers: Vec::new(),
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
