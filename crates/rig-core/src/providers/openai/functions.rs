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
}
