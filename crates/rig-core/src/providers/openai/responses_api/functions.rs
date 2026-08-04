//! OpenAI Responses API as config + pure functions.
//!
//! The data-oriented face of the canonical OpenAI provider (which speaks the
//! Responses API): a serde [`Config`], a [`DESCRIPTOR`] capability sheet, and
//! free functions decomposing a completion into its pure parts —
//! [`build_request_body`] / [`build_request`] (data → HTTP request, no IO)
//! and [`parse_response`] (bytes → normalized
//! [`completion::CompletionResponse`], no IO) — plus the async [`complete`]
//! and [`open_stream`] wrappers over [`HttpRuntime`].
//!
//! Request assembly goes through the typed conversion
//! ([`super::CompletionRequest`]`::try_from(`[`ResponsesRequestParams`]`)`),
//! response parsing through [`super::CompletionResponse`]`::try_into`, and
//! streaming through the sans-IO SSE machinery in [`super::streaming`].
//!
//! WebSocket mode is the third entry point: `websocket::connect`, available
//! with the `websocket` feature, takes the same [`Config`].
//!
//! The chat-completions flavored OpenAI face lives in
//! [`super::super::functions`] (`openai::functions`).

use http::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};

use super::{ResponsesRequestParams, SystemInstructionsPlacement};
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_runtime::HttpRuntime;
use crate::providers::descriptor::{
    ApiKeyLocation, ConfigError, ProviderDescriptor, optional_env_var, required_env_var,
};
use crate::telemetry::{CompletionOperation, completion_span};

/// Default OpenAI API base URL.
pub const DEFAULT_BASE_URL: &str = super::super::functions::DEFAULT_BASE_URL;

/// The Responses API capability sheet for the canonical OpenAI provider.
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    name: "openai",
    supports_tools: true,
    supports_response_format: true,
    // The Responses API has no `stream_options.include_usage` opt-in; usage
    // always arrives on the terminal `response.completed` event.
    stream_include_usage: false,
    emits_complete_single_chunk_tool_calls: false,
    // The Responses API constrains only the final assistant message via
    // `text.format`; tools are still called across turns. See issue #1928.
    composes_native_output_with_tools: true,
    max_embedding_documents: Some(1024),
    verify_path: Some("/models"),
};

/// Plain-data OpenAI Responses API provider configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Config {
    /// Reusable HTTP connection data.
    #[serde(flatten)]
    pub connection: crate::providers::HttpConnectionConfig,
    /// Model identifier requests are built for.
    pub model: String,
    /// Where Rig system instructions are placed in built requests.
    #[serde(default)]
    pub system_instructions_placement: SystemInstructionsPlacement,
    /// Whether function tool schemas are sanitized into OpenAI's strict-mode
    /// subset and marked `strict: true`. Disabled by default, matching Chat
    /// Completions.
    #[serde(default)]
    pub strict_tools: bool,
    /// Tools always appended to outgoing requests, on top of the ones carried
    /// by the [`CompletionRequest`]. This is how the Responses-only hosted
    /// tools (`web_search`, `file_search`, `computer_use`) are requested —
    /// they have no representation in Rig's normalized tool list.
    #[serde(default)]
    pub tools: Vec<super::ResponsesToolDefinition>,
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
            system_instructions_placement: SystemInstructionsPlacement::default(),
            strict_tools: false,
            tools: Vec::new(),
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

    /// Override where Rig system instructions are placed. See
    /// [`SystemInstructionsPlacement`].
    pub fn with_system_instructions_placement(
        mut self,
        placement: SystemInstructionsPlacement,
    ) -> Self {
        self.system_instructions_placement = placement;
        self
    }

    /// Send Rig system instructions as `system` messages in `input` instead of
    /// through the top-level `instructions` field. For OpenAI-compatible
    /// backends that reject `instructions`.
    pub fn with_system_instructions_as_messages(self) -> Self {
        self.with_system_instructions_placement(SystemInstructionsPlacement::InputSystemMessages)
    }

    /// Sanitize function tool schemas into OpenAI's strict-mode subset and mark
    /// them `strict: true`.
    pub fn with_strict_tools(mut self) -> Self {
        self.strict_tools = true;
        self
    }

    /// Append a tool to every request built from this config. Use for hosted
    /// Responses tools such as
    /// [`ResponsesToolDefinition::web_search`](super::ResponsesToolDefinition::web_search).
    pub fn with_tool(mut self, tool: impl Into<super::ResponsesToolDefinition>) -> Self {
        self.tools.push(tool.into());
        self
    }

    /// Append several tools to every request built from this config.
    pub fn with_tools<I, Tool>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = Tool>,
        Tool: Into<super::ResponsesToolDefinition>,
    {
        self.tools.extend(tools.into_iter().map(Into::into));
        self
    }
}

/// Build the serialized Responses API request body for `request`.
///
/// Pure: the exact bytes the wire sees. `stream` sets the body's `stream`
/// flag.
pub fn build_request_body(
    cfg: &Config,
    request: &CompletionRequest,
    stream: bool,
) -> Result<Vec<u8>, CompletionError> {
    Ok(serde_json::to_vec(&build_typed_request(
        cfg, request, stream,
    )?)?)
}

/// Build the typed Responses request for `request`. Pure.
///
/// Split out of [`build_request_body`] because WebSocket mode sends the same
/// typed request as an event payload rather than as an HTTP body.
pub(crate) fn build_typed_request(
    cfg: &Config,
    request: &CompletionRequest,
    stream: bool,
) -> Result<super::CompletionRequest, CompletionError> {
    let mut typed = super::CompletionRequest::try_from(ResponsesRequestParams {
        model: cfg.model.clone(),
        request: request.clone(),
        system_instructions_placement: cfg.system_instructions_placement,
    })?;
    typed.tools.extend(cfg.tools.iter().cloned());
    if cfg.strict_tools {
        typed.tools = typed
            .tools
            .into_iter()
            .map(super::ResponsesToolDefinition::normalize)
            .collect();
    }
    if stream {
        typed.stream = Some(true);
    }
    Ok(typed)
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
    let url = format!("{}/responses", cfg.base_url.trim_end_matches('/'));
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
    let raw = serde_json::from_str::<super::CompletionResponse>(body)?;
    raw.try_into()
}

/// Open a streaming completion for `request`, driving the same SSE machinery
/// as the trait path.
pub async fn open_stream(
    cfg: &Config,
    rt: &HttpRuntime,
    request: CompletionRequest,
) -> Result<crate::streaming::CompletionStream, CompletionError> {
    let model = request.model.clone().unwrap_or_else(|| cfg.model.clone());
    let span = completion_span(
        DESCRIPTOR.name,
        &model,
        CompletionOperation::ChatStreaming,
        &request,
    );
    let req = build_request(cfg, &request, true)?;
    Ok(super::streaming::stream_from_event_source(
        rt.sse_events(req, false),
        span,
    ))
}

/// Send `request` to the OpenAI Responses API and return the normalized
/// response.
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
    use crate::OneOrMany;
    use crate::message::Message;

    fn sample_request() -> CompletionRequest {
        CompletionRequest {
            model: None,
            chat_history: OneOrMany::many(vec![
                Message::system("Respond tersely.".to_string()),
                Message::user("hello"),
            ])
            .expect("non-empty"),
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
    fn build_request_body_matches_typed_conversion() {
        let cfg = Config::new("gpt-5").with_api_key("k");
        let body = build_request_body(&cfg, &sample_request(), false).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["model"], "gpt-5");
        assert_eq!(value["temperature"], 0.5);
        assert_eq!(value["max_output_tokens"], 64);
        // Leading system instructions are lifted into `instructions`.
        assert_eq!(value["instructions"], "Respond tersely.");
        assert!(value.get("stream").is_none());

        let streaming = build_request_body(&cfg, &sample_request(), true).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&streaming).expect("json");
        assert_eq!(value["stream"], true);
    }

    #[test]
    fn build_request_body_honors_system_message_placement() {
        let cfg = Config::new("gpt-5")
            .with_api_key("k")
            .with_system_instructions_placement(SystemInstructionsPlacement::InputSystemMessages);
        let body = build_request_body(&cfg, &sample_request(), false).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert!(value.get("instructions").is_none());
    }

    #[test]
    fn build_request_sets_url_and_auth() {
        let cfg = Config::new("gpt-5").with_api_key("secret");
        let req = build_request(&cfg, &sample_request(), false).expect("build");
        assert_eq!(req.uri(), "https://api.openai.com/v1/responses");
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
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "gpt-5",
            "usage": {
                "input_tokens": 3,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 5
            },
            "output": [{
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "annotations": [], "text": "hi"}]
            }],
            "tools": []
        })
        .to_string();
        let response = parse_response(http::StatusCode::OK, &body).expect("parse");
        assert_eq!(response.usage.input_tokens, 3);
        assert_eq!(response.usage.total_tokens, 5);
        assert_eq!(response.finish_reason, Some(completion::FinishReason::Stop));
    }

    #[test]
    fn parse_response_preserves_error_status() {
        let error = parse_response(
            http::StatusCode::UNAUTHORIZED,
            r#"{"error":{"message":"bad key"}}"#,
        )
        .expect_err("should error");
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::UNAUTHORIZED)
        );
    }
}
