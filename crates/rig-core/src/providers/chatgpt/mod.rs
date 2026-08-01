//! ChatGPT subscription OAuth provider.
//!
//! This provider targets the ChatGPT subscription backend exposed at
//! `https://chatgpt.com/backend-api/codex`.
//!
//! # Example
//! ```no_run
//! use rig_core::providers::chatgpt;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! # let request = rig_core::completion::CompletionRequest::from_prompt("hello");
//! let cfg = chatgpt::functions::config_from_env(chatgpt::GPT_5_3_CODEX).await?;
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//! let response = chatgpt::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```

pub mod auth;
pub mod functions;

#[allow(dead_code)]
mod base_client {
    crate::providers::client::define_http_client! {
        config = super::functions::Config,
        default_base_url = super::functions::DEFAULT_BASE_URL,
        api_key_required = true,
    }
}

/// Concrete ChatGPT client preserving account and request defaults.
#[derive(Clone, Debug)]
pub struct Client {
    base: base_client::Client,
    account_id: Option<String>,
    default_instructions: Option<String>,
    originator: String,
    user_agent: String,
}

/// Monomorphic ChatGPT client builder.
#[derive(Clone, Debug)]
pub struct ClientBuilder {
    base: base_client::ClientBuilder,
    account_id: Option<String>,
    default_instructions: Option<String>,
    originator: String,
    user_agent: String,
}

impl Client {
    /// Build from an explicit `CHATGPT_ACCESS_TOKEN` environment credential.
    pub fn from_env() -> Result<Self, crate::providers::ConfigError> {
        let config = functions::Config::from_env(String::new())?;
        Ok(Self::from_config_projection(config))
    }

    /// Build using the full ChatGPT environment/OAuth credential resolution.
    pub async fn from_env_with_oauth() -> Result<Self, functions::ConfigFromEnvError> {
        let config = functions::config_from_env(String::new()).await?;
        Ok(Self::from_config_projection(config))
    }

    fn from_config_projection(config: functions::Config) -> Self {
        let base = base_client::Client::from_connection(
            config.connection,
            crate::http_runtime::HttpRuntime::new(),
        );
        Self {
            base,
            account_id: config.account_id,
            default_instructions: config.default_instructions,
            originator: config.originator,
            user_agent: config.user_agent,
        }
    }

    /// Start a concrete client builder.
    pub fn builder() -> ClientBuilder {
        let defaults = functions::Config::new(String::new());
        ClientBuilder {
            base: base_client::Client::builder(),
            account_id: None,
            default_instructions: defaults.default_instructions,
            originator: defaults.originator,
            user_agent: defaults.user_agent,
        }
    }

    /// Build with an explicit ChatGPT access token.
    pub fn new(access_token: impl Into<String>) -> Self {
        let defaults = functions::Config::new(String::new());
        Self {
            base: base_client::Client::new(access_token),
            account_id: None,
            default_instructions: defaults.default_instructions,
            originator: defaults.originator,
            user_agent: defaults.user_agent,
        }
    }

    /// Materialize plain ChatGPT configuration for `model`.
    pub fn config(&self, model: impl Into<String>) -> functions::Config {
        let mut config = self.base.config(model);
        config.account_id = self.account_id.clone();
        config.default_instructions = self.default_instructions.clone();
        config.originator = self.originator.clone();
        config.user_agent = self.user_agent.clone();
        config
    }

    /// Canonical HTTP connection data.
    pub fn connection_config(&self) -> &crate::providers::HttpConnectionConfig {
        self.base.connection_config()
    }

    /// Shared HTTP runtime.
    pub fn http_runtime(&self) -> crate::http_runtime::HttpRuntime {
        self.base.http_runtime()
    }

    /// Compatibility alias for [`Self::http_runtime`].
    pub fn http(&self) -> crate::http_runtime::HttpRuntime {
        self.http_runtime()
    }
}

impl ClientBuilder {
    /// Set an inline ChatGPT access token.
    pub fn access_token(self, token: impl Into<String>) -> Self {
        Self {
            base: self.base.api_key(token),
            ..self
        }
    }

    /// Compatibility alias for [`Self::access_token`].
    pub fn api_key(self, token: impl Into<String>) -> Self {
        self.access_token(token)
    }

    /// Attach a ChatGPT account id.
    pub fn account_id(mut self, account_id: impl Into<String>) -> Self {
        self.account_id = Some(account_id.into());
        self
    }

    /// Override the backend base URL.
    pub fn base_url(self, base_url: impl Into<String>) -> Self {
        Self {
            base: self.base.base_url(base_url),
            ..self
        }
    }

    /// Override default instructions applied to every request.
    pub fn default_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.default_instructions = Some(instructions.into());
        self
    }

    /// Clear default instructions.
    pub fn without_default_instructions(mut self) -> Self {
        self.default_instructions = None;
        self
    }

    /// Override the required originator header.
    pub fn originator(mut self, originator: impl Into<String>) -> Self {
        self.originator = originator.into();
        self
    }

    /// Override the user-agent header.
    pub fn user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = user_agent.into();
        self
    }

    /// Append a connection-wide header.
    pub fn extra_header(self, name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            base: self.base.extra_header(name, value),
            ..self
        }
    }

    /// Reuse an existing HTTP runtime.
    pub fn http_runtime(self, http: crate::http_runtime::HttpRuntime) -> Self {
        Self {
            base: self.base.http_runtime(http),
            ..self
        }
    }

    /// Validate and build the client.
    pub fn build(self) -> Result<Client, crate::providers::ClientBuildError> {
        Ok(Client {
            base: self.base.build()?,
            account_id: self.account_id,
            default_instructions: self.default_instructions,
            originator: self.originator,
            user_agent: self.user_agent,
        })
    }
}

use crate::completion::{self, CompletionError};
use crate::providers::openai::responses_api::{
    self, CompletionRequest as ResponsesRequest, Include,
};
use std::path::PathBuf;

pub(crate) const CHATGPT_API_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
pub(crate) const DEFAULT_ORIGINATOR: &str = "rig";
pub(crate) const DEFAULT_INSTRUCTIONS: &str = "You are ChatGPT, a helpful AI assistant.";

/// `gpt-5.4`
pub const GPT_5_4: &str = "gpt-5.4";
/// `gpt-5.4-pro`
pub const GPT_5_4_PRO: &str = "gpt-5.4-pro";
/// `gpt-5.3-codex`
pub const GPT_5_3_CODEX: &str = "gpt-5.3-codex";
/// `gpt-5.3-codex-spark`
pub const GPT_5_3_CODEX_SPARK: &str = "gpt-5.3-codex-spark";
/// `gpt-5.3-instant`
pub const GPT_5_3_INSTANT: &str = "gpt-5.3-instant";
/// `gpt-5.3-chat-latest`
pub const GPT_5_3_CHAT_LATEST: &str = "gpt-5.3-chat-latest";

/// Build the ChatGPT Codex Responses request as plain data.
///
/// The single source of truth for ChatGPT request bodies;
/// [`functions::build_request_body`] routes through it. The ChatGPT backend rejects the
/// `system` role in `input`, so system instructions always use
/// [`responses_api::SystemInstructionsPlacement::AllInstructions`], and the
/// backend requires SSE — `stream` is always `Some(true)`, even for blocking
/// completions.
pub(crate) fn build_codex_responses_request(
    model: String,
    default_tools: &[responses_api::ResponsesToolDefinition],
    strict_tools: bool,
    default_instructions: Option<&str>,
    request: completion::CompletionRequest,
) -> Result<ResponsesRequest, CompletionError> {
    // Materialize the configured default as a canonical leading system message
    // *before* conversion, so the completion span and the wire body are derived
    // from the same request. Previously this was merged into
    // `request.instructions` after conversion, which meant
    // `gen_ai.system_instructions` silently omitted an instruction that was
    // actually sent — a gap no cassette can catch, because span attributes are
    // not part of the recorded bytes.
    let request = apply_default_instructions(request, default_instructions);

    let mut request = ResponsesRequest::try_from(responses_api::ResponsesRequestParams {
        model,
        request,
        system_instructions_placement: responses_api::SystemInstructionsPlacement::AllInstructions,
    })?;
    request.tools.extend(default_tools.iter().cloned());
    if strict_tools {
        request.tools = request
            .tools
            .into_iter()
            .map(responses_api::ResponsesToolDefinition::normalize)
            .collect();
    }

    // Byte-preservation for the empty-default case: the deleted merge emitted
    // `Some("")` here, which a canonical system message cannot express. No
    // instruction content is involved, so telemetry loses nothing.
    if request.instructions.is_none() && default_instructions.is_some_and(str::is_empty) {
        request.instructions = Some(String::new());
    }

    request.temperature = None;
    request.max_output_tokens = None;
    request.stream = Some(true);

    let include = request
        .additional_parameters
        .include
        .get_or_insert_with(Vec::new);
    if !include
        .iter()
        .any(|item| matches!(item, Include::ReasoningEncryptedContent))
    {
        include.push(Include::ReasoningEncryptedContent);
    }

    request.additional_parameters.background = None;
    request.additional_parameters.metadata.clear();
    request.additional_parameters.parallel_tool_calls = None;
    request.additional_parameters.service_tier = None;
    request.additional_parameters.store = Some(false);
    request.additional_parameters.text = None;
    request.additional_parameters.top_p = None;
    request.additional_parameters.user = None;

    Ok(request)
}

/// Parse a ChatGPT Codex SSE completion body into the normalized response,
/// including the streamed-text fallback for empty `response.completed`
/// payloads.
///
/// Used by [`functions::complete`]. Async
/// only because the empty-output fallback reuses the async SSE accumulator.
pub(crate) async fn parse_codex_sse_response(
    status: http::StatusCode,
    text: &str,
) -> Result<completion::CompletionResponse, CompletionError> {
    if !status.is_success() {
        return Err(CompletionError::from_http_response(
            status,
            text.to_string(),
        ));
    }

    let raw_response = responses_api::streaming::parse_sse_completion_body(text, "ChatGPT")?;

    let span = tracing::Span::current();
    span.record("gen_ai.response.id", raw_response.id.as_str());
    span.record("gen_ai.response.model", raw_response.model.as_str());

    match raw_response.clone().try_into() {
        Ok(response) => Ok(response),
        Err(CompletionError::ResponseError(_)) if raw_response.output.is_empty() => {
            responses_api::streaming::completion_response_from_sse_body(text, raw_response).await
        }
        Err(error) => Err(error),
    }
}

pub(crate) fn default_user_agent() -> String {
    format!(
        "rig/{} ({} {}; {})",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
        DEFAULT_ORIGINATOR
    )
}

pub(crate) fn default_auth_file() -> Option<PathBuf> {
    config_dir().map(|dir| dir.join("chatgpt").join("auth.json"))
}

fn config_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("APPDATA").map(PathBuf::from)
    }

    #[cfg(not(target_os = "windows"))]
    {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
    }
}

/// Prepend `default_instructions` to `request` as a leading canonical system
/// message, unless the instructions it already carries contain them.
///
/// This replaced a post-conversion merge: the
/// `AllInstructions` placement joins every system message with `"\n\n"`, so
/// prepending here yields the same `instructions` string the post-conversion
/// merge produced — while keeping telemetry and the wire body reading the same
/// request.
fn apply_default_instructions(
    request: completion::CompletionRequest,
    default_instructions: Option<&str>,
) -> completion::CompletionRequest {
    // An empty default carries no instruction text, so there is nothing to
    // materialize — and nothing telemetry could report. It is still preserved
    // on the wire below, because the Responses lift drops empty system
    // messages and could not round-trip it as one.
    let Some(default_instructions) = default_instructions.filter(|d| !d.is_empty()) else {
        return request;
    };

    let existing = request
        .chat_history
        .iter()
        .filter_map(|message| match message {
            completion::Message::System { content } => Some(content.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    // A request already carrying the
    // default is left alone rather than repeating it.
    if !existing.trim().is_empty() && existing.contains(default_instructions) {
        return request;
    }

    let mut request = request;
    // `OneOrMany::insert` keeps the non-empty invariant by construction — no
    // rebuild through the fallible `many` constructor.
    request
        .chat_history
        .insert(0, completion::Message::system(default_instructions));
    request
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OneOrMany;

    #[test]
    fn test_parse_chatgpt_sse_completion() {
        let body = r#"data: {"type":"response.output_text.delta","delta":"hi"}
data: {"type":"response.completed","response":{"id":"resp_1","object":"response","created_at":1,"status":"completed","error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"gpt-5","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","annotations":[],"text":"hi"}]}],"tools":[]}}
data: [DONE]"#;

        let response = responses_api::streaming::parse_sse_completion_body(body, "ChatGPT")
            .expect("expected response");
        assert_eq!(response.id, "resp_1");
        assert_eq!(response.model, "gpt-5");
    }

    /// The default is materialized as a leading system message on the request
    /// itself, so telemetry and the wire body cannot disagree about it. These
    /// pin the same three rules the deleted post-conversion merge had.
    fn instructions_of(request: completion::CompletionRequest) -> String {
        build_codex_responses_request(
            GPT_5_3_CODEX.to_string(),
            &[],
            false,
            Some(DEFAULT_INSTRUCTIONS),
            request,
        )
        .expect("request")
        .instructions
        .expect("instructions")
    }

    #[test]
    fn default_instructions_used_when_request_has_none() {
        let request = completion::CompletionRequest::from_prompt("hi");
        assert_eq!(instructions_of(request), DEFAULT_INSTRUCTIONS);
    }

    #[test]
    fn default_instructions_precede_request_instructions() {
        let request = completion::CompletionRequest::builder("hi")
            .preamble("Respond tersely.")
            .build();
        let merged = instructions_of(request);
        assert!(merged.starts_with(DEFAULT_INSTRUCTIONS));
        assert!(merged.ends_with("Respond tersely."));
    }

    #[test]
    fn default_instructions_are_not_duplicated() {
        let request = completion::CompletionRequest::builder("hi")
            .preamble("You are ChatGPT, a helpful AI assistant.\n\nRespond tersely.")
            .build();
        assert_eq!(
            instructions_of(request),
            "You are ChatGPT, a helpful AI assistant.\n\nRespond tersely."
        );
    }

    /// The request conversion the deleted `ResponsesCompletionModel` reached
    /// through `openai_model().create_completion_request(...)`; the shaping
    /// now lives entirely in `build_codex_responses_request`.
    fn chatgpt_conversion_request(
        chat_history: OneOrMany<completion::Message>,
    ) -> ResponsesRequest {
        build_codex_responses_request(
            GPT_5_3_CODEX.to_string(),
            &[],
            false,
            None,
            completion::CompletionRequest {
                model: Some("gpt-5.4".to_string()),
                chat_history,
                documents: Vec::new(),
                tools: Vec::new(),
                temperature: None,
                max_tokens: None,
                tool_choice: None,
                additional_params: None,
                output_schema: None,
                record_telemetry_content: false,
            },
        )
        .expect("request")
    }

    #[test]
    fn test_conversion_lifts_leading_system_messages_into_instructions() {
        let request = chatgpt_conversion_request(
            OneOrMany::many(vec![
                completion::Message::system("System one"),
                completion::Message::system("System two"),
                completion::Message::user("hi"),
            ])
            .expect("history"),
        );

        assert_eq!(
            request.instructions.as_deref(),
            Some("System one\n\nSystem two")
        );
        assert_eq!(request.input.len(), 1);
    }

    #[test]
    fn test_conversion_lifts_mid_conversation_system_messages() {
        let request = chatgpt_conversion_request(
            OneOrMany::many(vec![
                completion::Message::system("System one"),
                completion::Message::user("hi"),
                completion::Message::system("Mid-conversation instruction"),
                completion::Message::user("again"),
            ])
            .expect("history"),
        );

        assert_eq!(
            request.instructions.as_deref(),
            Some("System one\n\nMid-conversation instruction")
        );
        assert_eq!(request.input.len(), 2);
    }

    #[test]
    fn test_create_request_merges_default_and_request_instructions() {
        let request = build_codex_responses_request(
            GPT_5_3_CODEX.to_string(),
            &[],
            false,
            Some(DEFAULT_INSTRUCTIONS),
            completion::CompletionRequest {
                record_telemetry_content: false,
                model: None,
                chat_history: OneOrMany::many(vec![
                    crate::message::Message::system("Respond tersely.".to_string()),
                    completion::Message::user("hello"),
                ])
                .expect("non-empty"),
                documents: Vec::new(),
                tools: Vec::new(),
                temperature: None,
                max_tokens: None,
                tool_choice: None,
                additional_params: None,
                output_schema: None,
            },
        )
        .expect("request");

        let expected = format!("{DEFAULT_INSTRUCTIONS}\n\nRespond tersely.");
        assert_eq!(request.instructions.as_deref(), Some(expected.as_str()));
    }

    #[test]
    fn test_create_request_drops_temperature() {
        let request = build_codex_responses_request(
            GPT_5_3_CODEX.to_string(),
            &[],
            false,
            None,
            completion::CompletionRequest {
                model: None,
                chat_history: OneOrMany::one(completion::Message::user("hello")),
                documents: Vec::new(),
                tools: Vec::new(),
                temperature: Some(0.5),
                max_tokens: None,
                tool_choice: None,
                additional_params: None,
                output_schema: None,
                record_telemetry_content: false,
            },
        )
        .expect("request");

        assert!(request.temperature.is_none());
    }

    #[tokio::test]
    async fn test_completion_response_from_sse_body_falls_back_to_streamed_text() {
        let body = r#"data: {"type":"response.output_text.delta","delta":"hi"}
data: {"type":"response.completed","response":{"id":"resp_1","object":"response","created_at":1,"status":"completed","error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"gpt-5","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"output":[],"tools":[]}}
data: [DONE]"#;

        let raw_response = responses_api::streaming::parse_sse_completion_body(body, "ChatGPT")
            .expect("expected response");
        let response =
            responses_api::streaming::completion_response_from_sse_body(body, raw_response)
                .await
                .expect("fallback response");

        let text: String = response
            .choice
            .iter()
            .filter_map(|content| match content {
                completion::AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();

        assert_eq!(text, "hi");
        assert_eq!(response.usage.total_tokens, 2);
    }

    #[tokio::test]
    async fn completion_http_non_success_preserves_status_and_body() {
        use crate::http_runtime::HttpRuntime;
        use crate::test_utils::RecordingHttpClient;

        let cases = [
            (
                http::StatusCode::UNAUTHORIZED,
                r#"{"error":{"message":"expired access token","type":"invalid_request_error"}}"#,
                "expired access token",
            ),
            (
                http::StatusCode::TOO_MANY_REQUESTS,
                r#"{"error":{"message":"rate limited","type":"rate_limit_error"}}"#,
                "rate limited",
            ),
        ];

        for (status, body, message) in cases {
            let rt = HttpRuntime::recording(RecordingHttpClient::with_error_response(status, body));
            let cfg = functions::Config::new(GPT_5_4)
                .with_access_token("test-token")
                .with_account_id("account-id");
            let request = crate::completion::CompletionRequest::from_prompt("hello");

            let error = functions::complete(&cfg, &rt, request)
                .await
                .expect_err("completion should fail with non-success status");

            assert_eq!(error.provider_response_status(), Some(status));
            assert_eq!(error.provider_response_body(), Some(body));
            assert!(
                error.to_string().contains(message),
                "error should include provider body: {error}"
            );
        }
    }
}
