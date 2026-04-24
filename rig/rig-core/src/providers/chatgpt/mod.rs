//! ChatGPT subscription OAuth provider.
//!
//! This provider targets the ChatGPT subscription backend exposed at
//! `https://chatgpt.com/backend-api/codex`.
//!
//! # Example
//! ```no_run
//! use rig::client::{CompletionClient, ProviderClient};
//! use rig::providers::chatgpt;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = chatgpt::Client::from_env()?;
//! let model = client.completion_model(chatgpt::GPT_5_3_CODEX);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

mod auth;

use crate::OneOrMany;
use crate::client::{
    self, ApiKey, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
    ProviderClient, Transport,
};
use crate::completion::{self, CompletionError};
use crate::http_client::{self, HttpClientExt};
use crate::providers::openai::responses_api::{
    self, CompletionRequest as ResponsesRequest, Include,
};
use crate::streaming::StreamingCompletionResponse;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use tracing::{Level, enabled, info_span};

const CHATGPT_API_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const DEFAULT_ORIGINATOR: &str = "rig";
const DEFAULT_INSTRUCTIONS: &str = "You are ChatGPT, a helpful AI assistant.";

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

#[derive(Clone)]
pub enum ChatGPTAuth {
    AccessToken {
        access_token: String,
        account_id: Option<String>,
    },
    OAuth,
}

impl ApiKey for ChatGPTAuth {}

impl<S> From<S> for ChatGPTAuth
where
    S: Into<String>,
{
    fn from(value: S) -> Self {
        Self::AccessToken {
            access_token: value.into(),
            account_id: None,
        }
    }
}

impl Debug for ChatGPTAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AccessToken { .. } => f.write_str("AccessToken(<redacted>)"),
            Self::OAuth => f.write_str("OAuth"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatGPTBuilder {
    auth_file: Option<PathBuf>,
    default_instructions: Option<String>,
    device_code_handler: auth::DeviceCodeHandler,
    originator: String,
    user_agent: Option<String>,
}

#[derive(Clone)]
pub struct ChatGPTExt {
    auth: auth::Authenticator,
    default_instructions: Option<String>,
    originator: String,
    user_agent: String,
}

impl Debug for ChatGPTExt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatGPTExt")
            .field("auth", &self.auth)
            .field("default_instructions", &self.default_instructions)
            .field("originator", &self.originator)
            .field("user_agent", &self.user_agent)
            .finish()
    }
}

pub type Client<H = reqwest::Client> = client::Client<ChatGPTExt, H>;
pub type ClientBuilder<H = reqwest::Client> = client::ClientBuilder<ChatGPTBuilder, ChatGPTAuth, H>;

impl Default for ChatGPTBuilder {
    fn default() -> Self {
        Self {
            auth_file: default_auth_file(),
            default_instructions: Some(
                std::env::var("CHATGPT_DEFAULT_INSTRUCTIONS")
                    .ok()
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or_else(|| DEFAULT_INSTRUCTIONS.to_string()),
            ),
            device_code_handler: auth::DeviceCodeHandler::default(),
            originator: std::env::var("CHATGPT_ORIGINATOR")
                .ok()
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| DEFAULT_ORIGINATOR.to_string()),
            user_agent: std::env::var("CHATGPT_USER_AGENT")
                .ok()
                .filter(|value| !value.is_empty()),
        }
    }
}

impl Provider for ChatGPTExt {
    type Builder = ChatGPTBuilder;

    const VERIFY_PATH: &'static str = "";

    fn with_custom(&self, req: http_client::Builder) -> http_client::Result<http_client::Builder> {
        Ok(req
            .header("originator", &self.originator)
            .header("user-agent", &self.user_agent)
            .header(http::header::ACCEPT, "text/event-stream"))
    }

    fn build_uri(&self, base_url: &str, path: &str, _transport: Transport) -> String {
        format!(
            "{}/{}",
            base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    }
}

impl<H> Capabilities<H> for ChatGPTExt {
    type Completion = Capable<ResponsesCompletionModel<H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for ChatGPTExt {}

impl ProviderBuilder for ChatGPTBuilder {
    type Extension<H>
        = ChatGPTExt
    where
        H: HttpClientExt;
    type ApiKey = ChatGPTAuth;

    const BASE_URL: &'static str = CHATGPT_API_BASE_URL;

    fn build<H>(
        builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        let auth = match builder.get_api_key() {
            ChatGPTAuth::AccessToken {
                access_token,
                account_id,
            } => auth::AuthSource::AccessToken {
                access_token: access_token.clone(),
                account_id: account_id.clone(),
            },
            ChatGPTAuth::OAuth => auth::AuthSource::OAuth,
        };

        let ext = builder.ext();

        Ok(ChatGPTExt {
            auth: auth::Authenticator::new(
                auth,
                ext.auth_file.clone(),
                ext.device_code_handler.clone(),
            ),
            default_instructions: ext.default_instructions.clone(),
            originator: ext.originator.clone(),
            user_agent: ext.user_agent.clone().unwrap_or_else(default_user_agent),
        })
    }
}

impl ProviderClient for Client {
    type Input = ChatGPTAuth;
    type Error = crate::client::ProviderClientError;

    fn from_env() -> Result<Self, Self::Error> {
        let mut builder = Self::builder();

        if let Some(base_url) = crate::client::optional_env_var("CHATGPT_API_BASE")?
            .or(crate::client::optional_env_var("OPENAI_CHATGPT_API_BASE")?)
        {
            builder = builder.base_url(base_url);
        }

        if let Some(access_token) = crate::client::optional_env_var("CHATGPT_ACCESS_TOKEN")? {
            let account_id = crate::client::optional_env_var("CHATGPT_ACCOUNT_ID")?;
            builder
                .api_key(ChatGPTAuth::AccessToken {
                    access_token,
                    account_id,
                })
                .build()
                .map_err(Into::into)
        } else {
            builder.oauth().build().map_err(Into::into)
        }
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::builder().api_key(input).build().map_err(Into::into)
    }
}

impl<H> client::ClientBuilder<ChatGPTBuilder, crate::markers::Missing, H> {
    pub fn oauth(self) -> client::ClientBuilder<ChatGPTBuilder, ChatGPTAuth, H> {
        self.api_key(ChatGPTAuth::OAuth)
    }
}

impl<H> ClientBuilder<H> {
    pub fn on_device_code<F>(self, handler: F) -> Self
    where
        F: Fn(auth::DeviceCodePrompt) + Send + Sync + 'static,
    {
        self.over_ext(|mut ext| {
            ext.device_code_handler = auth::DeviceCodeHandler::new(handler);
            ext
        })
    }

    pub fn token_dir(self, path: impl AsRef<Path>) -> Self {
        let auth_file = path.as_ref().join("auth.json");
        self.over_ext(|mut ext| {
            ext.auth_file = Some(auth_file);
            ext
        })
    }

    pub fn auth_file(self, path: impl AsRef<Path>) -> Self {
        let auth_file = path.as_ref().to_path_buf();
        self.over_ext(|mut ext| {
            ext.auth_file = Some(auth_file);
            ext
        })
    }

    pub fn default_instructions(self, instructions: impl Into<String>) -> Self {
        let instructions = instructions.into();
        self.over_ext(|mut ext| {
            ext.default_instructions = Some(instructions);
            ext
        })
    }

    pub fn originator(self, originator: impl Into<String>) -> Self {
        let originator = originator.into();
        self.over_ext(|mut ext| {
            ext.originator = originator;
            ext
        })
    }

    pub fn user_agent(self, user_agent: impl Into<String>) -> Self {
        let user_agent = user_agent.into();
        self.over_ext(|mut ext| {
            ext.user_agent = Some(user_agent);
            ext
        })
    }
}

#[derive(Clone)]
pub struct ResponsesCompletionModel<H = reqwest::Client> {
    client: Client<H>,
    pub model: String,
    pub tools: Vec<responses_api::ResponsesToolDefinition>,
}

impl<H> ResponsesCompletionModel<H>
where
    Client<H>: HttpClientExt + Clone + Debug + 'static,
    H: Clone + Default + Debug + WasmCompatSend + WasmCompatSync + 'static,
{
    pub fn new(client: Client<H>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            tools: Vec::new(),
        }
    }

    pub fn with_tool(mut self, tool: impl Into<responses_api::ResponsesToolDefinition>) -> Self {
        self.tools.push(tool.into());
        self
    }

    pub fn with_tools<I, Tool>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = Tool>,
        Tool: Into<responses_api::ResponsesToolDefinition>,
    {
        self.tools.extend(tools.into_iter().map(Into::into));
        self
    }

    fn openai_model(&self) -> responses_api::GenericResponsesCompletionModel<ChatGPTExt, H> {
        let mut model = responses_api::GenericResponsesCompletionModel::new(
            self.client.clone(),
            self.model.clone(),
        );
        model.tools = self.tools.clone();
        model
    }

    fn create_request(
        &self,
        request: completion::CompletionRequest,
    ) -> Result<ResponsesRequest, CompletionError> {
        let mut request = self.openai_model().create_completion_request(request)?;

        if let Some(system_instructions) =
            normalize_system_messages_into_instructions(&mut request)?
        {
            request.instructions = Some(match request.instructions.as_deref() {
                Some(existing) if !existing.trim().is_empty() => {
                    format!("{system_instructions}\n\n{existing}")
                }
                _ => system_instructions,
            });
        }

        if let Some(default_instructions) = &self.client.ext().default_instructions {
            request.instructions = Some(merge_instructions(
                default_instructions,
                request.instructions.as_deref(),
            ));
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

    fn add_auth_headers(
        &self,
        req: http_client::Builder,
        context: &auth::AuthContext,
    ) -> http_client::Builder {
        let req = req
            .header(
                http::header::AUTHORIZATION,
                format!("Bearer {}", context.access_token),
            )
            .header("session_id", nanoid::nanoid!());

        if let Some(account_id) = &context.account_id {
            req.header("ChatGPT-Account-Id", account_id)
        } else {
            req
        }
    }

    async fn completion_from_sse(
        &self,
        request: ResponsesRequest,
    ) -> Result<completion::CompletionResponse<responses_api::CompletionResponse>, CompletionError>
    {
        let body = serde_json::to_vec(&request)?;
        let auth = self
            .client
            .ext()
            .auth
            .auth_context()
            .await
            .map_err(|err| CompletionError::ProviderError(err.to_string()))?;

        let req = self
            .add_auth_headers(self.client.post("/responses")?, &auth)
            .body(body)
            .map_err(|err| CompletionError::HttpError(err.into()))?;

        let response = self.client.send(req).await?;
        let text = http_client::text(response).await?;
        let raw_response = responses_api::streaming::parse_sse_completion_body(&text, "ChatGPT")?;

        match raw_response.clone().try_into() {
            Ok(response) => Ok(response),
            Err(CompletionError::ResponseError(message))
                if message == "Response contained no parts" =>
            {
                responses_api::streaming::completion_response_from_sse_body(
                    &text,
                    raw_response,
                    "ChatGPT",
                )
                .await
            }
            Err(error) => Err(error),
        }
    }
}

impl<H> Client<H>
where
    H: HttpClientExt + Clone + Debug + Default + WasmCompatSend + WasmCompatSync + 'static,
{
    pub async fn authorize(&self) -> Result<(), auth::AuthError> {
        self.ext().auth.auth_context().await.map(|_| ())
    }
}

impl<H> completion::CompletionModel for ResponsesCompletionModel<H>
where
    Client<H>: HttpClientExt + Clone + Debug + 'static,
    H: Clone + Default + Debug + WasmCompatSend + WasmCompatSync + 'static,
{
    type Response = responses_api::CompletionResponse;
    type StreamingResponse = responses_api::streaming::StreamingCompletionResponse;
    type Client = Client<H>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<Self::Response>, CompletionError> {
        let request = self.create_request(completion_request)?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "chatgpt",
                gen_ai.request.model = self.model,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = tracing::field::Empty,
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        tracing_futures::Instrument::instrument(
            async move {
                let response = self.completion_from_sse(request).await?;
                let span = tracing::Span::current();
                span.record("gen_ai.response.id", &response.raw_response.id);
                span.record("gen_ai.response.model", &response.raw_response.model);
                span.record("gen_ai.usage.output_tokens", response.usage.output_tokens);
                span.record("gen_ai.usage.input_tokens", response.usage.input_tokens);
                span.record(
                    "gen_ai.usage.cache_read.input_tokens",
                    response.usage.cached_input_tokens,
                );
                Ok(response)
            },
            span,
        )
        .await
    }

    async fn stream(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        Self::stream(self, completion_request).await
    }
}

impl<H> ResponsesCompletionModel<H>
where
    Client<H>: HttpClientExt + Clone + Debug + 'static,
    H: Clone + Default + Debug + WasmCompatSend + WasmCompatSync + 'static,
{
    pub async fn stream(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<
        StreamingCompletionResponse<responses_api::streaming::StreamingCompletionResponse>,
        CompletionError,
    > {
        let request = self.create_request(completion_request)?;

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "ChatGPT Responses streaming completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let auth = self
            .client
            .ext()
            .auth
            .auth_context()
            .await
            .map_err(|err| CompletionError::ProviderError(err.to_string()))?;

        let req = self
            .add_auth_headers(self.client.post("/responses")?, &auth)
            .body(body)
            .map_err(|err| CompletionError::HttpError(err.into()))?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "chatgpt",
                gen_ai.request.model = self.model,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let client = self.client.clone();
        let event_source = crate::http_client::sse::GenericEventSource::new(client, req)
            .allow_missing_content_type();

        Ok(responses_api::streaming::stream_from_event_source(
            event_source,
            span,
            "ChatGPT",
        ))
    }
}

fn default_user_agent() -> String {
    format!(
        "rig/{} ({} {}; {})",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
        DEFAULT_ORIGINATOR
    )
}

fn default_auth_file() -> Option<PathBuf> {
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

fn normalize_system_messages_into_instructions(
    request: &mut ResponsesRequest,
) -> Result<Option<String>, CompletionError> {
    let mut system_instructions = Vec::new();
    let mut filtered_items = Vec::new();

    for item in request.input.clone() {
        if let Some(system_text) = item.system_text() {
            let system_text = system_text.trim();
            if !system_text.is_empty() {
                system_instructions.push(system_text.to_string());
            }
        } else {
            filtered_items.push(item);
        }
    }

    request.input = OneOrMany::many(filtered_items).map_err(|_| {
        CompletionError::RequestError(
            "ChatGPT responses request input must contain at least one non-system item".into(),
        )
    })?;

    if system_instructions.is_empty() {
        Ok(None)
    } else {
        Ok(Some(system_instructions.join("\n\n")))
    }
}

fn merge_instructions(default_instructions: &str, existing_instructions: Option<&str>) -> String {
    match existing_instructions
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(existing) if existing.contains(default_instructions) => existing.to_string(),
        Some(existing) => format!("{default_instructions}\n\n{existing}"),
        None => default_instructions.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_client_initialization() {
        let _client = crate::providers::chatgpt::Client::builder()
            .oauth()
            .build()
            .expect("Client::builder()");
    }

    #[test]
    fn test_merge_instructions_uses_default_when_missing() {
        assert_eq!(
            merge_instructions(DEFAULT_INSTRUCTIONS, None),
            DEFAULT_INSTRUCTIONS
        );
    }

    #[test]
    fn test_merge_instructions_appends_existing_request_instructions() {
        let merged = merge_instructions(DEFAULT_INSTRUCTIONS, Some("Respond tersely."));
        assert!(merged.starts_with(DEFAULT_INSTRUCTIONS));
        assert!(merged.ends_with("Respond tersely."));
    }

    #[test]
    fn test_merge_instructions_avoids_duplicate_default() {
        let merged = merge_instructions(
            DEFAULT_INSTRUCTIONS,
            Some("You are ChatGPT, a helpful AI assistant.\n\nRespond tersely."),
        );
        assert_eq!(
            merged,
            "You are ChatGPT, a helpful AI assistant.\n\nRespond tersely."
        );
    }

    #[test]
    fn test_normalize_system_messages_into_instructions() {
        let completion_request = completion::CompletionRequest {
            model: Some("gpt-5.4".to_string()),
            preamble: Some("System one".to_string()),
            chat_history: OneOrMany::many(vec![
                completion::Message::system("System two"),
                completion::Message::user("hi"),
            ])
            .expect("history"),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        };
        let mut request = ResponsesRequest::try_from(("gpt-5.4".to_string(), completion_request))
            .expect("request");

        let instructions = normalize_system_messages_into_instructions(&mut request)
            .expect("normalize")
            .expect("instructions");

        assert_eq!(instructions, "System one\n\nSystem two");
        assert_eq!(request.input.len(), 1);
    }

    #[test]
    fn test_create_request_drops_temperature() {
        let client = crate::providers::chatgpt::Client::builder()
            .oauth()
            .build()
            .expect("client");
        let model = ResponsesCompletionModel::new(client, GPT_5_3_CODEX);

        let request = model
            .create_request(completion::CompletionRequest {
                model: None,
                preamble: None,
                chat_history: OneOrMany::one(completion::Message::user("hello")),
                documents: Vec::new(),
                tools: Vec::new(),
                temperature: Some(0.5),
                max_tokens: None,
                tool_choice: None,
                additional_params: None,
                output_schema: None,
            })
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
        let response = responses_api::streaming::completion_response_from_sse_body(
            body,
            raw_response,
            "ChatGPT",
        )
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
}
