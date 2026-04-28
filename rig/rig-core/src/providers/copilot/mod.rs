//! GitHub Copilot provider.
//!
//! Supports Chat Completions, Responses, and Embeddings against
//! `https://api.githubcopilot.com`.
//!
//! `Client::completion_model(...)` automatically routes Codex-class models
//! through `/responses` and conversational models through
//! `/chat/completions`.
//!
//! # Example
//! ```no_run
//! use rig::client::{CompletionClient, ProviderClient};
//! use rig::providers::copilot;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = copilot::Client::from_env()?;
//! let model = client.completion_model(copilot::GPT_4O);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

mod auth;

use crate::client::{
    self, ApiKey, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
    ProviderClient, Transport,
};
use crate::completion::{self, CompletionError, GetTokenUsage};
use crate::embeddings::{self, EmbeddingError};
use crate::http_client::{self, HttpClientExt};
use crate::providers::internal::openai_chat_completions_compatible::{
    self, CompatibleChoiceData, CompatibleChunk, CompatibleFinishReason, CompatibleStreamProfile,
    CompatibleToolCallChunk,
};
use crate::providers::openai;
use crate::providers::openai::responses_api::{self, CompletionRequest as ResponsesRequest};
use crate::streaming::{self, RawStreamingChoice, StreamingCompletionResponse};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use async_stream::stream;
use futures::StreamExt;
use http::Request;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use tracing::info_span;
use tracing_futures::Instrument as _;

const GITHUB_COPILOT_API_BASE_URL: &str = "https://api.githubcopilot.com";
const EDITOR_PLUGIN_VERSION: &str = "copilot-chat/0.26.7";
const USER_AGENT: &str = "GitHubCopilotChat/0.26.7";
const API_VERSION: &str = "2025-04-01";

/// `gpt-4`
pub const GPT_4: &str = "gpt-4";
/// `gpt-4o`
pub const GPT_4O: &str = "gpt-4o";
/// `gpt-4o-mini`
pub const GPT_4O_MINI: &str = "gpt-4o-mini";
/// `gpt-4.1`
pub const GPT_4_1: &str = "gpt-4.1";
/// `gpt-4.1-mini`
pub const GPT_4_1_MINI: &str = "gpt-4.1-mini";
/// `gpt-4.1-nano`
pub const GPT_4_1_NANO: &str = "gpt-4.1-nano";
/// `gpt-5.3-codex`
pub const GPT_5_3_CODEX: &str = "gpt-5.3-codex";
/// `gpt-5.1-codex`
pub const GPT_5_1_CODEX: &str = "gpt-5.1-codex";
/// `claude-sonnet-4` completion model (Anthropic, via Copilot)
pub const CLAUDE_SONNET_4: &str = "claude-sonnet-4";
/// `claude-3.5-sonnet` completion model (Anthropic, via Copilot)
pub const CLAUDE_3_5_SONNET: &str = "claude-3.5-sonnet";
/// `gemini-2.0-flash-001` completion model (Google, via Copilot)
pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash-001";
/// `o3-mini` reasoning model (OpenAI, via Copilot)
pub const O3_MINI: &str = "o3-mini";
/// `text-embedding-3-small`
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// `text-embedding-3-large`
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
/// `text-embedding-ada-002`
pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

pub use openai::EncodingFormat;

#[derive(Clone)]
pub enum CopilotAuth {
    ApiKey(String),
    GitHubAccessToken(String),
    OAuth,
}

impl ApiKey for CopilotAuth {}

impl<S> From<S> for CopilotAuth
where
    S: Into<String>,
{
    fn from(value: S) -> Self {
        Self::ApiKey(value.into())
    }
}

impl Debug for CopilotAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiKey(_) => f.write_str("ApiKey(<redacted>)"),
            Self::GitHubAccessToken(_) => f.write_str("GitHubAccessToken(<redacted>)"),
            Self::OAuth => f.write_str("OAuth"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CopilotBuilder {
    access_token_file: Option<PathBuf>,
    api_key_file: Option<PathBuf>,
    device_code_handler: auth::DeviceCodeHandler,
}

#[derive(Clone)]
pub struct CopilotExt {
    auth: auth::Authenticator,
}

impl Debug for CopilotExt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CopilotExt")
            .field("auth", &self.auth)
            .finish()
    }
}

pub type Client<H = reqwest::Client> = client::Client<CopilotExt, H>;
pub type ClientBuilder<H = reqwest::Client> = client::ClientBuilder<CopilotBuilder, CopilotAuth, H>;

impl Default for CopilotBuilder {
    fn default() -> Self {
        let token_dir = default_token_dir();
        Self {
            access_token_file: token_dir.as_ref().map(|dir| dir.join("access-token")),
            api_key_file: token_dir.map(|dir| dir.join("api-key.json")),
            device_code_handler: auth::DeviceCodeHandler::default(),
        }
    }
}

impl Provider for CopilotExt {
    type Builder = CopilotBuilder;

    const VERIFY_PATH: &'static str = "";
}

impl<H> Capabilities<H> for CopilotExt {
    type Completion = Capable<CompletionModel<H>>;
    type Embeddings = Capable<EmbeddingModel<H>>;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for CopilotExt {}

impl ProviderBuilder for CopilotBuilder {
    type Extension<H>
        = CopilotExt
    where
        H: HttpClientExt;
    type ApiKey = CopilotAuth;

    const BASE_URL: &'static str = GITHUB_COPILOT_API_BASE_URL;

    fn build<H>(
        builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        let auth = match builder.get_api_key() {
            CopilotAuth::ApiKey(api_key) => auth::AuthSource::ApiKey(api_key.clone()),
            CopilotAuth::GitHubAccessToken(access_token) => {
                auth::AuthSource::GitHubAccessToken(access_token.clone())
            }
            CopilotAuth::OAuth => auth::AuthSource::OAuth,
        };

        let ext = builder.ext();
        Ok(CopilotExt {
            auth: auth::Authenticator::new(
                auth,
                ext.access_token_file.clone(),
                ext.api_key_file.clone(),
                ext.device_code_handler.clone(),
            ),
        })
    }
}

impl ProviderClient for Client {
    type Input = CopilotAuth;
    type Error = crate::client::ProviderClientError;

    fn from_env() -> Result<Self, Self::Error> {
        let mut builder = Self::builder();
        fn get(name: &str) -> Option<String> {
            std::env::var(name).ok()
        }

        if let Some(base_url) = env_base_url(&get) {
            builder = builder.base_url(base_url);
        }

        if let Some(api_key) = env_api_key(&get) {
            builder.api_key(api_key).build().map_err(Into::into)
        } else if let Some(access_token) = env_github_access_token(&get) {
            builder
                .github_access_token(access_token)
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

impl<H> client::ClientBuilder<CopilotBuilder, crate::markers::Missing, H> {
    pub fn github_access_token(
        self,
        access_token: impl Into<String>,
    ) -> client::ClientBuilder<CopilotBuilder, CopilotAuth, H> {
        self.api_key(CopilotAuth::GitHubAccessToken(access_token.into()))
    }

    pub fn oauth(self) -> client::ClientBuilder<CopilotBuilder, CopilotAuth, H> {
        self.api_key(CopilotAuth::OAuth)
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
        let path = path.as_ref();
        self.over_ext(|mut ext| {
            ext.access_token_file = Some(path.join("access-token"));
            ext.api_key_file = Some(path.join("api-key.json"));
            ext
        })
    }

    pub fn access_token_file(self, path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_path_buf();
        self.over_ext(|mut ext| {
            ext.access_token_file = Some(path);
            ext
        })
    }

    pub fn api_key_file(self, path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_path_buf();
        self.over_ext(|mut ext| {
            ext.api_key_file = Some(path);
            ext
        })
    }
}

fn env_value<F>(get: &F, name: &str) -> Option<String>
where
    F: Fn(&str) -> Option<String>,
{
    get(name).filter(|value| !value.trim().is_empty())
}

fn first_env_value<F>(get: &F, keys: &[&str]) -> Option<String>
where
    F: Fn(&str) -> Option<String>,
{
    keys.iter().find_map(|key| env_value(get, key))
}

fn env_api_key<F>(get: &F) -> Option<String>
where
    F: Fn(&str) -> Option<String>,
{
    first_env_value(get, &["GITHUB_COPILOT_API_KEY", "COPILOT_API_KEY"])
}

fn env_github_access_token<F>(get: &F) -> Option<String>
where
    F: Fn(&str) -> Option<String>,
{
    first_env_value(get, &["COPILOT_GITHUB_ACCESS_TOKEN", "GITHUB_TOKEN"])
}

fn env_base_url<F>(get: &F) -> Option<String>
where
    F: Fn(&str) -> Option<String>,
{
    first_env_value(get, &["GITHUB_COPILOT_API_BASE", "COPILOT_BASE_URL"])
}

impl<H> Client<H>
where
    H: HttpClientExt + Clone + Debug + Default + WasmCompatSend + WasmCompatSync + 'static,
{
    pub async fn authorize(&self) -> Result<(), auth::AuthError> {
        self.ext().auth.auth_context().await.map(|_| ())
    }
}

fn default_headers(
    api_key: &str,
    initiator: &'static str,
    has_vision: bool,
) -> Vec<(&'static str, String)> {
    let mut headers = vec![
        (
            http::header::AUTHORIZATION.as_str(),
            format!("Bearer {api_key}"),
        ),
        ("copilot-integration-id", "vscode-chat".to_string()),
        ("editor-version", "vscode/1.95.0".to_string()),
        ("editor-plugin-version", EDITOR_PLUGIN_VERSION.to_string()),
        ("user-agent", USER_AGENT.to_string()),
        ("openai-intent", "conversation-panel".to_string()),
        ("x-github-api-version", API_VERSION.to_string()),
        ("x-request-id", nanoid::nanoid!()),
        (
            "x-vscode-user-agent-library-version",
            "electron-fetch".to_string(),
        ),
        ("X-Initiator", initiator.to_string()),
    ];

    if has_vision {
        headers.push(("copilot-vision-request", "true".to_string()));
    }

    headers
}

fn apply_headers(
    builder: http_client::Builder,
    headers: &[(&'static str, String)],
) -> http_client::Builder {
    headers
        .iter()
        .fold(builder, |builder, (key, value)| builder.header(*key, value))
}

fn runtime_base_url<'a, H>(client: &'a Client<H>, auth: &'a auth::AuthContext) -> Cow<'a, str> {
    if client.base_url() == GITHUB_COPILOT_API_BASE_URL {
        auth.api_base
            .as_deref()
            .map(Cow::Borrowed)
            .unwrap_or_else(|| Cow::Borrowed(client.base_url()))
    } else {
        Cow::Borrowed(client.base_url())
    }
}

fn post_with_auth_base<H>(
    client: &Client<H>,
    auth: &auth::AuthContext,
    path: &str,
    transport: Transport,
) -> http_client::Result<http_client::Builder> {
    let uri = client
        .ext()
        .build_uri(runtime_base_url(client, auth).as_ref(), path, transport);
    let mut req = Request::post(uri);

    if let Some(headers) = req.headers_mut() {
        headers.extend(client.headers().iter().map(|(k, v)| (k.clone(), v.clone())));
    }

    client.ext().with_custom(req)
}

fn request_initiator(request: &completion::CompletionRequest) -> &'static str {
    for message in request.chat_history.iter() {
        match message {
            crate::completion::Message::Assistant { .. } => return "agent",
            crate::completion::Message::User { content } => {
                if content
                    .iter()
                    .any(|item| matches!(item, crate::message::UserContent::ToolResult(_)))
                {
                    return "agent";
                }
            }
            crate::completion::Message::System { .. } => {}
        }
    }

    "user"
}

fn request_has_vision(request: &completion::CompletionRequest) -> bool {
    request.chat_history.iter().any(|message| match message {
        crate::completion::Message::User { content } => content
            .iter()
            .any(|item| matches!(item, crate::message::UserContent::Image(_))),
        _ => false,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompletionRoute {
    ChatCompletions,
    Responses,
}

fn route_for_model(model: &str) -> CompletionRoute {
    if model.to_ascii_lowercase().contains("codex") {
        CompletionRoute::Responses
    } else {
        CompletionRoute::ChatCompletions
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "api", rename_all = "snake_case")]
pub enum CopilotCompletionResponse {
    Chat(ChatCompletionResponse),
    Responses(Box<responses_api::CompletionResponse>),
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "api", rename_all = "snake_case")]
pub enum CopilotStreamingResponse {
    Chat(openai::completion::streaming::StreamingCompletionResponse),
    Responses(responses_api::streaming::StreamingCompletionResponse),
}

impl GetTokenUsage for CopilotStreamingResponse {
    fn token_usage(&self) -> Option<completion::Usage> {
        match self {
            Self::Chat(response) => response.token_usage(),
            Self::Responses(response) => response.token_usage(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub created: Option<u64>,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChatChoice>,
    pub usage: Option<openai::completion::Usage>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatChoice {
    #[serde(default)]
    pub index: usize,
    pub message: openai::completion::Message,
    pub logprobs: Option<serde_json::Value>,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

impl TryFrom<ChatCompletionResponse> for completion::CompletionResponse<ChatCompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: ChatCompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        let content = match &choice.message {
            openai::completion::Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut content = content
                    .iter()
                    .filter_map(|c| {
                        let s = match c {
                            openai::completion::AssistantContent::Text { text } => text,
                            openai::completion::AssistantContent::Refusal { refusal } => refusal,
                        };
                        if s.is_empty() {
                            None
                        } else {
                            Some(completion::AssistantContent::text(s))
                        }
                    })
                    .collect::<Vec<_>>();

                content.extend(
                    tool_calls
                        .iter()
                        .map(|call| {
                            completion::AssistantContent::tool_call(
                                &call.id,
                                &call.function.name,
                                call.function.arguments.clone(),
                            )
                        })
                        .collect::<Vec<_>>(),
                );
                Ok(content)
            }
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a valid message or tool call".into(),
            )),
        }?;

        let choice = crate::OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        let usage = response
            .usage
            .as_ref()
            .map(|usage| completion::Usage {
                input_tokens: usage.prompt_tokens as u64,
                output_tokens: (usage.total_tokens - usage.prompt_tokens) as u64,
                total_tokens: usage.total_tokens as u64,
                cached_input_tokens: usage
                    .prompt_tokens_details
                    .as_ref()
                    .map(|d| d.cached_tokens as u64)
                    .unwrap_or(0),
                cache_creation_input_tokens: 0,
            })
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
            message_id: None,
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatApiErrorResponse {
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

impl ChatApiErrorResponse {
    pub fn error_message(&self) -> &str {
        self.message
            .as_deref()
            .or(self.error.as_deref())
            .unwrap_or("unknown error")
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ChatApiResponse<T> {
    Ok(T),
    Err(ChatApiErrorResponse),
}

#[derive(Clone)]
pub struct CompletionModel<H = reqwest::Client> {
    client: Client<H>,
    pub model: String,
    pub strict_tools: bool,
    pub tool_result_array_content: bool,
}

impl<H> CompletionModel<H>
where
    Client<H>: HttpClientExt + Clone + Debug + 'static,
    H: Clone + Default + Debug + WasmCompatSend + WasmCompatSync + 'static,
{
    pub fn new(client: Client<H>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            strict_tools: false,
            tool_result_array_content: false,
        }
    }

    pub fn with_strict_tools(mut self) -> Self {
        self.strict_tools = true;
        self
    }

    pub fn with_tool_result_array_content(mut self) -> Self {
        self.tool_result_array_content = true;
        self
    }

    fn route(&self) -> CompletionRoute {
        route_for_model(&self.model)
    }

    async fn auth_context(&self) -> Result<auth::AuthContext, CompletionError> {
        self.client
            .ext()
            .auth
            .auth_context()
            .await
            .map_err(|err| CompletionError::ProviderError(err.to_string()))
    }

    fn chat_request(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<openai::completion::CompletionRequest, CompletionError> {
        openai::completion::CompletionRequest::try_from(openai::completion::OpenAIRequestParams {
            model: self.model.clone(),
            request: completion_request,
            strict_tools: self.strict_tools,
            tool_result_array_content: self.tool_result_array_content,
        })
    }

    fn responses_request(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<ResponsesRequest, CompletionError> {
        ResponsesRequest::try_from((self.model.clone(), completion_request))
    }

    async fn completion_chat(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<CopilotCompletionResponse>, CompletionError> {
        let initiator = request_initiator(&completion_request);
        let has_vision = request_has_vision(&completion_request);
        let request = self.chat_request(completion_request)?;
        let body = serde_json::to_vec(&request)?;
        let auth = self.auth_context().await?;

        let headers = default_headers(&auth.api_key, initiator, has_vision);
        let req = apply_headers(
            post_with_auth_base(&self.client, &auth, "/chat/completions", Transport::Http)?,
            &headers,
        )
        .body(body)
        .map_err(|err| CompletionError::HttpError(err.into()))?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "copilot",
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

        async move {
            let response = self.client.send(req).await?;

            if response.status().is_success() {
                let body = http_client::text(response).await?;
                match serde_json::from_str::<ChatApiResponse<ChatCompletionResponse>>(&body)? {
                    ChatApiResponse::Ok(response) => {
                        let core = completion::CompletionResponse::try_from(response.clone())?;
                        let span = tracing::Span::current();
                        span.record("gen_ai.response.id", response.id.as_str());
                        span.record("gen_ai.response.model", response.model.as_str());
                        if let Some(usage) = &response.usage {
                            span.record("gen_ai.usage.input_tokens", usage.prompt_tokens);
                            span.record(
                                "gen_ai.usage.output_tokens",
                                usage.total_tokens - usage.prompt_tokens,
                            );
                            span.record(
                                "gen_ai.usage.cache_read.input_tokens",
                                usage
                                    .prompt_tokens_details
                                    .as_ref()
                                    .map(|details| details.cached_tokens)
                                    .unwrap_or(0),
                            );
                        }

                        Ok(completion::CompletionResponse {
                            choice: core.choice,
                            usage: core.usage,
                            raw_response: CopilotCompletionResponse::Chat(response),
                            message_id: core.message_id,
                        })
                    }
                    ChatApiResponse::Err(err) => Err(CompletionError::ProviderError(
                        err.error_message().to_string(),
                    )),
                }
            } else {
                let body = http_client::text(response).await?;
                Err(CompletionError::ProviderError(body))
            }
        }
        .instrument(span)
        .await
    }

    async fn completion_responses(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<CopilotCompletionResponse>, CompletionError> {
        let initiator = request_initiator(&completion_request);
        let has_vision = request_has_vision(&completion_request);
        let request = self.responses_request(completion_request)?;
        let auth = self.auth_context().await?;

        let headers = default_headers(&auth.api_key, initiator, has_vision);
        let req = apply_headers(
            post_with_auth_base(&self.client, &auth, "/responses", Transport::Http)?,
            &headers,
        )
        .body(serde_json::to_vec(&request)?)
        .map_err(|err| CompletionError::HttpError(err.into()))?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "copilot",
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

        async move {
            let response = self.client.send(req).await?;
            if response.status().is_success() {
                let body = http_client::text(response).await?;
                let response = serde_json::from_str::<responses_api::CompletionResponse>(&body)?;
                let core = completion::CompletionResponse::try_from(response.clone())?;

                let span = tracing::Span::current();
                span.record("gen_ai.response.id", response.id.as_str());
                span.record("gen_ai.response.model", response.model.as_str());
                if let Some(usage) = &response.usage {
                    span.record("gen_ai.usage.input_tokens", usage.input_tokens);
                    span.record("gen_ai.usage.output_tokens", usage.output_tokens);
                    span.record(
                        "gen_ai.usage.cache_read.input_tokens",
                        usage
                            .input_tokens_details
                            .as_ref()
                            .map(|details| details.cached_tokens)
                            .unwrap_or(0),
                    );
                }

                Ok(completion::CompletionResponse {
                    choice: core.choice,
                    usage: core.usage,
                    raw_response: CopilotCompletionResponse::Responses(Box::new(response)),
                    message_id: core.message_id,
                })
            } else {
                let body = http_client::text(response).await?;
                Err(CompletionError::ProviderError(body))
            }
        }
        .instrument(span)
        .await
    }

    async fn stream_chat(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<StreamingCompletionResponse<CopilotStreamingResponse>, CompletionError> {
        let initiator = request_initiator(&completion_request);
        let has_vision = request_has_vision(&completion_request);
        let request = self.chat_request(completion_request)?;
        let auth = self.auth_context().await?;
        let headers = default_headers(&auth.api_key, initiator, has_vision);
        let mut request_json = serde_json::to_value(&request)?;
        let request_object = request_json.as_object_mut().ok_or_else(|| {
            CompletionError::ResponseError("copilot request body must be a JSON object".into())
        })?;
        request_object.insert("stream".to_owned(), json!(true));
        request_object.insert(
            "stream_options".to_owned(),
            json!({ "include_usage": true }),
        );

        let req = apply_headers(
            post_with_auth_base(&self.client, &auth, "/chat/completions", Transport::Sse)?,
            &headers,
        )
        .body(serde_json::to_vec(&request_json)?)
        .map_err(|err| CompletionError::HttpError(err.into()))?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "copilot",
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

        tracing::Instrument::instrument(
            send_copilot_chat_streaming_request(self.client.clone(), req),
            span,
        )
        .await
    }

    async fn stream_responses(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<StreamingCompletionResponse<CopilotStreamingResponse>, CompletionError> {
        let initiator = request_initiator(&completion_request);
        let has_vision = request_has_vision(&completion_request);
        let mut request = self.responses_request(completion_request)?;
        request.stream = Some(true);
        let auth = self.auth_context().await?;

        let headers = default_headers(&auth.api_key, initiator, has_vision);
        let req = apply_headers(
            post_with_auth_base(&self.client, &auth, "/responses", Transport::Sse)?,
            &headers,
        )
        .body(serde_json::to_vec(&request)?)
        .map_err(|err| CompletionError::HttpError(err.into()))?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "copilot",
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
        let mut event_source = crate::http_client::sse::GenericEventSource::new(client, req);

        let stream = tracing_futures::Instrument::instrument(
            stream! {
                let mut final_usage = responses_api::ResponsesUsage::new();
                let mut tool_calls: Vec<streaming::RawStreamingChoice<CopilotStreamingResponse>> = Vec::new();
                let mut tool_call_internal_ids: HashMap<String, String> = HashMap::new();
                let span = tracing::Span::current();

                let mut terminated_with_error = false;

                while let Some(event_result) = event_source.next().await {
                    match event_result {
                        Ok(crate::http_client::sse::Event::Open) => continue,
                        Ok(crate::http_client::sse::Event::Message(evt)) => {
                            if evt.data.trim().is_empty() {
                                continue;
                            }

                            let Ok(data) = serde_json::from_str::<responses_api::streaming::StreamingCompletionChunk>(&evt.data) else {
                                continue;
                            };

                            if let responses_api::streaming::StreamingCompletionChunk::Delta(chunk) = &data {
                                use responses_api::streaming::{ItemChunkKind, StreamingItemDoneOutput};

                                match &chunk.data {
                                    ItemChunkKind::OutputItemAdded(message) => {
                                        if let StreamingItemDoneOutput { item: responses_api::Output::FunctionCall(func), .. } = message {
                                            let internal_call_id = tool_call_internal_ids
                                                .entry(func.id.clone())
                                                .or_insert_with(|| nanoid::nanoid!())
                                                .clone();
                                            yield Ok(RawStreamingChoice::ToolCallDelta {
                                                id: func.id.clone(),
                                                internal_call_id,
                                                content: streaming::ToolCallDeltaContent::Name(func.name.clone()),
                                            });
                                        }
                                    }
                                    ItemChunkKind::OutputItemDone(message) => match message {
                                        StreamingItemDoneOutput { item: responses_api::Output::FunctionCall(func), .. } => {
                                            let internal_id = tool_call_internal_ids
                                                .entry(func.id.clone())
                                                .or_insert_with(|| nanoid::nanoid!())
                                                .clone();
                                            let raw_tool_call = streaming::RawStreamingToolCall::new(
                                                func.id.clone(),
                                                func.name.clone(),
                                                func.arguments.clone(),
                                            )
                                            .with_internal_call_id(internal_id)
                                            .with_call_id(func.call_id.clone());
                                            tool_calls.push(RawStreamingChoice::ToolCall(raw_tool_call));
                                        }
                                        StreamingItemDoneOutput { item: responses_api::Output::Reasoning { summary, id, encrypted_content, .. }, .. } => {
                                            for reasoning_choice in responses_api::streaming::reasoning_choices_from_done_item(
                                                id,
                                                summary,
                                                encrypted_content.as_deref(),
                                            ) {
                                                match reasoning_choice {
                                                    RawStreamingChoice::Reasoning { id, content } => {
                                                        yield Ok(RawStreamingChoice::Reasoning { id, content });
                                                    }
                                                    RawStreamingChoice::ReasoningDelta { id, reasoning } => {
                                                        yield Ok(RawStreamingChoice::ReasoningDelta { id, reasoning });
                                                    }
                                                    _ => {}
                                                }
                                            }
                                        }
                                        StreamingItemDoneOutput { item: responses_api::Output::Message(msg), .. } => {
                                            yield Ok(RawStreamingChoice::MessageId(msg.id.clone()));
                                        }
                                        StreamingItemDoneOutput { item: responses_api::Output::Unknown, .. } => {}
                                    },
                                    ItemChunkKind::OutputTextDelta(delta) => {
                                        yield Ok(RawStreamingChoice::Message(delta.delta.clone()))
                                    }
                                    ItemChunkKind::ReasoningSummaryTextDelta(delta) => {
                                        yield Ok(RawStreamingChoice::ReasoningDelta { id: None, reasoning: delta.delta.clone() })
                                    }
                                    ItemChunkKind::RefusalDelta(delta) => {
                                        yield Ok(RawStreamingChoice::Message(delta.delta.clone()))
                                    }
                                    ItemChunkKind::FunctionCallArgsDelta(delta) => {
                                        let internal_call_id = tool_call_internal_ids
                                            .entry(delta.item_id.clone())
                                            .or_insert_with(|| nanoid::nanoid!())
                                            .clone();
                                        yield Ok(RawStreamingChoice::ToolCallDelta {
                                            id: delta.item_id.clone(),
                                            internal_call_id,
                                            content: streaming::ToolCallDeltaContent::Delta(delta.delta.clone())
                                        })
                                    }
                                    _ => continue,
                                }
                            }

                            if let responses_api::streaming::StreamingCompletionChunk::Response(chunk) = data {
                                let responses_api::streaming::ResponseChunk { kind, response, .. } = *chunk;
                                match kind {
                                    responses_api::streaming::ResponseChunkKind::ResponseCompleted => {
                                        span.record("gen_ai.response.id", response.id.as_str());
                                        span.record("gen_ai.response.model", response.model.as_str());
                                        if let Some(usage) = response.usage {
                                            final_usage = usage;
                                        }
                                    }
                                    responses_api::streaming::ResponseChunkKind::ResponseFailed
                                    | responses_api::streaming::ResponseChunkKind::ResponseIncomplete => {
                                        let error = response
                                            .error
                                            .as_ref()
                                            .map(|err| err.message.clone())
                                            .unwrap_or_else(|| "Copilot response stream failed".into());
                                        terminated_with_error = true;
                                        yield Err(CompletionError::ProviderError(error));
                                        break;
                                    }
                                    _ => continue,
                                }
                            }
                        }
                        Err(crate::http_client::Error::StreamEnded) => {
                            break;
                        }
                        Err(error) => {
                            terminated_with_error = true;
                            yield Err(CompletionError::ProviderError(error.to_string()));
                            break;
                        }
                    }
                }

                event_source.close();

                if terminated_with_error {
                    return;
                }

                for tool_call in &tool_calls {
                    yield Ok(tool_call.to_owned())
                }

                span.record("gen_ai.usage.input_tokens", final_usage.input_tokens);
                span.record("gen_ai.usage.output_tokens", final_usage.output_tokens);
                span.record(
                    "gen_ai.usage.cache_read.input_tokens",
                    final_usage
                        .input_tokens_details
                        .as_ref()
                        .map(|details| details.cached_tokens)
                        .unwrap_or(0),
                );

                yield Ok(RawStreamingChoice::FinalResponse(
                    CopilotStreamingResponse::Responses(
                        responses_api::streaming::StreamingCompletionResponse { usage: final_usage }
                    )
                ));
            },
            span,
        );

        Ok(StreamingCompletionResponse::stream(Box::pin(stream)))
    }
}

impl<H> completion::CompletionModel for CompletionModel<H>
where
    Client<H>: HttpClientExt + Clone + Debug + 'static,
    H: Clone + Default + Debug + WasmCompatSend + WasmCompatSync + 'static,
{
    type Response = CopilotCompletionResponse;
    type StreamingResponse = CopilotStreamingResponse;
    type Client = Client<H>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<Self::Response>, CompletionError> {
        match self.route() {
            CompletionRoute::ChatCompletions => self.completion_chat(completion_request).await,
            CompletionRoute::Responses => self.completion_responses(completion_request).await,
        }
    }

    async fn stream(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        match self.route() {
            CompletionRoute::ChatCompletions => self.stream_chat(completion_request).await,
            CompletionRoute::Responses => self.stream_responses(completion_request).await,
        }
    }
}

#[derive(Clone)]
pub struct EmbeddingModel<H = reqwest::Client> {
    client: Client<H>,
    pub model: String,
    pub encoding_format: Option<openai::EncodingFormat>,
    pub user: Option<String>,
    ndims: usize,
}

#[derive(Deserialize)]
struct CopilotEmbeddingResponse {
    data: Vec<CopilotEmbeddingData>,
}

#[derive(Deserialize)]
struct CopilotEmbeddingData {
    embedding: Vec<serde_json::Number>,
}

impl<H> EmbeddingModel<H>
where
    Client<H>: HttpClientExt + Clone + Debug + 'static,
    H: Clone + Default + Debug + 'static,
{
    pub fn new(client: Client<H>, model: impl Into<String>, ndims: usize) -> Self {
        Self {
            client,
            model: model.into(),
            encoding_format: None,
            user: None,
            ndims,
        }
    }
}

impl<H> embeddings::EmbeddingModel for EmbeddingModel<H>
where
    Client<H>: HttpClientExt + Clone + Debug + WasmCompatSend + WasmCompatSync + 'static,
    H: Clone + Default + Debug + WasmCompatSend + WasmCompatSync + 'static,
{
    const MAX_DOCUMENTS: usize = 1024;
    type Client = Client<H>;

    fn make(client: &Self::Client, model: impl Into<String>, ndims: Option<usize>) -> Self {
        let model = model.into();
        let dims = ndims.unwrap_or(match model.as_str() {
            TEXT_EMBEDDING_3_LARGE => 3072,
            TEXT_EMBEDDING_3_SMALL | TEXT_EMBEDDING_ADA_002 => 1536,
            _ => 0,
        });
        Self::new(client.clone(), model, dims)
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents = documents.into_iter().collect::<Vec<_>>();
        let auth = self
            .client
            .ext()
            .auth
            .auth_context()
            .await
            .map_err(|err| EmbeddingError::ProviderError(err.to_string()))?;

        let headers = default_headers(&auth.api_key, "user", false);
        let mut body = json!({
            "model": self.model,
            "input": documents,
        });

        let body_object = body.as_object_mut().ok_or_else(|| {
            EmbeddingError::ResponseError("embedding request body must be a JSON object".into())
        })?;

        if self.ndims > 0 && self.model.as_str() != TEXT_EMBEDDING_ADA_002 {
            body_object.insert("dimensions".to_owned(), json!(self.ndims));
        }
        if let Some(encoding_format) = &self.encoding_format {
            body_object.insert("encoding_format".to_owned(), json!(encoding_format));
        }
        if let Some(user) = &self.user {
            body_object.insert("user".to_owned(), json!(user));
        }

        let req = apply_headers(
            post_with_auth_base(&self.client, &auth, "/embeddings", Transport::Http)?,
            &headers,
        )
        .body(serde_json::to_vec(&body)?)
        .map_err(|err| EmbeddingError::HttpError(err.into()))?;

        let response = self.client.send(req).await?;
        if response.status().is_success() {
            let body: Vec<u8> = response.into_body().await?;
            #[derive(Deserialize)]
            struct NestedApiError {
                error: NestedApiErrorMessage,
            }

            #[derive(Deserialize)]
            struct NestedApiErrorMessage {
                message: String,
            }

            let body: CopilotEmbeddingResponse = match serde_json::from_slice(&body) {
                Ok(parsed) => parsed,
                Err(parse_error) => {
                    if let Ok(err) = serde_json::from_slice::<NestedApiError>(&body) {
                        return Err(EmbeddingError::ProviderError(err.error.message));
                    }

                    let preview = String::from_utf8_lossy(&body);
                    let preview = if preview.len() > 512 {
                        format!("{}...", &preview[..512])
                    } else {
                        preview.into_owned()
                    };

                    return Err(EmbeddingError::ProviderError(format!(
                        "Failed to parse Copilot embeddings response: {parse_error}; body: {preview}"
                    )));
                }
            };

            Ok(body
                .data
                .into_iter()
                .zip(documents.into_iter())
                .map(|(embedding, document)| embeddings::Embedding {
                    document,
                    vec: embedding
                        .embedding
                        .into_iter()
                        .filter_map(|n| n.as_f64())
                        .collect(),
                })
                .collect())
        } else {
            let text = http_client::text(response).await?;
            Err(EmbeddingError::ProviderError(text))
        }
    }
}

#[derive(Deserialize, Debug)]
struct ChatStreamingFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Deserialize, Debug)]
struct ChatStreamingToolCall {
    index: usize,
    id: Option<String>,
    function: ChatStreamingFunction,
}

impl From<&ChatStreamingToolCall> for CompatibleToolCallChunk {
    fn from(value: &ChatStreamingToolCall) -> Self {
        Self {
            index: value.index,
            id: value.id.clone(),
            name: value.function.name.clone(),
            arguments: value.function.arguments.clone(),
        }
    }
}

#[derive(Deserialize, Debug, Default)]
struct ChatStreamingDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default, deserialize_with = "crate::json_utils::null_or_vec")]
    tool_calls: Vec<ChatStreamingToolCall>,
}

#[derive(Deserialize, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
enum ChatFinishReason {
    ToolCalls,
    Stop,
    ContentFilter,
    Length,
    #[serde(untagged)]
    Other(String),
}

#[derive(Deserialize, Debug)]
struct ChatStreamingChoice {
    delta: ChatStreamingDelta,
    finish_reason: Option<ChatFinishReason>,
}

#[derive(Deserialize, Debug)]
struct ChatStreamingChunk {
    id: Option<String>,
    model: Option<String>,
    choices: Vec<ChatStreamingChoice>,
    usage: Option<openai::completion::Usage>,
}

#[derive(Clone, Copy)]
struct CopilotChatCompatibleProfile;

impl CompatibleStreamProfile for CopilotChatCompatibleProfile {
    type Usage = openai::completion::Usage;
    type Detail = ();
    type FinalResponse = CopilotStreamingResponse;

    fn normalize_chunk(
        &self,
        data: &str,
    ) -> Result<Option<CompatibleChunk<Self::Usage, Self::Detail>>, CompletionError> {
        let data = match serde_json::from_str::<ChatStreamingChunk>(data) {
            Ok(data) => data,
            Err(error) => {
                tracing::debug!(?error, "Couldn't parse Copilot chat SSE payload");
                return Ok(None);
            }
        };

        Ok(Some(
            openai_chat_completions_compatible::normalize_first_choice_chunk(
                data.id,
                data.model,
                data.usage,
                &data.choices,
                |choice| CompatibleChoiceData {
                    finish_reason: if choice.finish_reason == Some(ChatFinishReason::ToolCalls) {
                        CompatibleFinishReason::ToolCalls
                    } else {
                        CompatibleFinishReason::Other
                    },
                    text: choice.delta.content.clone(),
                    reasoning: choice.delta.reasoning_content.clone(),
                    tool_calls: openai_chat_completions_compatible::tool_call_chunks(
                        &choice.delta.tool_calls,
                    ),
                    details: Vec::new(),
                },
            ),
        ))
    }

    fn build_final_response(&self, usage: Self::Usage) -> Self::FinalResponse {
        CopilotStreamingResponse::Chat(openai::completion::streaming::StreamingCompletionResponse {
            usage,
        })
    }

    fn uses_distinct_tool_call_eviction(&self) -> bool {
        true
    }
}

async fn send_copilot_chat_streaming_request<T>(
    http_client: T,
    req: Request<Vec<u8>>,
) -> Result<StreamingCompletionResponse<CopilotStreamingResponse>, CompletionError>
where
    T: HttpClientExt + Clone + 'static,
{
    openai_chat_completions_compatible::send_compatible_streaming_request(
        http_client,
        req,
        CopilotChatCompatibleProfile,
    )
    .await
}

fn default_token_dir() -> Option<PathBuf> {
    config_dir().map(|dir| dir.join("github_copilot"))
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

#[cfg(test)]
mod tests {
    use super::{
        ChatApiErrorResponse, ChatCompletionResponse, Client, CompletionRoute,
        TEXT_EMBEDDING_3_SMALL, env_api_key, env_base_url, env_github_access_token,
        route_for_model,
    };
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel;
    use crate::http_client::mock::MockStreamingClient;
    use crate::http_client::{self, HttpClientExt, LazyBody, MultipartForm, Request, Response};
    use crate::providers::internal::openai_chat_completions_compatible::test_support::{
        sse_bytes_from_data_lines, sse_bytes_from_json_events,
    };
    use crate::streaming::StreamedAssistantContent;
    use bytes::Bytes;
    use futures::StreamExt;
    use std::collections::HashMap;
    use std::future::{self, Future};
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct CapturedRequest {
        uri: String,
        body: Bytes,
    }

    #[derive(Debug, Clone, Default)]
    struct RecordingHttpClient {
        requests: Arc<Mutex<Vec<CapturedRequest>>>,
        response_body: Bytes,
    }

    impl RecordingHttpClient {
        fn new(response_body: impl Into<Bytes>) -> Self {
            Self {
                requests: Arc::new(Mutex::new(Vec::new())),
                response_body: response_body.into(),
            }
        }

        fn requests(&self) -> Vec<CapturedRequest> {
            self.requests.lock().expect("requests lock").clone()
        }
    }

    impl HttpClientExt for RecordingHttpClient {
        fn send<T, U>(
            &self,
            req: Request<T>,
        ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>>
        + crate::wasm_compat::WasmCompatSend
        + 'static
        where
            T: Into<Bytes> + crate::wasm_compat::WasmCompatSend,
            U: From<Bytes> + crate::wasm_compat::WasmCompatSend + 'static,
        {
            let requests = Arc::clone(&self.requests);
            let response_body = self.response_body.clone();
            let (parts, body) = req.into_parts();
            let body = body.into();

            requests
                .lock()
                .expect("requests lock")
                .push(CapturedRequest {
                    uri: parts.uri.to_string(),
                    body,
                });

            async move {
                let body: LazyBody<U> = Box::pin(async move { Ok(U::from(response_body)) });
                Response::builder()
                    .status(http::StatusCode::OK)
                    .body(body)
                    .map_err(http_client::Error::Protocol)
            }
        }

        fn send_multipart<U>(
            &self,
            _req: Request<MultipartForm>,
        ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>>
        + crate::wasm_compat::WasmCompatSend
        + 'static
        where
            U: From<Bytes> + crate::wasm_compat::WasmCompatSend + 'static,
        {
            future::ready(Err(http_client::Error::InvalidStatusCode(
                http::StatusCode::NOT_IMPLEMENTED,
            )))
        }

        fn send_streaming<T>(
            &self,
            _req: Request<T>,
        ) -> impl Future<Output = http_client::Result<http_client::StreamingResponse>>
        + crate::wasm_compat::WasmCompatSend
        where
            T: Into<Bytes>,
        {
            future::ready(Err(http_client::Error::InvalidStatusCode(
                http::StatusCode::NOT_IMPLEMENTED,
            )))
        }
    }

    #[derive(Debug, Clone, Default)]
    struct SequencedStreamingHttpClient {
        chunks: Arc<Mutex<Option<Vec<http_client::Result<Bytes>>>>>,
    }

    impl SequencedStreamingHttpClient {
        fn new(chunks: Vec<http_client::Result<Bytes>>) -> Self {
            Self {
                chunks: Arc::new(Mutex::new(Some(chunks))),
            }
        }
    }

    impl HttpClientExt for SequencedStreamingHttpClient {
        fn send<T, U>(
            &self,
            _req: Request<T>,
        ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>>
        + crate::wasm_compat::WasmCompatSend
        + 'static
        where
            T: Into<Bytes> + crate::wasm_compat::WasmCompatSend,
            U: From<Bytes> + crate::wasm_compat::WasmCompatSend + 'static,
        {
            future::ready(Err(http_client::Error::InvalidStatusCode(
                http::StatusCode::NOT_IMPLEMENTED,
            )))
        }

        fn send_multipart<U>(
            &self,
            _req: Request<MultipartForm>,
        ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>>
        + crate::wasm_compat::WasmCompatSend
        + 'static
        where
            U: From<Bytes> + crate::wasm_compat::WasmCompatSend + 'static,
        {
            future::ready(Err(http_client::Error::InvalidStatusCode(
                http::StatusCode::NOT_IMPLEMENTED,
            )))
        }

        fn send_streaming<T>(
            &self,
            _req: Request<T>,
        ) -> impl Future<Output = http_client::Result<http_client::StreamingResponse>>
        + crate::wasm_compat::WasmCompatSend
        where
            T: Into<Bytes>,
        {
            let chunks = self
                .chunks
                .lock()
                .expect("chunks lock")
                .take()
                .expect("streaming chunks should only be consumed once");

            async move {
                let byte_stream = futures::stream::iter(chunks);
                let boxed_stream: http_client::sse::BoxedStream = Box::pin(byte_stream);

                Response::builder()
                    .status(http::StatusCode::OK)
                    .header(http::header::CONTENT_TYPE, "text/event-stream")
                    .body(boxed_stream)
                    .map_err(http_client::Error::Protocol)
            }
        }
    }

    fn env_map(entries: &[(&str, &str)]) -> HashMap<String, String> {
        entries
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect()
    }

    fn minimal_chat_response() -> &'static str {
        r#"{
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 4,
                "total_tokens": 7
            }
        }"#
    }

    fn minimal_responses_response() -> &'static str {
        r#"{
            "id": "resp_123",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "gpt-5.3-codex",
            "usage": {
                "input_tokens": 4,
                "input_tokens_details": {
                    "cached_tokens": 0
                },
                "output_tokens": 3,
                "output_tokens_details": {
                    "reasoning_tokens": 0
                },
                "total_tokens": 7
            },
            "output": [{
                "type": "message",
                "id": "msg_123",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": "hello"
                }]
            }],
            "tools": []
        }"#
    }

    fn minimal_embeddings_response() -> &'static str {
        r#"{
            "data": [
                {
                    "embedding": [0.1, 0.2, 0.3]
                },
                {
                    "embedding": [0.4, 0.5, 0.6]
                }
            ]
        }"#
    }

    #[test]
    fn deserialize_standard_openai_response() {
        let json = r#"{
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;

        let response: ChatCompletionResponse =
            serde_json::from_str(json).expect("standard OpenAI response should deserialize");
        assert_eq!(response.id, "chatcmpl-abc123");
        assert_eq!(response.object.as_deref(), Some("chat.completion"));
        assert_eq!(response.created, Some(1700000000));
        assert_eq!(response.model, "gpt-4o");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn deserialize_copilot_response_without_object_and_created() {
        let response: ChatCompletionResponse = serde_json::from_str(minimal_chat_response())
            .expect("Copilot response should deserialize");

        assert_eq!(response.id, "chatcmpl-123");
        assert_eq!(response.object, None);
        assert_eq!(response.created, None);
        assert_eq!(response.model, "gpt-4o");
        assert_eq!(response.choices.len(), 1);
    }

    #[test]
    fn deserialize_copilot_response_without_finish_reason() {
        let json = r#"{
            "id": "chatcmpl-claude-001",
            "model": "claude-3.5-sonnet",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Here is my analysis."
                }
            }],
            "usage": {
                "prompt_tokens": 50,
                "total_tokens": 80
            }
        }"#;

        let response: ChatCompletionResponse =
            serde_json::from_str(json).expect("Claude-via-Copilot response should deserialize");

        assert_eq!(response.model, "claude-3.5-sonnet");
        assert_eq!(response.choices[0].finish_reason, None);
        assert_eq!(response.choices[0].index, 0);
    }

    #[test]
    fn error_response_with_message_field() {
        let json = r#"{"message": "rate limit exceeded"}"#;
        let err: ChatApiErrorResponse = serde_json::from_str(json).expect("message-shaped error");

        assert_eq!(err.error_message(), "rate limit exceeded");
    }

    #[test]
    fn error_response_with_error_field() {
        let json = r#"{"error": "model not found"}"#;
        let err: ChatApiErrorResponse = serde_json::from_str(json).expect("error-shaped error");

        assert_eq!(err.error_message(), "model not found");
    }

    #[test]
    fn routes_codex_models_to_responses() {
        assert_eq!(route_for_model("gpt-5.3-codex"), CompletionRoute::Responses);
        assert_eq!(
            route_for_model("gpt-5.1-CODEX-mini"),
            CompletionRoute::Responses
        );
        assert_eq!(route_for_model("gpt-5.2"), CompletionRoute::ChatCompletions);
        assert_eq!(
            route_for_model("claude-sonnet-4.5"),
            CompletionRoute::ChatCompletions
        );
    }

    #[tokio::test]
    async fn completion_model_routes_chat_requests_to_chat_completions() {
        let http_client = RecordingHttpClient::new(minimal_chat_response());
        let client = Client::builder()
            .api_key("copilot-token")
            .http_client(http_client.clone())
            .build()
            .expect("build client");
        let model = client.completion_model("gpt-4o");
        let request = model.completion_request("hello").build();

        let _response = model.completion(request).await.expect("chat completion");

        let requests = http_client.requests();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].uri.ends_with("/chat/completions"));
        assert!(String::from_utf8_lossy(&requests[0].body).contains("\"model\":\"gpt-4o\""));
    }

    #[tokio::test]
    async fn completion_model_routes_codex_requests_to_responses() {
        let http_client = RecordingHttpClient::new(minimal_responses_response());
        let client = Client::builder()
            .api_key("copilot-token")
            .http_client(http_client.clone())
            .build()
            .expect("build client");
        let model = client.completion_model("gpt-5.3-codex");
        let request = model.completion_request("hello").build();

        let _response = model
            .completion(request)
            .await
            .expect("responses completion");

        let requests = http_client.requests();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].uri.ends_with("/responses"));
        assert!(String::from_utf8_lossy(&requests[0].body).contains("\"model\":\"gpt-5.3-codex\""));
    }

    #[tokio::test]
    async fn embeddings_accept_minimal_copilot_response_shape() {
        use crate::client::EmbeddingsClient;
        use crate::embeddings::EmbeddingModel as _;

        let http_client = RecordingHttpClient::new(minimal_embeddings_response());
        let client = Client::builder()
            .api_key("copilot-token")
            .http_client(http_client.clone())
            .build()
            .expect("build client");
        let model = client.embedding_model(TEXT_EMBEDDING_3_SMALL);

        let embeddings = model
            .embed_texts(["one".to_string(), "two".to_string()])
            .await
            .expect("embeddings should deserialize");

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].vec, vec![0.1, 0.2, 0.3]);
        assert_eq!(embeddings[1].vec, vec![0.4, 0.5, 0.6]);

        let requests = http_client.requests();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].uri.ends_with("/embeddings"));
        assert!(
            String::from_utf8_lossy(&requests[0].body)
                .contains("\"model\":\"text-embedding-3-small\"")
        );
    }

    #[tokio::test]
    async fn responses_stream_terminates_after_terminal_error() {
        let tool_call_done = serde_json::json!({
            "type": "response.output_item.done",
            "sequence_number": 1,
            "item": {
                "type": "function_call",
                "id": "fc_123",
                "arguments": "{}",
                "call_id": "call_123",
                "name": "example_tool",
                "status": "completed"
            }
        });
        let failed = serde_json::json!({
            "type": "response.failed",
            "sequence_number": 2,
            "response": {
                "id": "resp_123",
                "object": "response",
                "created_at": 1700000000,
                "status": "failed",
                "error": {
                    "code": "server_error",
                    "message": "Copilot response stream failed"
                },
                "incomplete_details": null,
                "instructions": null,
                "max_output_tokens": null,
                "model": "gpt-5.3-codex",
                "usage": null,
                "output": [],
                "tools": []
            }
        });
        let http_client = MockStreamingClient {
            sse_bytes: sse_bytes_from_json_events(&[tool_call_done, failed]),
        };
        let client = Client::builder()
            .api_key("copilot-token")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model("gpt-5.3-codex");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        let err = match stream.next().await.expect("stream should yield an item") {
            Ok(_) => panic!("stream should surface a provider error"),
            Err(err) => err,
        };
        assert_eq!(
            err.to_string(),
            "ProviderError: Copilot response stream failed"
        );
        assert!(
            stream.next().await.is_none(),
            "responses stream should terminate immediately after a terminal error"
        );
    }

    #[tokio::test]
    async fn chat_stream_terminates_after_transport_error() {
        let chunks = vec![
            Ok(sse_bytes_from_data_lines([
                "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_123\",\"function\":{\"name\":\"ping\",\"arguments\":\"\"}}]},\"finish_reason\":null}],\"usage\":null}",
            ])),
            Err(http_client::Error::InvalidStatusCode(
                http::StatusCode::BAD_GATEWAY,
            )),
        ];

        let http_client = SequencedStreamingHttpClient::new(chunks);
        let client = Client::builder()
            .api_key("copilot-token")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model("gpt-4o");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        let mut saw_error = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamedAssistantContent::ToolCallDelta { .. }) => {}
                Err(err) => {
                    assert_eq!(
                        err.to_string(),
                        "ProviderError: Invalid status code: 502 Bad Gateway"
                    );
                    saw_error = true;
                    break;
                }
                Ok(_) => panic!("unexpected non-error stream item before transport failure"),
            }
        }

        assert!(saw_error, "stream should surface the transport error");
        assert!(
            stream.next().await.is_none(),
            "chat stream should terminate immediately after a transport error"
        );
    }

    #[test]
    fn env_api_key_prefers_github_prefixed_vars() {
        let env = env_map(&[
            ("COPILOT_API_KEY", "copilot-key"),
            ("GITHUB_COPILOT_API_KEY", "github-key"),
            ("GITHUB_TOKEN", "bootstrap-token"),
        ]);
        let get = |name: &str| env.get(name).cloned();

        assert_eq!(env_api_key(&get).as_deref(), Some("github-key"));
    }

    #[test]
    fn env_github_access_token_prefers_explicit_bootstrap_var() {
        let env = env_map(&[
            ("COPILOT_GITHUB_ACCESS_TOKEN", "explicit-bootstrap"),
            ("GITHUB_TOKEN", "fallback-bootstrap"),
        ]);
        let get = |name: &str| env.get(name).cloned();

        assert_eq!(
            env_github_access_token(&get).as_deref(),
            Some("explicit-bootstrap")
        );
    }

    #[test]
    fn env_base_url_prefers_github_prefixed_vars() {
        let env = env_map(&[
            ("COPILOT_BASE_URL", "https://copilot.example"),
            ("GITHUB_COPILOT_API_BASE", "https://github.example"),
        ]);
        let get = |name: &str| env.get(name).cloned();

        assert_eq!(
            env_base_url(&get).as_deref(),
            Some("https://github.example")
        );
    }

    #[test]
    fn env_without_api_key_falls_back_to_oauth() {
        let env = env_map(&[("COPILOT_BASE_URL", "https://copilot.example")]);
        let get = |name: &str| env.get(name).cloned();

        assert!(env_api_key(&get).is_none());
        assert!(env_github_access_token(&get).is_none());
        assert_eq!(
            env_base_url(&get).as_deref(),
            Some("https://copilot.example")
        );
    }

    #[test]
    fn env_github_token_is_not_treated_as_copilot_api_key() {
        let env = env_map(&[("GITHUB_TOKEN", "bootstrap-token")]);
        let get = |name: &str| env.get(name).cloned();

        assert!(env_api_key(&get).is_none());
        assert_eq!(
            env_github_access_token(&get).as_deref(),
            Some("bootstrap-token")
        );
    }
}
