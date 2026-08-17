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
//! use rig_core::client::{CompletionClient, ProviderClient};
//! use rig_core::providers::copilot;
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
    self, ApiKey, DebugExt, ModelLister, Provider, ProviderBuilder, ProviderClient, Transport,
};
use crate::completion::NormalizeCompletionResponse;
use crate::completion::{self, CompletionError};
use crate::embeddings::{self, EmbeddingError};
use crate::http_client::{self, HttpClientExt};
use crate::model::{Model, ModelList, ModelListingError};
use crate::providers::internal::completion_send::send_completion;
use crate::providers::internal::envelope::DirectPayload;
use crate::providers::openai;
use crate::providers::openai::responses_api::{self, CompletionRequest as ResponsesRequest};
use crate::streaming::StreamingCompletionResponse;
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder, SpanCombinator};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use futures::StreamExt;
use http::Request;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::borrow::Cow;
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use tracing_futures::Instrument as _;

const GITHUB_COPILOT_API_BASE_URL: &str = "https://api.githubcopilot.com";
pub(crate) const EDITOR_PLUGIN_VERSION: &str = "copilot-chat/0.35.0";
pub(crate) const USER_AGENT: &str = "GitHubCopilotChat/0.35.0";
pub(crate) const EDITOR_VERSION: &str = "vscode/1.107.0";
const API_VERSION: &str = "2025-04-01";

/// Copilot conversation intent sent in the `openai-intent` request header.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CopilotIntent {
    /// Generic chat panel conversation semantics.
    #[default]
    Panel,
    /// Edit-oriented conversation semantics.
    Edits,
}

impl CopilotIntent {
    fn as_header(self) -> &'static str {
        match self {
            Self::Panel => "conversation-panel",
            Self::Edits => "conversation-edits",
        }
    }
}

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
/// `gpt-5.5`
pub const GPT_5_5: &str = "gpt-5.5";
/// `gpt-5.4`
pub const GPT_5_4: &str = "gpt-5.4";
/// `claude-sonnet-4` completion model (Anthropic, via Copilot)
pub const CLAUDE_SONNET_4: &str = "claude-sonnet-4";
/// `claude-sonnet-4.6`
pub const CLAUDE_SONNET_4_6: &str = "claude-sonnet-4.6";
/// `claude-opus-4.6`
pub const CLAUDE_OPUS_4_6: &str = "claude-opus-4.6";
/// `claude-opus-4.7`
pub const CLAUDE_OPUS_4_7: &str = "claude-opus-4.7";
/// `claude-3.5-sonnet` completion model (Anthropic, via Copilot)
pub const CLAUDE_3_5_SONNET: &str = "claude-3.5-sonnet";
/// `gemini-3-flash-preview` completion model (Google, via Copilot)
pub const GEMINI_3_FLASH: &str = "gemini-3-flash-preview";
/// `gemini-3.1-pro-preview` completion model (Google, via Copilot)
pub const GEMINI_3_1_PRO_FLASH: &str = "gemini-3.1-pro-preview";
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
    allow_device_flow: bool,
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
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<CopilotBuilder, CopilotAuth, H>;

impl Default for CopilotBuilder {
    fn default() -> Self {
        let token_dir = default_token_dir();
        Self {
            access_token_file: token_dir.as_ref().map(|dir| dir.join("access-token")),
            api_key_file: token_dir.map(|dir| dir.join("api-key.json")),
            device_code_handler: auth::DeviceCodeHandler::default(),
            allow_device_flow: true,
        }
    }
}

impl Provider for CopilotExt {
    type Builder = CopilotBuilder;

    const VERIFY_PATH: &'static str = "";
}

client::impl_capabilities!(
    CopilotExt,
    completion = CompletionModel<H>,
    embeddings = EmbeddingModel<H>,
    model_listing = CopilotModelLister<H>,
);

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
                ext.allow_device_flow,
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

    /// Control whether OAuth may fall back to an interactive device-code login
    /// when the cached token is missing or cannot refresh.
    ///
    /// Default is `true` for CLI-style interactive use. Services should set it
    /// to `false` so unattended background work returns a clear auth error
    /// instead of printing a device code and waiting.
    pub fn allow_device_flow(self, allow: bool) -> Self {
        self.over_ext(|mut ext| {
            ext.allow_device_flow = allow;
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
    intent: CopilotIntent,
) -> Vec<(&'static str, String)> {
    let mut headers = vec![
        (
            http::header::AUTHORIZATION.as_str(),
            format!("Bearer {api_key}"),
        ),
        ("copilot-integration-id", "vscode-chat".to_string()),
        ("editor-version", EDITOR_VERSION.to_string()),
        ("editor-plugin-version", EDITOR_PLUGIN_VERSION.to_string()),
        ("user-agent", USER_AGENT.to_string()),
        ("openai-intent", intent.as_header().to_string()),
        ("x-github-api-version", API_VERSION.to_string()),
        ("x-request-id", crate::id::generate()),
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
    if client.base_url() != GITHUB_COPILOT_API_BASE_URL {
        return Cow::Borrowed(client.base_url());
    }

    if let Some(api_base) = auth.api_base.as_deref() {
        return Cow::Borrowed(api_base);
    }

    if let Some(base_url) = base_url_from_token(&auth.api_key) {
        return Cow::Owned(base_url);
    }

    Cow::Borrowed(client.base_url())
}

/// Derive the Copilot REST base URL from a chat token's `proxy-ep=` segment.
///
/// The endpoint is parsed from a credential string, not from explicit caller
/// configuration. For that reason, token-derived routing is limited to GitHub
/// Copilot service hosts and HTTPS. Callers that need a custom non-GitHub host
/// can still opt in explicitly with [`ClientBuilder::base_url`].
fn base_url_from_token(token: &str) -> Option<String> {
    let proxy_ep = token
        .split(';')
        .find_map(|part| part.trim().strip_prefix("proxy-ep="))?
        .trim();

    normalize_copilot_proxy_endpoint(proxy_ep)
}

fn normalize_copilot_proxy_endpoint(proxy_ep: &str) -> Option<String> {
    if proxy_ep.is_empty() {
        return None;
    }

    let candidate = if proxy_ep.starts_with("http://") || proxy_ep.starts_with("https://") {
        proxy_ep.to_string()
    } else {
        format!("https://{proxy_ep}")
    };

    let mut url = url::Url::parse(&candidate).ok()?;
    if url.scheme() != "https" || !url.username().is_empty() || url.password().is_some() {
        return None;
    }
    if url.path() != "/" || url.query().is_some() || url.fragment().is_some() {
        return None;
    }

    let host = url.host_str()?.to_ascii_lowercase();
    if !is_allowed_token_derived_copilot_host(&host) {
        return None;
    }

    let api_host = host
        .strip_prefix("proxy.")
        .map(|suffix| format!("api.{suffix}"))
        .unwrap_or(host);
    url.set_host(Some(&api_host)).ok()?;

    Some(url.to_string().trim_end_matches('/').to_string())
}

fn is_allowed_token_derived_copilot_host(host: &str) -> bool {
    host == "githubcopilot.com" || host.ends_with(".githubcopilot.com")
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

fn get_with_auth_base<H>(
    client: &Client<H>,
    auth: &auth::AuthContext,
    path: &str,
    transport: Transport,
) -> http_client::Result<http_client::Builder> {
    let uri = client
        .ext()
        .build_uri(runtime_base_url(client, auth).as_ref(), path, transport);
    let mut req = Request::get(uri);

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

/// Per-request inputs shared by every Copilot route, read off the incoming
/// request before a route-specific conversion consumes it.
struct RequestFacts {
    initiator: &'static str,
    has_vision: bool,
    system_instructions: Option<String>,
    record_telemetry_content: bool,
}

impl RequestFacts {
    fn capture(request: &completion::CompletionRequest) -> Self {
        Self {
            initiator: request_initiator(request),
            has_vision: request_has_vision(request),
            system_instructions: request.preamble.clone(),
            record_telemetry_content: request.record_telemetry_content,
        }
    }
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
    Chat(Box<openai::completion::CompletionResponse>),
    Responses(Box<responses_api::CompletionResponse>),
}

/// The forward direction for the route-tagged raw type, so
/// [`CompletionModel::raw_completion`] followed by `normalize` is a complete
/// typed route regardless of which route answered — each variant delegates to
/// its wire type's own conversion. This is also what
/// [`completion::CompletionModel::completion`] uses, so the two cannot drift.
impl NormalizeCompletionResponse for CopilotCompletionResponse {
    fn normalize(self, provider: &str) -> Result<completion::CompletionResponse, CompletionError> {
        match self {
            Self::Chat(response) => response.normalize(provider),
            Self::Responses(response) => response.normalize(provider),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "api", rename_all = "snake_case")]
pub enum CopilotStreamingResponse {
    Chat(openai::completion::streaming::StreamingCompletionResponse),
    Responses(responses_api::streaming::StreamingCompletionResponse),
}

impl From<&CopilotStreamingResponse> for completion::Usage {
    fn from(response: &CopilotStreamingResponse) -> Self {
        match response {
            CopilotStreamingResponse::Chat(response) => (&response.usage).into(),
            CopilotStreamingResponse::Responses(response) => (&response.usage).into(),
        }
    }
}

impl From<(&str, CopilotStreamingResponse)> for crate::streaming::StreamFinal {
    fn from((provider, response): (&str, CopilotStreamingResponse)) -> Self {
        // Both Copilot routes reuse an upstream terminal record, so each maps
        // through that route's own conversion rather than re-deriving it here.
        match response {
            CopilotStreamingResponse::Chat(response) => (provider, response).into(),
            CopilotStreamingResponse::Responses(response) => (provider, response).into(),
        }
    }
}

/// Stable descriptor name reported on normalized Copilot responses.
pub const PROVIDER_NAME: &str = "copilot";

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

impl<T> crate::providers::internal::envelope::ProviderEnvelope for ChatApiResponse<T> {
    type Payload = T;

    fn into_payload(self) -> Result<T, String> {
        match self {
            Self::Ok(payload) => Ok(payload),
            Self::Err(error) => Err(error.error_message().to_owned()),
        }
    }
}

#[derive(Clone)]
pub struct CompletionModel<H = reqwest::Client> {
    client: Client<H>,
    pub model: String,
    pub strict_tools: bool,
    pub tool_result_array_content: bool,
    pub intent: CopilotIntent,
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
            intent: CopilotIntent::default(),
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

    /// Set the Copilot `openai-intent` header for completion and streaming requests.
    pub fn with_intent(mut self, intent: CopilotIntent) -> Self {
        self.intent = intent;
        self
    }

    /// Use the generic chat panel `openai-intent` header for completion and streaming requests.
    pub fn with_panel_intent(self) -> Self {
        self.with_intent(CopilotIntent::Panel)
    }

    /// Use the edit-oriented `openai-intent` header for completion and streaming requests.
    pub fn with_edits_intent(self) -> Self {
        self.with_intent(CopilotIntent::Edits)
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
            supports_response_format: true,
            supports_tools: true,
        })
    }

    fn responses_request(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<ResponsesRequest, CompletionError> {
        let mut request = ResponsesRequest::try_from(responses_api::ResponsesRequestParams {
            model: self.model.clone(),
            request: completion_request,
            system_instructions_placement:
                responses_api::SystemInstructionsPlacement::InputSystemMessages,
        })?;
        // Copilot's Responses endpoint expects strict function tool schemas for
        // reliable tool calls. Preserve that provider-specific behavior while
        // keeping Chat Completions strict mode opt-in.
        request.tools = request
            .tools
            .into_iter()
            .map(responses_api::ResponsesToolDefinition::with_strict)
            .collect();
        Ok(request)
    }

    /// Authenticates, signs a POST to `path`, and opens the route's completion
    /// span.
    ///
    /// Call this only *after* the route's request conversion: auth happens
    /// inside, so calling it earlier would report an auth failure ahead of a
    /// malformed request and invert the routes' error precedence.
    async fn signed_request(
        &self,
        facts: &RequestFacts,
        path: &str,
        transport: Transport,
        model: &str,
        operation: CompletionOperation,
        body: Vec<u8>,
    ) -> Result<(Request<Vec<u8>>, tracing::Span), CompletionError> {
        let auth = self.auth_context().await?;

        let headers = default_headers(
            &auth.api_key,
            facts.initiator,
            facts.has_vision,
            self.intent,
        );
        let req = apply_headers(
            post_with_auth_base(&self.client, &auth, path, transport)?,
            &headers,
        )
        .body(body)
        .map_err(|err| CompletionError::HttpError(err.into()))?;

        let span = CompletionSpanBuilder::new("copilot", model, operation)
            .system_instructions(
                facts.system_instructions.as_deref(),
                facts.record_telemetry_content,
            )
            .build();

        Ok((req, span))
    }

    /// The chat wire type has no transport-metadata slot, so the captured
    /// request id rides alongside; `completion()` stamps it onto the
    /// normalized response.
    async fn raw_completion_chat(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<(openai::completion::CompletionResponse, Option<String>), CompletionError> {
        let facts = RequestFacts::capture(&completion_request);
        let request = self.chat_request(completion_request)?;
        let (req, span) = self
            .signed_request(
                &facts,
                "/chat/completions",
                Transport::Http,
                &request.model,
                CompletionOperation::Chat,
                serde_json::to_vec(&request)?,
            )
            .await?;

        send_completion::<_, ChatApiResponse<openai::completion::CompletionResponse>, _>(
            &self.client,
            req,
            "Copilot chat completion",
            // The OpenAI-compatible default; a gateway that omits the header
            // yields None. Matches the streaming path, which goes through the
            // shared OpenAI wrapper and captures the same header.
            Some("x-request-id"),
            |response| {
                let span = tracing::Span::current();
                span.record_response_metadata(response);
                let usage = response
                    .usage
                    .as_ref()
                    .map(|usage| usage.to_normalized())
                    .unwrap_or_default();
                span.record_token_usage(&usage);
            },
        )
        .instrument(span)
        .await
    }

    async fn raw_completion_responses(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<responses_api::CompletionResponse, CompletionError> {
        let facts = RequestFacts::capture(&completion_request);
        let request = self.responses_request(completion_request)?;
        let (req, span) = self
            .signed_request(
                &facts,
                "/responses",
                Transport::Http,
                &request.model,
                CompletionOperation::Chat,
                serde_json::to_vec(&request)?,
            )
            .await?;

        send_completion::<_, DirectPayload<responses_api::CompletionResponse>, _>(
            &self.client,
            req,
            "Copilot responses completion",
            // See the chat path: the OpenAI-compatible default header.
            Some("x-request-id"),
            |response| {
                let span = tracing::Span::current();
                span.record("gen_ai.response.id", response.id.as_str());
                span.record("gen_ai.response.model", response.model.as_str());
                if let Some(usage) = &response.usage {
                    span.record_token_usage(&usage.into());
                }
            },
        )
        .instrument(span)
        .await
        .map(|(mut payload, provider_request_id)| {
            payload.provider_request_id = provider_request_id;
            payload
        })
    }

    async fn raw_stream_chat(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<crate::streaming::RawStreamingResult<CopilotStreamingResponse>, CompletionError>
    {
        let facts = RequestFacts::capture(&completion_request);
        let request = self.chat_request(completion_request)?;
        let mut request_json = serde_json::to_value(&request)?;
        let request_object = request_json.as_object_mut().ok_or_else(|| {
            CompletionError::ResponseError("copilot request body must be a JSON object".into())
        })?;
        request_object.insert("stream".to_owned(), json!(true));
        request_object.insert(
            "stream_options".to_owned(),
            json!({ "include_usage": true }),
        );

        let (req, span) = self
            .signed_request(
                &facts,
                "/chat/completions",
                Transport::Sse,
                &request.model,
                CompletionOperation::ChatStreaming,
                serde_json::to_vec(&request_json)?,
            )
            .await?;

        tracing::Instrument::instrument(
            send_copilot_chat_raw_streaming_request(self.client.clone(), req),
            span,
        )
        .await
    }

    async fn raw_stream_responses(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<crate::streaming::RawStreamingResult<CopilotStreamingResponse>, CompletionError>
    {
        let facts = RequestFacts::capture(&completion_request);
        let mut request = self.responses_request(completion_request)?;
        request.stream = Some(true);
        let (req, span) = self
            .signed_request(
                &facts,
                "/responses",
                Transport::Sse,
                &request.model,
                CompletionOperation::ChatStreaming,
                serde_json::to_vec(&request)?,
            )
            .await?;

        let client = self.client.clone();
        // The OpenAI-compatible default header, matching the chat route.
        let (event_source, request_id_slot) =
            crate::http_client::sse::GenericEventSource::new(client, req)
                .capture_request_id("x-request-id");

        // Copilot's `/responses` route relays OpenAI's Responses SSE wire
        // verbatim, so the shared classify + `RawChoiceAccumulator` machinery
        // is the event interpreter — only the auth/transport above and the
        // route-carrying terminal wrapper below are Copilot-specific.
        let raw = responses_api::streaming::raw_stream_from_event_source(event_source, span);
        let raw = crate::providers::internal::sse_transport::stamp_terminal_request_id(
            raw,
            Some(request_id_slot),
            Some("x-request-id"),
            |response, id| response.provider_request_id = Some(id),
        );
        let stream = raw.map(|item| {
            item.and_then(|choice| {
                choice.try_map_final(|response| Ok(CopilotStreamingResponse::Responses(response)))
            })
        });

        Ok(Box::pin(stream))
    }

    /// Execute a completion on whichever route this model is configured for and
    /// return Copilot's own wire response.
    ///
    /// This is the escape hatch for fields rig does not normalize;
    /// [`completion::CompletionModel::completion`] shares the same request,
    /// transport, telemetry and error path.
    ///
    /// On the chat route the transport request id (`x-request-id`) is not on
    /// the wire type and is dropped here; use
    /// [`Self::raw_completion_with_request_id`] when the typed route must
    /// reproduce everything `completion` returns.
    pub async fn raw_completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<CopilotCompletionResponse, CompletionError> {
        self.raw_completion_with_request_id(completion_request)
            .await
            .map(|(response, _)| response)
    }

    /// [`Self::raw_completion`] plus the transport request id from the
    /// `x-request-id` response header.
    ///
    /// The pair exists because the chat route's wire type
    /// ([`openai::completion::CompletionResponse`]) has no slot for a
    /// transport id — it is the shared OpenAI-compatible shape — while the
    /// normalized [`completion::CompletionResponse`] carries one. Without this
    /// method, `raw_completion(..)` followed by
    /// [`NormalizeCompletionResponse::normalize`] would silently lack the
    /// `provider_request_id` that [`completion::CompletionModel::completion`]
    /// reports. Reassemble with
    /// [`with_optional_provider_request_id`](completion::CompletionResponse::with_optional_provider_request_id).
    /// On the responses route the wire type carries the id itself; the pair's
    /// second element is that same value, so reassembly is a no-op there.
    pub async fn raw_completion_with_request_id(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<(CopilotCompletionResponse, Option<String>), CompletionError> {
        match self.route() {
            CompletionRoute::ChatCompletions => self
                .raw_completion_chat(completion_request)
                .await
                .map(|(response, id)| (CopilotCompletionResponse::Chat(Box::new(response)), id)),
            CompletionRoute::Responses => self
                .raw_completion_responses(completion_request)
                .await
                .map(|response| {
                    let id = response.provider_request_id.clone();
                    (CopilotCompletionResponse::Responses(Box::new(response)), id)
                }),
        }
    }

    /// Open a stream on whichever route this model is configured for, keeping
    /// the terminal record provider-native.
    pub async fn raw_stream(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<crate::streaming::RawStreamingResult<CopilotStreamingResponse>, CompletionError>
    {
        match self.route() {
            CompletionRoute::ChatCompletions => self.raw_stream_chat(completion_request).await,
            CompletionRoute::Responses => self.raw_stream_responses(completion_request).await,
        }
    }

    /// Open a stream normalized to rig's terminal record. Delegates to
    /// [`CompletionModel::raw_stream`] — one request either way.
    async fn stream_normalized(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        let raw = self.raw_stream(completion_request).await?;

        Ok(StreamingCompletionResponse::stream(
            PROVIDER_NAME,
            crate::streaming::normalize_stream(
                raw,
                |response| Ok((PROVIDER_NAME, response).into()),
            ),
        ))
    }
}

impl<H> crate::client::ConstructCompletionModel<Client<H>> for CompletionModel<H>
where
    Client<H>: HttpClientExt + Clone + Debug + 'static,
    H: Clone + Default + Debug + WasmCompatSend + WasmCompatSync + 'static,
{
    fn construct(client: &Client<H>, model: String) -> Self {
        Self::new(client.clone(), model)
    }
}

impl<H> completion::CompletionModel for CompletionModel<H>
where
    Client<H>: HttpClientExt + Clone + Debug + 'static,
    H: Clone + Default + Debug + WasmCompatSend + WasmCompatSync + 'static,
{
    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        // The captured value is the route-tagged `CopilotCompletionResponse` —
        // what `raw_completion` returns — not the inner route type, so it
        // round-trips into the same type the typed escape hatch yields.
        let (response, provider_request_id) = self
            .raw_completion_with_request_id(completion_request)
            .await?;
        let captured = serde_json::to_value(&response)?;
        Ok(response
            .normalize(PROVIDER_NAME)?
            .with_optional_provider_request_id(provider_request_id)
            .with_raw(captured))
    }

    async fn stream(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        self.stream_normalized(completion_request).await
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
    // Copilot fronts several vendors, so usage is not guaranteed on the wire.
    #[serde(default)]
    usage: Option<openai::completion::Usage>,
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
        let response = self.embed_texts_with_usage(documents).await?;
        Ok(response.embeddings)
    }

    async fn embed_texts_with_usage(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
        let documents = documents.into_iter().collect::<Vec<_>>();
        let auth = self
            .client
            .ext()
            .auth
            .auth_context()
            .await
            .map_err(|err| EmbeddingError::ProviderError(err.to_string()))?;

        let headers = default_headers(&auth.api_key, "user", false, CopilotIntent::Panel);
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
        let status = response.status();
        if status.is_success() {
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
                        tracing::warn!(message = %err.error.message, "provider returned an error response");
                        return Err(EmbeddingError::from_http_response(
                            status,
                            String::from_utf8_lossy(&body).into_owned(),
                        ));
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

            // Embeddings consume only prompt tokens, so a missing usage
            // payload normalizes to the documented zero-usage sentinel.
            let usage = body
                .usage
                .as_ref()
                .map(|usage| usage.to_normalized())
                .unwrap_or_default();

            let embeddings = body
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
                .collect();

            Ok(embeddings::EmbeddingResponse { embeddings, usage })
        } else {
            let text = http_client::text(response).await?;
            Err(EmbeddingError::from_http_response(status, text))
        }
    }
}

const MODEL_LISTING_PATH: &str = "/models";
const MODEL_LISTING_PROVIDER: &str = "Copilot";

#[derive(Debug, Deserialize)]
struct ListModelsResponse {
    data: Vec<ListModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ListModelEntry {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    vendor: Option<String>,
    #[serde(default)]
    capabilities: Option<ListModelEntryCapabilities>,
}

#[derive(Debug, Deserialize)]
struct ListModelEntryCapabilities {
    #[serde(default, rename = "type")]
    r#type: Option<String>,
}

impl From<ListModelEntry> for Model {
    fn from(value: ListModelEntry) -> Self {
        let mut model = Model::from_id(value.id);
        model.name = value.name;
        model.owned_by = value.vendor;
        if let Some(caps) = value.capabilities {
            model.r#type = caps.r#type;
        }
        model
    }
}

/// [`ModelLister`] implementation for the GitHub Copilot API (`GET /models`).
#[derive(Clone)]
pub struct CopilotModelLister<H = reqwest::Client> {
    client: Client<H>,
}

impl<H> ModelLister<H> for CopilotModelLister<H>
where
    H: HttpClientExt + Clone + Debug + Default + WasmCompatSend + WasmCompatSync + 'static,
{
    type Client = Client<H>;

    fn new(client: Self::Client) -> Self {
        Self { client }
    }

    async fn list_all(&self) -> Result<ModelList, ModelListingError> {
        let auth = self.client.ext().auth.auth_context().await.map_err(|err| {
            ModelListingError::AuthError {
                message: err.to_string(),
            }
        })?;

        let headers = default_headers(&auth.api_key, "user", false, CopilotIntent::Panel);
        let req = apply_headers(
            get_with_auth_base(&self.client, &auth, MODEL_LISTING_PATH, Transport::Http)?,
            &headers,
        )
        .body(http_client::NoBody)?;

        let response = self.client.send::<_, Vec<u8>>(req).await.map_err(|error| {
            crate::providers::internal::model_listing::map_transport_error(
                MODEL_LISTING_PROVIDER,
                MODEL_LISTING_PATH,
                error,
            )
        })?;

        let api_resp: ListModelsResponse =
            crate::providers::internal::model_listing::decode_json_response(
                response,
                MODEL_LISTING_PROVIDER,
                MODEL_LISTING_PATH,
            )
            .await?;
        let models = api_resp.data.into_iter().map(Model::from).collect();

        Ok(ModelList::new(models))
    }
}

async fn send_copilot_chat_raw_streaming_request<T>(
    http_client: T,
    req: Request<Vec<u8>>,
) -> Result<crate::streaming::RawStreamingResult<CopilotStreamingResponse>, CompletionError>
where
    T: HttpClientExt + Clone + 'static,
{
    // Copilot's `/chat/completions` route relays OpenAI's chat-completions
    // SSE wire verbatim, so OpenAI's shared streaming profile (tolerant
    // deserializers, reasoning handling, finish-reason mapping) is the event
    // interpreter — only the auth/transport in the caller and the
    // route-carrying terminal wrapper below are Copilot-specific.
    let raw =
        openai::completion::streaming::send_compatible_raw_streaming_request(http_client, req)
            .await?;
    let stream = raw.map(|item| {
        item.and_then(|choice| {
            choice.try_map_final(|response| Ok(CopilotStreamingResponse::Chat(response)))
        })
    });

    Ok(Box::pin(stream))
}

fn default_token_dir() -> Option<PathBuf> {
    config_dir().map(|dir| dir.join("github_copilot"))
}

use crate::providers::internal::auth::config_dir;

#[cfg(test)]
mod tests {
    use super::{
        ChatApiErrorResponse, Client, CompletionRoute, CopilotIntent, TEXT_EMBEDDING_3_SMALL,
        base_url_from_token, default_headers, env_api_key, env_base_url, env_github_access_token,
        route_for_model,
    };
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel;
    use crate::http_client;
    use crate::providers::internal::openai_chat_completions_compatible::test_support::{
        sse_bytes_from_data_lines, sse_bytes_from_json_events,
    };
    use crate::providers::openai;
    use crate::streaming::StreamedAssistantContent;
    use crate::test_utils::MockStreamingClient;
    use crate::test_utils::{RecordingHttpClient, SequencedStreamingHttpClient};
    use futures::StreamExt;
    use std::collections::HashMap;

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

        let response: openai::completion::CompletionResponse =
            serde_json::from_str(json).expect("standard OpenAI response should deserialize");
        assert_eq!(response.id, "chatcmpl-abc123");
        assert_eq!(response.object, "chat.completion");
        assert_eq!(response.created, 1700000000);
        assert_eq!(response.model, "gpt-4o");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].finish_reason, "stop");
    }

    #[test]
    fn deserialize_copilot_response_without_object_and_created() {
        let response: openai::completion::CompletionResponse =
            serde_json::from_str(minimal_chat_response())
                .expect("Copilot response should deserialize");

        assert_eq!(response.id, "chatcmpl-123");
        assert_eq!(response.object, "");
        assert_eq!(response.created, 0);
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

        let response: openai::completion::CompletionResponse =
            serde_json::from_str(json).expect("Claude-via-Copilot response should deserialize");

        assert_eq!(response.model, "claude-3.5-sonnet");
        assert_eq!(response.choices[0].finish_reason, "");
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

    #[test]
    fn copilot_intent_headers_use_panel_by_default_and_edits_when_requested() {
        let panel_headers = default_headers("token", "user", false, CopilotIntent::default());
        assert_eq!(
            panel_headers
                .iter()
                .find(|(name, _)| *name == "openai-intent")
                .map(|(_, value)| value.as_str()),
            Some("conversation-panel")
        );

        let edits_headers = default_headers("token", "user", false, CopilotIntent::Edits);
        assert_eq!(
            edits_headers
                .iter()
                .find(|(name, _)| *name == "openai-intent")
                .map(|(_, value)| value.as_str()),
            Some("conversation-edits")
        );
    }

    #[test]
    fn copilot_completion_model_intent_builders_update_intent() {
        let client = Client::builder()
            .api_key("copilot-token")
            .build()
            .expect("build client");

        let default_model = client.completion_model("gpt-4o");
        assert_eq!(default_model.intent.as_header(), "conversation-panel");

        let edits_model = client
            .completion_model("gpt-4o")
            .with_intent(CopilotIntent::Edits);
        assert_eq!(edits_model.intent.as_header(), "conversation-edits");

        let panel_model = client
            .completion_model("gpt-4o")
            .with_edits_intent()
            .with_panel_intent();
        assert_eq!(panel_model.intent.as_header(), "conversation-panel");
    }

    #[test]
    fn base_url_from_token_derives_api_endpoint() {
        assert_eq!(
            base_url_from_token("tid=1;proxy-ep=proxy.individual.githubcopilot.com;exp=2")
                .as_deref(),
            Some("https://api.individual.githubcopilot.com")
        );
        assert_eq!(
            base_url_from_token("tid=1;proxy-ep=https://proxy.individual.githubcopilot.com;exp=2")
                .as_deref(),
            Some("https://api.individual.githubcopilot.com")
        );
        assert_eq!(base_url_from_token("tid=1;exp=2"), None);
    }

    #[test]
    fn base_url_from_token_rejects_unsafe_or_non_copilot_endpoints() {
        assert_eq!(
            base_url_from_token("tid=1;proxy-ep=http://proxy.individual.githubcopilot.com;exp=2"),
            None
        );
        assert_eq!(
            base_url_from_token("tid=1;proxy-ep=https://evil.example.com;exp=2"),
            None
        );
        assert_eq!(base_url_from_token("tid=1;proxy-ep=://bad;exp=2"), None);
        assert_eq!(base_url_from_token("tid=1;proxy-ep=;exp=2"), None);
        assert_eq!(
            base_url_from_token(
                "tid=1;proxy-ep=https://proxy.individual.githubcopilot.com/base;exp=2"
            ),
            None
        );
    }

    #[tokio::test]
    async fn api_key_with_proxy_endpoint_overrides_base_url() {
        let http_client = RecordingHttpClient::new(minimal_chat_response());
        let client = Client::builder()
            .api_key("tid=1;proxy-ep=proxy.individual.githubcopilot.com;exp=2")
            .http_client(http_client.clone())
            .build()
            .expect("build client");
        let model = client.completion_model("gpt-4o");
        let request = model.completion_request("hello").build();

        let _response = model.completion(request).await.expect("chat completion");

        let requests = http_client.requests();
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0]
                .uri
                .starts_with("https://api.individual.githubcopilot.com"),
            "expected proxy-derived base URL, got {}",
            requests[0].uri
        );
    }

    #[tokio::test]
    async fn explicit_base_url_wins_over_token_proxy_endpoint() {
        let http_client = RecordingHttpClient::new(minimal_chat_response());
        let client = Client::builder()
            .api_key("tid=1;proxy-ep=proxy.individual.githubcopilot.com;exp=2")
            .base_url("https://custom.example.com")
            .http_client(http_client.clone())
            .build()
            .expect("build client");
        let model = client.completion_model("gpt-4o");
        let request = model.completion_request("hello").build();

        let _response = model.completion(request).await.expect("chat completion");

        let requests = http_client.requests();
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0].uri.starts_with("https://custom.example.com"),
            "expected explicit base URL, got {}",
            requests[0].uri
        );
    }

    #[tokio::test]
    async fn completion_model_edits_intent_sets_request_header() {
        let http_client = RecordingHttpClient::new(minimal_chat_response());
        let client = Client::builder()
            .api_key("copilot-token")
            .http_client(http_client.clone())
            .build()
            .expect("build client");
        let model = client.completion_model("gpt-4o").with_edits_intent();
        let request = model.completion_request("hello").build();

        let _response = model.completion(request).await.expect("chat completion");

        let requests = http_client.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0]
                .headers
                .get("openai-intent")
                .and_then(|value| value.to_str().ok()),
            Some("conversation-edits")
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
            "output_index": 0,
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

        // The fully-delivered tool call is content, so it is flushed *before*
        // the terminal error: consumers that stop at the first `Err` still
        // see the completed work.
        let tool_call = stream
            .next()
            .await
            .expect("fully-delivered tool call should be flushed before the error")
            .expect("flushed tool call should not be an error");
        assert!(
            matches!(
                tool_call,
                StreamedAssistantContent::ToolCall { ref tool_call, .. }
                    if tool_call.function.name == "example_tool"
            ),
            "expected the flushed tool call, got {tool_call:?}"
        );
        let err = match stream.next().await.expect("stream should yield an item") {
            Ok(item) => panic!("stream should surface a provider error, got {item:?}"),
            Err(err) => err,
        };
        // The terminal `response.failed` event carries the provider's error
        // payload, so the full raw event JSON is preserved for inspection
        // (status: None — the error arrived over an already-established stream),
        // matching the OpenAI Responses SSE path.
        assert!(matches!(
            err,
            crate::completion::CompletionError::ProviderResponse(_)
        ));
        assert_eq!(err.provider_response_status(), None);
        let json = err
            .provider_response_json()
            .expect("preserved body should parse as JSON")
            .expect("preserved body should not be empty");
        let response_error = json
            .get("response")
            .and_then(|response| response.get("error"))
            .expect("preserved body should retain the provider error object");
        assert_eq!(
            response_error.get("code").and_then(|value| value.as_str()),
            Some("server_error")
        );
        assert_eq!(
            response_error
                .get("message")
                .and_then(|value| value.as_str()),
            Some("Copilot response stream failed")
        );
        assert!(
            stream.next().await.is_none(),
            "responses stream should end without a terminal record after a terminal error"
        );
    }

    #[tokio::test]
    async fn responses_stream_object_less_failed_still_attaches_the_raw_event() {
        // #2258 F4 decision: the old Copilot code kept a deliberate two-tier
        // shape — `response.failed` WITHOUT an error object surfaced as a
        // `ProviderError` with `provider_response_body() == None`. The shared
        // Responses interpreter unifies this: the raw event body is ALWAYS
        // attached, error object or not, so callers can inspect what the
        // provider actually sent. Documented in MIGRATING.
        let failed = serde_json::json!({
            "type": "response.failed",
            "sequence_number": 1,
            "response": {
                "id": "resp_123",
                "object": "response",
                "created_at": 1700000000,
                "status": "failed",
                "error": null,
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
            sse_bytes: sse_bytes_from_json_events(&[failed]),
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
            Ok(item) => panic!("stream should surface a provider error, got {item:?}"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            crate::completion::CompletionError::ProviderResponse(_)
        ));
        assert_eq!(err.provider_response_status(), None);
        assert!(
            err.provider_response_body()
                .is_some_and(|body| body.contains("response.failed")),
            "an object-less response.failed must still carry the raw event body"
        );
        assert!(
            stream.next().await.is_none(),
            "responses stream should end after the terminal error"
        );
    }

    #[tokio::test]
    async fn responses_stream_incomplete_is_a_terminal_with_partial_content() {
        // The content exists only in the delta; the terminal
        // `response.incomplete` body has an empty `output`.
        let text_delta = serde_json::json!({
            "type": "response.output_text.delta",
            "content_index": 0,
            "delta": "partial",
            "item_id": "msg_1",
            "logprobs": [],
            "output_index": 0,
            "sequence_number": 1
        });
        let incomplete = serde_json::json!({
            "type": "response.incomplete",
            "sequence_number": 2,
            "response": {
                "id": "resp_123",
                "object": "response",
                "created_at": 1700000000,
                "status": "incomplete",
                "error": null,
                "incomplete_details": { "reason": "max_output_tokens" },
                "instructions": null,
                "max_output_tokens": null,
                "model": "gpt-5.3-codex",
                "usage": { "input_tokens": 1, "output_tokens": 2, "total_tokens": 3 },
                "output": [],
                "tools": []
            }
        });
        let http_client = MockStreamingClient {
            sse_bytes: sse_bytes_from_json_events(&[text_delta, incomplete]),
        };
        let client = Client::builder()
            .api_key("copilot-token")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model("gpt-5.3-codex");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        let mut text = String::new();
        let mut terminal = None;
        while let Some(item) = stream.next().await {
            match item.expect("incomplete turn should not surface an error") {
                StreamedAssistantContent::Text(chunk) => text.push_str(&chunk.text),
                StreamedAssistantContent::Final(final_response) => terminal = Some(final_response),
                other => panic!("unexpected stream item: {other:?}"),
            }
        }

        assert_eq!(text, "partial");
        let terminal = terminal.expect("incomplete turn should emit a terminal record");
        assert_eq!(
            terminal.finish_reason,
            Some(crate::completion::FinishReason::Length)
        );
        assert_eq!(terminal.usage.input_tokens, 1);
        assert_eq!(terminal.usage.output_tokens, 2);
        assert_eq!(terminal.usage.total_tokens, 3);
    }

    #[tokio::test]
    async fn chat_stream_surfaces_malformed_frame_and_still_completes() {
        let http_client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                "{\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}],\"usage\":null}",
                "{not valid json",
                "{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":null}",
                "[DONE]",
            ]),
        };
        let client = Client::builder()
            .api_key("copilot-token")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model("gpt-4o");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        let mut text = String::new();
        let mut saw_error = false;
        let mut terminal = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamedAssistantContent::Text(chunk)) => text.push_str(&chunk.text),
                Ok(StreamedAssistantContent::Final(final_response)) => {
                    terminal = Some(final_response)
                }
                Ok(other) => panic!("unexpected stream item: {other:?}"),
                Err(err) => {
                    assert!(
                        matches!(err, crate::completion::CompletionError::JsonError(_)),
                        "expected a JSON parse error item, got {err:?}"
                    );
                    saw_error = true;
                }
            }
        }

        // The malformed frame is surfaced as an error item, and the content
        // and genuine terminal on either side of it both still arrive.
        assert_eq!(text, "hello");
        assert!(saw_error, "malformed frame should surface an error item");
        let terminal = terminal.expect("stream should still emit its terminal record");
        assert_eq!(
            terminal.finish_reason,
            Some(crate::completion::FinishReason::Stop)
        );
    }

    #[tokio::test]
    async fn chat_stream_surfaces_recognizable_chunk_with_malformed_field() {
        // The frame is recognizably a chat completion chunk (it has
        // `choices`), but the payload fails the full parse — a data-level
        // defect surfaced as an error item, not a skippable unknown event.
        let http_client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                "{\"object\":\"chat.completion.chunk\",\"choices\":42}",
                "{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":null}",
                "[DONE]",
            ]),
        };
        let client = Client::builder()
            .api_key("copilot-token")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model("gpt-4o");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        let mut saw_error = false;
        let mut terminal = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamedAssistantContent::Final(final_response)) => {
                    terminal = Some(final_response)
                }
                Ok(other) => panic!("unexpected stream item: {other:?}"),
                Err(err) => {
                    assert!(
                        matches!(err, crate::completion::CompletionError::JsonError(_)),
                        "expected a JSON parse error item, got {err:?}"
                    );
                    saw_error = true;
                }
            }
        }

        assert!(
            saw_error,
            "a recognizable chunk with a malformed field should surface an error item"
        );
        let terminal = terminal.expect("stream should still emit its terminal record");
        assert_eq!(
            terminal.finish_reason,
            Some(crate::completion::FinishReason::Stop)
        );
    }

    #[tokio::test]
    async fn chat_stream_skips_unrecognized_event_and_still_completes() {
        // Valid JSON that is not recognizably a chat completion chunk (no
        // `choices`, no `"object": "chat.completion.chunk"`) is an event this
        // client doesn't know yet — skipped semantically for forward
        // compatibility, surfaced verbatim on the raw passthrough channel.
        let http_client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                "{\"type\":\"copilot.heartbeat\",\"payload\":{}}",
                "{\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}],\"usage\":null}",
                "{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":null}",
                "[DONE]",
            ]),
        };
        let client = Client::builder()
            .api_key("copilot-token")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model("gpt-4o");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        let mut text = String::new();
        let mut terminal = None;
        let mut unknown = None;
        while let Some(item) = stream.next().await {
            match item.expect("unrecognized events must not surface errors") {
                StreamedAssistantContent::Text(chunk) => text.push_str(&chunk.text),
                StreamedAssistantContent::Final(final_response) => terminal = Some(final_response),
                StreamedAssistantContent::Unknown(value) => unknown = Some(value),
                other => panic!("unexpected stream item: {other:?}"),
            }
        }

        assert_eq!(text, "hello");
        assert_eq!(
            unknown,
            Some(serde_json::json!({"type": "copilot.heartbeat", "payload": {}}).into()),
            "the unrecognized frame must surface verbatim on the raw channel"
        );
        let terminal = terminal.expect("stream should still emit its terminal record");
        assert_eq!(
            terminal.finish_reason,
            Some(crate::completion::FinishReason::Stop)
        );
    }

    #[tokio::test]
    async fn responses_stream_preserves_reasoning_metadata_on_final_response() {
        let metadata = serde_json::json!({
            "context": "all_turns",
            "effort": "ultra",
            "summary": null,
            "future_control": true
        });
        let completed = serde_json::json!({
            "type": "response.completed",
            "sequence_number": 1,
            "response": {
                "id": "resp_123",
                "object": "response",
                "created_at": 1700000000,
                "status": "completed",
                "error": null,
                "incomplete_details": null,
                "instructions": null,
                "max_output_tokens": null,
                "model": "gpt-5.3-codex",
                "reasoning": metadata.clone(),
                "usage": null,
                "output": [],
                "tools": []
            }
        });
        let http_client = MockStreamingClient {
            sse_bytes: sse_bytes_from_json_events(&[completed]),
        };
        let client = Client::builder()
            .api_key("copilot-token")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model("gpt-5.3-codex");
        let request = model.completion_request("hello").build();
        // Reasoning metadata is Copilot's own terminal payload, not part of
        // the normalized `StreamFinal`, so this reads it through `raw_stream`.
        let mut stream = model
            .raw_stream(request)
            .await
            .expect("stream should start");

        while let Some(item) = stream.next().await {
            if let crate::streaming::RawStreamingChoice::FinalResponse(
                super::CopilotStreamingResponse::Responses(response),
            ) = item.expect("completed stream should not error")
            {
                assert_eq!(response.reasoning_context.as_deref(), Some("all_turns"));
                assert_eq!(response.reasoning_metadata.as_ref(), metadata.as_object());
                return;
            }
        }

        panic!("responses stream should yield a final response");
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

        // The fully-delivered tool call is content, so it is flushed *before*
        // the terminal error: consumers that stop at the first `Err` still
        // see the completed work.
        let mut saw_error = false;
        let mut saw_tool_call = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamedAssistantContent::ToolCallDelta { .. }) => {
                    assert!(!saw_error, "deltas should precede the terminal error");
                }
                Ok(StreamedAssistantContent::ToolCall { tool_call, .. }) => {
                    assert!(
                        !saw_error,
                        "flushed tool call should precede the terminal error"
                    );
                    assert_eq!(tool_call.function.name, "ping");
                    saw_tool_call = true;
                }
                Err(err) => {
                    assert_eq!(
                        err.to_string(),
                        "HttpError: Invalid status code: 502 Bad Gateway"
                    );
                    assert_eq!(
                        err.provider_response_status(),
                        Some(http::StatusCode::BAD_GATEWAY)
                    );
                    assert_eq!(err.provider_response_body(), None);
                    saw_error = true;
                }
                Ok(other) => panic!("unexpected stream item: {other:?}"),
            }
        }

        assert!(
            saw_tool_call,
            "fully-delivered tool call should be flushed before the error"
        );
        assert!(saw_error, "stream should surface the transport error");
        assert!(
            stream.next().await.is_none(),
            "chat stream should end without a terminal record after a transport error"
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

#[cfg(test)]
mod response_identity_tests {
    use super::*;

    /// Both Copilot routes' streaming terminals carry the transport request id
    /// (stamped by the shared SSE capture) into the normalized `StreamFinal`.
    /// Deterministic and credential-free: the transport halves — the shared
    /// OpenAI chat wrapper's capture and `stamp_terminal_request_id` on the
    /// Responses route — are covered by the shared-path tests; this locks the
    /// Copilot-specific conversion layer.
    #[test]
    fn streaming_terminals_carry_request_id_into_stream_final() {
        let mut chat_terminal = openai::completion::streaming::StreamingCompletionResponse::<
            openai::completion::Usage,
        >::new(openai::completion::Usage::default());
        chat_terminal.provider_request_id = Some("req-chat".to_string());
        let chat_final: crate::streaming::StreamFinal =
            (PROVIDER_NAME, CopilotStreamingResponse::Chat(chat_terminal)).into();
        assert_eq!(chat_final.provider_request_id.as_deref(), Some("req-chat"));

        let mut responses_terminal = responses_api::streaming::StreamingCompletionResponse::new(
            serde_json::from_value(
                serde_json::json!({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
            )
            .expect("usage should parse"),
        );
        responses_terminal.provider_request_id = Some("req-responses".to_string());
        let responses_final: crate::streaming::StreamFinal = (
            PROVIDER_NAME,
            CopilotStreamingResponse::Responses(responses_terminal),
        )
            .into();
        assert_eq!(
            responses_final.provider_request_id.as_deref(),
            Some("req-responses")
        );
    }

    /// The Responses-route unary wire type carries the stamped id through
    /// `normalize` into the core response; the chat route has no wire slot,
    /// so `completion()` stamps the normalized response from the returned
    /// pair — asserted here at the conversion layer for the responses half.
    #[test]
    fn responses_unary_wire_id_survives_normalize() {
        use crate::completion::NormalizeCompletionResponse;

        let payload = serde_json::json!({
            "id": "resp_123",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-test",
            "output": [{
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hi", "annotations": []}]
            }]
        });
        let mut response: responses_api::CompletionResponse =
            serde_json::from_value(payload).expect("wire response should parse");
        response.provider_request_id = Some("req-unary".to_string());

        let normalized = response
            .normalize(PROVIDER_NAME)
            .expect("response should normalize");
        assert_eq!(normalized.provider_request_id.as_deref(), Some("req-unary"));
        assert_eq!(normalized.response_id.as_deref(), Some("resp_123"));
        assert_eq!(normalized.provider, PROVIDER_NAME);
    }
}

/// Raw-capture and Part A parity, unit form, for both Copilot routes over the
/// recording mock transport. `with_error_response_headers` with `200 OK` is
/// the one unary double that carries response headers, which is what lets a
/// unit test exercise the `x-request-id` half of the contract: on the chat
/// route the id lives only on the header (the shared OpenAI chat wire type has
/// no slot), on the responses route the driver stamps it onto the wire type.
/// The captured value is the route-tagged [`CopilotCompletionResponse`] — what
/// `raw_completion` returns — so it must round-trip through the
/// `#[serde(tag = "api")]` enum, including the responses variant whose inner
/// type has a hand-written `Serialize`.
#[cfg(test)]
mod raw_capture_tests {
    use super::*;
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel as _;
    use crate::test_utils::RecordingHttpClient;

    const REQUEST_ID: &str = "req_unit_copilot_0001";

    /// A chat-completions body carrying `system_fingerprint`, which the
    /// normalized response provably lacks.
    const CHAT_BODY: &str = r#"{
        "id": "chatcmpl-copilot-raw",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-2024-11-20",
        "system_fingerprint": "fp_copilot_chat",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "logprobs": null,
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
    }"#;

    /// A Responses body carrying `service_tier`, which the normalized
    /// response provably lacks.
    const RESPONSES_BODY: &str = r#"{
        "id": "resp_copilot_raw",
        "object": "response",
        "created_at": 1700000000,
        "status": "completed",
        "error": null,
        "incomplete_details": null,
        "instructions": null,
        "max_output_tokens": null,
        "model": "gpt-5.3-codex",
        "service_tier": "default",
        "usage": {
            "input_tokens": 4,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 7
        },
        "output": [{
            "type": "message",
            "id": "msg_copilot_raw",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hello", "annotations": []}]
        }],
        "tools": []
    }"#;

    fn model(model: &str, body: &'static str) -> CompletionModel<RecordingHttpClient> {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-request-id", http::HeaderValue::from_static(REQUEST_ID));
        let http_client =
            RecordingHttpClient::with_error_response_headers(http::StatusCode::OK, body, headers);
        let client = Client::builder()
            .api_key("copilot-token")
            .http_client(http_client)
            .build()
            .expect("build client");
        client.completion_model(model)
    }

    /// Run one completion for a route and check the shared capture contract:
    /// `raw` deserializes into [`CopilotCompletionResponse`] under the
    /// expected route tag and re-serializes identically; re-normalizing the
    /// capture (with the header id reattached, exactly as `completion()`
    /// does) reproduces every normalized field; and the response reports the
    /// header's id.
    async fn assert_capture_contract(
        model: &CompletionModel<RecordingHttpClient>,
        expected_api_tag: &str,
    ) -> (completion::CompletionResponse, CopilotCompletionResponse) {
        let response = model
            .completion(model.completion_request("hello").build())
            .await
            .expect("completion");
        let raw = &response.raw;
        assert_eq!(raw["api"], expected_api_tag);
        let typed: CopilotCompletionResponse =
            serde_json::from_value(raw.clone()).expect("raw must deserialize");
        assert_eq!(
            serde_json::to_value(&typed).expect("re-serialize"),
            *raw,
            "the capture must be exactly what the route-tagged raw type serializes to"
        );

        let renormalized = typed
            .clone()
            .normalize(PROVIDER_NAME)
            .expect("re-normalize the capture")
            .with_optional_provider_request_id(Some(REQUEST_ID.to_string()));
        assert_eq!(response.identity(), renormalized.identity());
        assert_eq!(response.finish_reason(), renormalized.finish_reason());
        assert_eq!(response.model, renormalized.model);
        assert_eq!(response.usage, renormalized.usage);
        assert_eq!(response.choice, renormalized.choice);
        assert_eq!(response.provider_request_id.as_deref(), Some(REQUEST_ID));
        (response, typed)
    }

    /// Part A parity for one route: `raw_completion_with_request_id` →
    /// `normalize` → `with_optional_provider_request_id` reproduces
    /// `completion()` on identity, finish reason, model and usage, and the
    /// id is the header on both.
    async fn assert_parity_contract(model: &CompletionModel<RecordingHttpClient>) {
        let (raw, id) = model
            .raw_completion_with_request_id(model.completion_request("hello").build())
            .await
            .expect("typed route");
        assert_eq!(id.as_deref(), Some(REQUEST_ID));
        let reassembled = raw
            .normalize(PROVIDER_NAME)
            .expect("normalize")
            .with_optional_provider_request_id(id);

        let normalized = model
            .completion(model.completion_request("hello").build())
            .await
            .expect("normalized route");

        assert_eq!(reassembled.identity(), normalized.identity());
        assert_eq!(reassembled.finish_reason(), normalized.finish_reason());
        assert_eq!(reassembled.model, normalized.model);
        assert_eq!(reassembled.usage, normalized.usage);
        assert_eq!(reassembled.provider_request_id.as_deref(), Some(REQUEST_ID));
        assert_eq!(normalized.provider_request_id.as_deref(), Some(REQUEST_ID));
        assert_eq!(normalized.provider, PROVIDER_NAME);
    }

    /// Chat route: the capture is tagged `api: chat`, wraps the shared OpenAI
    /// chat wire type, and keeps `system_fingerprint`.
    #[tokio::test]
    async fn chat_route_raw_round_trips_into_the_route_tagged_type() {
        let model = model("gpt-4o", CHAT_BODY);

        let (response, typed) = assert_capture_contract(&model, "chat").await;

        let CopilotCompletionResponse::Chat(chat) = typed else {
            panic!("the chat route must capture the chat variant");
        };
        assert_eq!(chat.system_fingerprint.as_deref(), Some("fp_copilot_chat"));
        assert_eq!(
            response.finish_reason(),
            Some(completion::FinishReason::Stop)
        );
        assert_eq!(
            response.identity().response_id.as_deref(),
            Some("chatcmpl-copilot-raw")
        );
    }

    /// Chat route Part A: the wire type has no id slot, so only the pair
    /// reproduces `completion()` — this is the case the method exists for.
    #[tokio::test]
    async fn chat_route_raw_completion_with_request_id_reproduces_completion() {
        let model = model("gpt-4o", CHAT_BODY);

        assert_parity_contract(&model).await;

        // And plain `raw_completion` → `normalize` provably lacks the id:
        // the reason the pair is public.
        let raw = model
            .raw_completion(model.completion_request("hello").build())
            .await
            .expect("typed route");
        let normalized = raw.normalize(PROVIDER_NAME).expect("normalize");
        assert_eq!(normalized.provider_request_id, None);
    }

    /// Responses route: the capture is tagged `api: responses` and wraps the
    /// Responses wire type, whose hand-written `Serialize` mirrors the body
    /// (`service_tier` kept; the stamped transport id, which is not body,
    /// deliberately not emitted — so the deserialized capture reports `None`
    /// there while the normalized response beside it carries the header).
    #[tokio::test]
    async fn responses_route_raw_round_trips_into_the_route_tagged_type() {
        let model = model("gpt-5.3-codex", RESPONSES_BODY);

        let (response, typed) = assert_capture_contract(&model, "responses").await;

        let CopilotCompletionResponse::Responses(responses) = typed else {
            panic!("the responses route must capture the responses variant");
        };
        assert!(matches!(
            responses.additional_parameters.service_tier,
            Some(responses_api::OpenAIServiceTier::Default)
        ));
        assert_eq!(responses.provider_request_id, None);
        assert_eq!(
            response.identity().message_id.as_deref(),
            Some("msg_copilot_raw")
        );
        assert_eq!(
            response.identity().response_id.as_deref(),
            Some("resp_copilot_raw")
        );
    }

    /// Responses route Part A: the wire type carries the id itself, so the
    /// pair's second element equals the raw type's own id and reattaching it
    /// is a no-op — the same pair still reproduces `completion()`.
    #[tokio::test]
    async fn responses_route_raw_completion_with_request_id_reproduces_completion() {
        let model = model("gpt-5.3-codex", RESPONSES_BODY);

        assert_parity_contract(&model).await;

        let (raw, id) = model
            .raw_completion_with_request_id(model.completion_request("hello").build())
            .await
            .expect("typed route");
        let CopilotCompletionResponse::Responses(responses) = &raw else {
            panic!("codex models route to /responses");
        };
        assert_eq!(responses.provider_request_id, id);
        assert_eq!(id.as_deref(), Some(REQUEST_ID));
    }

    /// Both variants of the route-tagged unary raw type round-trip through
    /// serde, hand-built from parsed wire bodies rather than through the
    /// transport: the internally tagged enum has to merge its `api` tag into
    /// whatever the inner type serializes as, and the responses variant's
    /// inner type serializes through a hand-written `Serialize` (with a
    /// flattened tail) rather than a derive.
    #[test]
    fn copilot_completion_response_round_trips_both_variants() {
        let chat: openai::completion::CompletionResponse =
            serde_json::from_str(CHAT_BODY).expect("chat body parses");
        let responses: responses_api::CompletionResponse =
            serde_json::from_str(RESPONSES_BODY).expect("responses body parses");

        for (variant, tag) in [
            (CopilotCompletionResponse::Chat(Box::new(chat)), "chat"),
            (
                CopilotCompletionResponse::Responses(Box::new(responses)),
                "responses",
            ),
        ] {
            let value = serde_json::to_value(&variant).expect("serialize");
            assert_eq!(value["api"], tag);
            let back: CopilotCompletionResponse =
                serde_json::from_value(value.clone()).expect("deserialize");
            assert_eq!(
                serde_json::to_value(&back).expect("re-serialize"),
                value,
                "{tag}: the route-tagged raw type must round-trip"
            );
            assert_eq!(
                back.normalize(PROVIDER_NAME).expect("normalize").provider,
                PROVIDER_NAME
            );
        }
    }

    /// Both variants of the route-tagged streaming terminal round-trip
    /// through serde — this is the value `StreamFinal.raw` carries for a
    /// Copilot stream, so a consumer must be able to read it back as
    /// [`CopilotStreamingResponse`].
    #[test]
    fn copilot_streaming_response_round_trips_both_variants() {
        let mut chat = openai::completion::streaming::StreamingCompletionResponse::<
            openai::completion::Usage,
        >::new(openai::completion::Usage::default());
        chat.finish_reason = Some(completion::FinishReason::Stop);
        chat.response_id = Some("chatcmpl-stream".to_string());
        chat.model = Some("gpt-4o".to_string());
        chat.provider_request_id = Some("req-chat".to_string());
        chat.additional_params = Some(
            serde_json::from_value(json!({"service_tier": "default"})).expect("additional params"),
        );

        let mut responses = responses_api::streaming::StreamingCompletionResponse::new(
            serde_json::from_value(
                json!({"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}),
            )
            .expect("usage should parse"),
        );
        responses.provider_request_id = Some("req-responses".to_string());

        for (variant, tag) in [
            (CopilotStreamingResponse::Chat(chat), "chat"),
            (CopilotStreamingResponse::Responses(responses), "responses"),
        ] {
            let value = serde_json::to_value(&variant).expect("serialize");
            assert_eq!(value["api"], tag);
            let back: CopilotStreamingResponse =
                serde_json::from_value(value.clone()).expect("deserialize");
            assert_eq!(
                serde_json::to_value(&back).expect("re-serialize"),
                value,
                "{tag}: the route-tagged terminal must round-trip"
            );
            let original: crate::streaming::StreamFinal = (PROVIDER_NAME, variant).into();
            let restored: crate::streaming::StreamFinal = (PROVIDER_NAME, back).into();
            assert_eq!(
                restored, original,
                "{tag}: normalization must agree across the round-trip"
            );
        }
    }
}
