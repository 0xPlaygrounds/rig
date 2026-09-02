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
//! ```ignore
//! use rig_core::client::{CompletionClient};
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
    self, ApiKey, HasCompletion, HasEmbeddings, HasModelListing, ModelLister, ModelTransport,
    Provider, ProviderClientResult,
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

/// Builder settings for [`Copilot`]: token cache locations and the
/// device-code login policy.
#[derive(Debug, Clone)]
pub struct CopilotConfig {
    access_token_file: Option<PathBuf>,
    api_key_file: Option<PathBuf>,
    device_code_handler: auth::DeviceCodeHandler,
    allow_device_flow: bool,
}

/// The GitHub Copilot provider. Authentication is a runtime token exchange
/// driven by the `auth::Authenticator` built from the key the client was
/// given; requests carry the exchanged token, not a default header.
#[derive(Clone)]
pub struct Copilot {
    auth: auth::Authenticator,
}

impl Debug for Copilot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Copilot").field("auth", &self.auth).finish()
    }
}

pub type Client<H = crate::http_client::BoxedHttpClient> = client::Client<Copilot, H>;
pub type ClientBuilder<H = crate::markers::Missing> = client::ClientBuilder<Copilot, H>;

impl Default for CopilotConfig {
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

impl Provider for Copilot {
    const NAME: &'static str = PROVIDER_NAME;
    const BASE_URL: &'static str = GITHUB_COPILOT_API_BASE_URL;
    const VERIFY_PATH: &'static str = "";
    type ApiKey = CopilotAuth;
    type Config = CopilotConfig;
    type EnvInput = CopilotAuth;

    fn build(config: CopilotConfig, api_key: &CopilotAuth) -> http_client::Result<Self> {
        let auth = match api_key {
            CopilotAuth::ApiKey(api_key) => auth::AuthSource::ApiKey(api_key.clone()),
            CopilotAuth::GitHubAccessToken(access_token) => {
                auth::AuthSource::GitHubAccessToken(access_token.clone())
            }
            CopilotAuth::OAuth => auth::AuthSource::OAuth,
        };

        Ok(Copilot {
            auth: auth::Authenticator::new(
                auth,
                config.access_token_file,
                config.api_key_file,
                config.device_code_handler,
                config.allow_device_flow,
            ),
        })
    }

    fn from_env<H: HttpClientExt>(http: H) -> ProviderClientResult<Client<H>> {
        let mut builder = Client::builder();
        fn get(name: &str) -> Option<String> {
            std::env::var(name).ok()
        }

        if let Some(base_url) = env_base_url(&get) {
            builder = builder.base_url(base_url);
        }

        if let Some(api_key) = env_api_key(&get) {
            builder.api_key(api_key).http_client(http).build()
        } else if let Some(access_token) = env_github_access_token(&get) {
            builder
                .github_access_token(access_token)
                .http_client(http)
                .build()
        } else {
            builder.oauth().http_client(http).build()
        }
    }

    fn from_val<H: HttpClientExt>(input: CopilotAuth, http: H) -> ProviderClientResult<Client<H>> {
        Client::new_with(input, http)
    }
}

impl HasCompletion for Copilot {
    type Model<H>
        = CompletionModel<H>
    where
        H: ModelTransport;

    fn completion_model<H: ModelTransport>(client: &Client<H>, model: String) -> Self::Model<H> {
        CompletionModel::new(client.clone(), model)
    }
}

impl HasEmbeddings for Copilot {
    type Model<H>
        = EmbeddingModel<H>
    where
        H: ModelTransport;

    fn embedding_model<H: ModelTransport>(
        client: &Client<H>,
        model: String,
        ndims: Option<usize>,
    ) -> Self::Model<H> {
        EmbeddingModel::make(client, model, ndims)
    }
}

impl HasModelListing for Copilot {
    type Lister<H>
        = CopilotModelLister<H>
    where
        H: ModelTransport;

    fn model_lister<H: ModelTransport>(client: &Client<H>) -> Self::Lister<H> {
        CopilotModelLister::new(client.clone())
    }
}

impl<H> ClientBuilder<H> {
    pub fn github_access_token(self, access_token: impl Into<String>) -> Self {
        self.api_key(CopilotAuth::GitHubAccessToken(access_token.into()))
    }

    pub fn oauth(self) -> Self {
        self.api_key(CopilotAuth::OAuth)
    }
}

impl<H> ClientBuilder<H> {
    pub fn on_device_code<F>(self, handler: F) -> Self
    where
        F: Fn(auth::DeviceCodePrompt) + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map_config(|mut ext| {
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
        self.map_config(|mut ext| {
            ext.allow_device_flow = allow;
            ext
        })
    }

    pub fn token_dir(self, path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        self.map_config(|mut ext| {
            ext.access_token_file = Some(path.join("access-token"));
            ext.api_key_file = Some(path.join("api-key.json"));
            ext
        })
    }

    pub fn access_token_file(self, path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_path_buf();
        self.map_config(|mut ext| {
            ext.access_token_file = Some(path);
            ext
        })
    }

    pub fn api_key_file(self, path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_path_buf();
        self.map_config(|mut ext| {
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
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    pub async fn authorize(&self) -> Result<(), auth::AuthError> {
        self.provider()
            .auth
            .auth_context(self.http_client())
            .await
            .map(|_| ())
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
) -> http_client::Result<http_client::Builder> {
    let uri = client
        .provider()
        .build_uri(runtime_base_url(client, auth).as_ref(), path);
    let mut req = Request::post(uri);

    if let Some(headers) = req.headers_mut() {
        headers.extend(client.headers().iter().map(|(k, v)| (k.clone(), v.clone())));
    }

    client.provider().prepare(req)
}

fn get_with_auth_base<H>(
    client: &Client<H>,
    auth: &auth::AuthContext,
    path: &str,
) -> http_client::Result<http_client::Builder> {
    let uri = client
        .provider()
        .build_uri(runtime_base_url(client, auth).as_ref(), path);
    let mut req = Request::get(uri);

    if let Some(headers) = req.headers_mut() {
        headers.extend(client.headers().iter().map(|(k, v)| (k.clone(), v.clone())));
    }

    client.provider().prepare(req)
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
            system_instructions: request.system_instructions().map(str::to_owned),
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
pub struct CompletionModel<H = crate::http_client::BoxedHttpClient> {
    client: Client<H>,
    pub model: String,
    pub strict_tools: bool,
    pub tool_result_array_content: bool,
    pub intent: CopilotIntent,
}

impl<H> CompletionModel<H>
where
    Client<H>: HttpClientExt + Clone + 'static,
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
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
            .provider()
            .auth
            .auth_context(self.client.http_client())
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
            supports_image_tool_results: false,
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
        let req = apply_headers(post_with_auth_base(&self.client, &auth, path)?, &headers)
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
                    .map(super::openai::completion::Usage::to_normalized)
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

    async fn stream_chat(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<crate::streaming::StreamingResult, CompletionError> {
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

    async fn stream_responses(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<crate::streaming::StreamingResult, CompletionError> {
        let facts = RequestFacts::capture(&completion_request);
        let mut request = self.responses_request(completion_request)?;
        request.stream = Some(true);
        let (req, span) = self
            .signed_request(
                &facts,
                "/responses",
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
        // verbatim, so the shared Responses adapter is the event interpreter
        // — only the auth/transport above is Copilot-specific; the terminal
        // record is attributed to Copilot.
        let stream = responses_api::streaming::responses_stream_from_event_source(
            PROVIDER_NAME,
            event_source,
            span,
        );
        Ok(
            crate::providers::internal::sse_transport::stamp_terminal_request_id(
                stream,
                Some(request_id_slot),
                Some("x-request-id"),
            ),
        )
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

    /// Open a stream on whichever route this model is configured for.
    async fn stream_normalized(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        let stream = match self.route() {
            CompletionRoute::ChatCompletions => self.stream_chat(completion_request).await?,
            CompletionRoute::Responses => self.stream_responses(completion_request).await?,
        };
        Ok(StreamingCompletionResponse::stream(PROVIDER_NAME, stream))
    }
}

impl<H> completion::CompletionModel for CompletionModel<H>
where
    Client<H>: HttpClientExt + Clone + 'static,
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
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
pub struct EmbeddingModel<H = crate::http_client::BoxedHttpClient> {
    client: Client<H>,
    pub model: String,
    pub encoding_format: Option<openai::EncodingFormat>,
    pub user: Option<String>,
    ndims: usize,
}

/// Copilot's embeddings wire response: what
/// [`EmbeddingModel::raw_embed_texts`] returns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopilotEmbeddingResponse {
    pub data: Vec<CopilotEmbeddingData>,
    // Copilot fronts several vendors, so usage is not guaranteed on the wire.
    #[serde(default)]
    pub usage: Option<openai::completion::Usage>,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopilotEmbeddingData {
    pub embedding: Vec<serde_json::Number>,
}

impl embeddings::NormalizeEmbeddingResponse for CopilotEmbeddingResponse {
    fn normalize(
        self,
        provider: &str,
        documents: Vec<String>,
    ) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
        // Embeddings consume only prompt tokens, so a missing usage
        // payload normalizes to the documented zero-usage sentinel.
        let usage = self
            .usage
            .as_ref()
            .map(super::openai::completion::Usage::to_normalized)
            .unwrap_or_default();

        let embeddings = self
            .data
            .into_iter()
            .zip(documents)
            .map(|(embedding, document)| embeddings::Embedding {
                document,
                vec: embedding
                    .embedding
                    .into_iter()
                    .filter_map(|n| n.as_f64())
                    .collect(),
            })
            .collect();

        Ok(embeddings::EmbeddingResponse::new(embeddings, provider)
            .with_optional_model(self.model)
            .with_usage(usage))
    }
}

impl<H> EmbeddingModel<H>
where
    Client<H>: HttpClientExt + Clone + 'static,
    H: Clone + 'static,
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

impl<H> EmbeddingModel<H>
where
    Client<H>: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Perform the request and return Copilot's native response instead of
    /// the normalized [`embeddings::EmbeddingResponse`]. Same request,
    /// transport, parser, and error path as
    /// [`embeddings::EmbeddingModel::embed_texts_response`].
    pub async fn raw_embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<CopilotEmbeddingResponse, EmbeddingError> {
        self.raw_embed_texts_with_request_id(documents)
            .await
            .map(|(response, _)| response)
    }

    /// [`Self::raw_embed_texts`] plus the `x-request-id` transport request
    /// id, when the response carried one.
    pub async fn raw_embed_texts_with_request_id(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<(CopilotEmbeddingResponse, Option<String>), EmbeddingError> {
        let documents = documents.into_iter().collect::<Vec<_>>();
        self.raw_embed_texts_slice(&documents).await
    }

    /// Borrow-shaped twin of [`Self::raw_embed_texts_with_request_id`]: the
    /// batch is only serialized into the request body, so callers that keep
    /// their documents (the normalize path) can lend them instead of cloning
    /// the batch.
    async fn raw_embed_texts_slice(
        &self,
        documents: &[String],
    ) -> Result<(CopilotEmbeddingResponse, Option<String>), EmbeddingError> {
        let auth = self
            .client
            .provider()
            .auth
            .auth_context(self.client.http_client())
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
            post_with_auth_base(&self.client, &auth, "/embeddings")?,
            &headers,
        )
        .body(serde_json::to_vec(&body)?)
        .map_err(|err| EmbeddingError::HttpError(err.into()))?;

        let response = self.client.send(req).await?;
        let (parts, body) = response.into_parts();
        let status = parts.status;
        let provider_request_id =
            crate::providers::internal::transcription::request_id_from_headers(
                &parts.headers,
                Some("x-request-id"),
            );
        let body: Vec<u8> = body.await?;
        if status.is_success() {
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

            Ok((body, provider_request_id))
        } else {
            Err(EmbeddingError::from_http_response(
                status,
                String::from_utf8_lossy(&body).into_owned(),
            ))
        }
    }
}

impl<H> embeddings::EmbeddingModel for EmbeddingModel<H>
where
    Client<H>: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    fn max_documents(&self) -> usize {
        1024
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts_response(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
        crate::telemetry::instrument_modality(
            PROVIDER_NAME,
            &self.model,
            crate::telemetry::ModalityOperation::Embeddings,
            async {
                use embeddings::NormalizeEmbeddingResponse as _;

                let documents = documents.into_iter().collect::<Vec<_>>();
                let (response, provider_request_id) =
                    self.raw_embed_texts_slice(&documents).await?;
                let captured = serde_json::to_value(&response)?;
                Ok(response
                    .normalize(PROVIDER_NAME, documents)?
                    .with_optional_provider_request_id(provider_request_id)
                    .with_raw(captured))
            },
        )
        .await
    }
}

impl<H> EmbeddingModel<H>
where
    Client<H>: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Build the model, defaulting `ndims` from the model identifier when the
    /// caller gave none — the body behind `EmbeddingsClient::embedding_model`.
    pub fn make(client: &Client<H>, model: String, ndims: Option<usize>) -> Self {
        let dims = ndims.unwrap_or(match model.as_str() {
            TEXT_EMBEDDING_3_LARGE => 3072,
            TEXT_EMBEDDING_3_SMALL | TEXT_EMBEDDING_ADA_002 => 1536,
            _ => 0,
        });
        Self::new(client.clone(), model, dims)
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
pub struct CopilotModelLister<H = crate::http_client::BoxedHttpClient> {
    client: Client<H>,
}

impl<H> ModelLister<H> for CopilotModelLister<H>
where
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    async fn list_all(&self) -> Result<ModelList, ModelListingError> {
        let auth = self
            .client
            .provider()
            .auth
            .auth_context(self.client.http_client())
            .await
            .map_err(|err| ModelListingError::AuthError {
                message: err.to_string(),
            })?;

        let headers = default_headers(&auth.api_key, "user", false, CopilotIntent::Panel);
        let req = apply_headers(
            get_with_auth_base(&self.client, &auth, MODEL_LISTING_PATH)?,
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

impl<H> CopilotModelLister<H>
where
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Build the lister over `client`.
    pub fn new(client: Client<H>) -> Self {
        Self { client }
    }
}

async fn send_copilot_chat_raw_streaming_request<T>(
    http_client: T,
    req: Request<Vec<u8>>,
) -> Result<crate::streaming::StreamingResult, CompletionError>
where
    T: HttpClientExt + Clone + 'static,
{
    // Copilot's `/chat/completions` route relays OpenAI's chat-completions
    // SSE wire verbatim, so OpenAI's shared streaming profile (tolerant
    // deserializers, reasoning handling, finish-reason mapping) is the event
    // interpreter — only the auth/transport in the caller is
    // Copilot-specific; the terminal record is attributed to Copilot.
    openai::completion::streaming::send_compatible_raw_streaming_request(
        http_client,
        req,
        PROVIDER_NAME.to_owned(),
    )
    .await
}

fn default_token_dir() -> Option<PathBuf> {
    config_dir().map(|dir| dir.join("github_copilot"))
}

use crate::providers::internal::auth::config_dir;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod response_identity_tests;

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
mod raw_capture_tests;
