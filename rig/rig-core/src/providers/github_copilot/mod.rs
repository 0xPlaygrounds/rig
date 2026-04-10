//! GitHub Copilot subscription OAuth provider.
//!
//! Supports Chat Completions, Responses, and Embeddings against
//! `https://api.githubcopilot.com`.

mod auth;

use crate::client::{
    self, ApiKey, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
    ProviderClient, Transport,
};
use crate::completion::{self, CompletionError};
use crate::embeddings::{self, EmbeddingError};
use crate::http_client::{self, HttpClientExt};
use crate::providers::openai;
use crate::providers::openai::responses_api::{self, CompletionRequest as ResponsesRequest};
use crate::streaming::StreamingCompletionResponse;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use async_stream::stream;
use futures::StreamExt;
use http::Request;
use serde_json::json;
use std::borrow::Cow;
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use tracing::{Instrument, info_span};

const GITHUB_COPILOT_API_BASE_URL: &str = "https://api.githubcopilot.com";
const EDITOR_PLUGIN_VERSION: &str = "copilot-chat/0.26.7";
const USER_AGENT: &str = "GitHubCopilotChat/0.26.7";
const API_VERSION: &str = "2025-04-01";

/// `gpt-4`
pub const GPT_4: &str = "gpt-4";
/// `gpt-4o`
pub const GPT_4O: &str = "gpt-4o";
/// `gpt-5.1-codex`
pub const GPT_5_1_CODEX: &str = "gpt-5.1-codex";
/// `text-embedding-3-small`
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// `text-embedding-3-large`
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
/// `text-embedding-ada-002`
pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

/// Returns whether a GitHub Copilot model must be routed through `/responses`.
///
/// Current Copilot routing accepts most conversational models on
/// `/chat/completions`, while Codex-class models are exposed through
/// `/responses`.
pub fn requires_responses_api(model: &str) -> bool {
    model.to_ascii_lowercase().contains("codex")
}

#[derive(Clone)]
pub enum GitHubCopilotAuth {
    ApiKey(String),
    OAuth,
}

impl ApiKey for GitHubCopilotAuth {}

impl<S> From<S> for GitHubCopilotAuth
where
    S: Into<String>,
{
    fn from(value: S) -> Self {
        Self::ApiKey(value.into())
    }
}

impl Debug for GitHubCopilotAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiKey(_) => f.write_str("ApiKey(<redacted>)"),
            Self::OAuth => f.write_str("OAuth"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GitHubCopilotBuilder {
    access_token_file: Option<PathBuf>,
    api_key_file: Option<PathBuf>,
    device_code_handler: auth::DeviceCodeHandler,
}

#[derive(Clone)]
pub struct GitHubCopilotExt {
    auth: auth::Authenticator,
}

impl Debug for GitHubCopilotExt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GitHubCopilotExt")
            .field("auth", &self.auth)
            .finish()
    }
}

pub type Client<H = reqwest::Client> = client::Client<GitHubCopilotExt, H>;
pub type ClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<GitHubCopilotBuilder, GitHubCopilotAuth, H>;

impl Default for GitHubCopilotBuilder {
    fn default() -> Self {
        let token_dir = default_token_dir();
        Self {
            access_token_file: token_dir.as_ref().map(|dir| dir.join("access-token")),
            api_key_file: token_dir.map(|dir| dir.join("api-key.json")),
            device_code_handler: auth::DeviceCodeHandler::default(),
        }
    }
}

impl Provider for GitHubCopilotExt {
    type Builder = GitHubCopilotBuilder;

    const VERIFY_PATH: &'static str = "";
}

impl<H> Capabilities<H> for GitHubCopilotExt {
    type Completion = Capable<CompletionModel<H>>;
    type Embeddings = Capable<EmbeddingModel<H>>;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for GitHubCopilotExt {}

impl ProviderBuilder for GitHubCopilotBuilder {
    type Extension<H>
        = GitHubCopilotExt
    where
        H: HttpClientExt;
    type ApiKey = GitHubCopilotAuth;

    const BASE_URL: &'static str = GITHUB_COPILOT_API_BASE_URL;

    fn build<H>(
        builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        let auth = match builder.get_api_key() {
            GitHubCopilotAuth::ApiKey(api_key) => auth::AuthSource::ApiKey(api_key.clone()),
            GitHubCopilotAuth::OAuth => auth::AuthSource::OAuth,
        };

        let ext = builder.ext();
        Ok(GitHubCopilotExt {
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
    type Input = GitHubCopilotAuth;

    fn from_env() -> Self {
        let mut builder = Self::builder();
        if let Ok(base_url) = std::env::var("GITHUB_COPILOT_API_BASE") {
            builder = builder.base_url(base_url);
        }

        if let Ok(api_key) = std::env::var("GITHUB_COPILOT_API_KEY") {
            builder.api_key(api_key).build().unwrap()
        } else {
            builder.oauth().build().unwrap()
        }
    }

    fn from_val(input: Self::Input) -> Self {
        Self::builder().api_key(input).build().unwrap()
    }
}

impl<H> client::ClientBuilder<GitHubCopilotBuilder, crate::markers::Missing, H> {
    pub fn oauth(self) -> client::ClientBuilder<GitHubCopilotBuilder, GitHubCopilotAuth, H> {
        self.api_key(GitHubCopilotAuth::OAuth)
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

impl<H> Client<H>
where
    H: HttpClientExt + Clone + Debug + Default + WasmCompatSend + WasmCompatSync + 'static,
{
    pub async fn authorize(&self) -> Result<(), auth::AuthError> {
        self.ext().auth.auth_context().await.map(|_| ())
    }

    /// Construct a GitHub Copilot Responses API model.
    ///
    /// Use this for Codex-class models that only support `/responses`.
    pub fn responses_model(&self, model: impl Into<String>) -> ResponsesCompletionModel<H> {
        ResponsesCompletionModel::new(self.clone(), model)
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
}

impl<H> completion::CompletionModel for CompletionModel<H>
where
    Client<H>: HttpClientExt + Clone + Debug + 'static,
    H: Clone + Default + Debug + WasmCompatSend + WasmCompatSync + 'static,
{
    type Response = openai::CompletionResponse;
    type StreamingResponse = openai::StreamingCompletionResponse;
    type Client = Client<H>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<Self::Response>, CompletionError> {
        let initiator = request_initiator(&completion_request);
        let has_vision = request_has_vision(&completion_request);
        let request = openai::completion::CompletionRequest::try_from(
            openai::completion::OpenAIRequestParams {
                model: self.model.clone(),
                request: completion_request,
                strict_tools: self.strict_tools,
                tool_result_array_content: self.tool_result_array_content,
            },
        )?;

        let body = serde_json::to_vec(&request)?;
        let auth = self
            .client
            .ext()
            .auth
            .auth_context()
            .await
            .map_err(|err| CompletionError::ProviderError(err.to_string()))?;

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
                gen_ai.provider.name = "github_copilot",
                gen_ai.request.model = self.model,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cached_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        async move {
            let response = self.client.send(req).await?;

            if response.status().is_success() {
                let text = http_client::text(response).await?;
                match serde_json::from_str::<openai::ApiResponse<openai::CompletionResponse>>(
                    &text,
                )? {
                    openai::ApiResponse::Ok(response) => response.try_into(),
                    openai::ApiResponse::Err(err) => {
                        Err(CompletionError::ProviderError(err.message))
                    }
                }
            } else {
                let text = http_client::text(response).await?;
                Err(CompletionError::ProviderError(text))
            }
        }
        .instrument(span)
        .await
    }

    async fn stream(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let initiator = request_initiator(&completion_request);
        let has_vision = request_has_vision(&completion_request);
        let request = openai::completion::CompletionRequest::try_from(
            openai::completion::OpenAIRequestParams {
                model: self.model.clone(),
                request: completion_request,
                strict_tools: self.strict_tools,
                tool_result_array_content: self.tool_result_array_content,
            },
        )?;

        let auth = self
            .client
            .ext()
            .auth
            .auth_context()
            .await
            .map_err(|err| CompletionError::ProviderError(err.to_string()))?;

        let headers = default_headers(&auth.api_key, initiator, has_vision);
        let mut request_json = serde_json::to_value(&request)?;
        request_json["stream"] = json!(true);
        request_json["stream_options"] = json!({ "include_usage": true });

        let req = apply_headers(
            post_with_auth_base(&self.client, &auth, "/chat/completions", Transport::Sse)?,
            &headers,
        )
        .body(serde_json::to_vec(&request_json)?)
        .map_err(|err| CompletionError::HttpError(err.into()))?;

        openai::send_compatible_streaming_request(self.client.clone(), req).await
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
    Client<H>: HttpClientExt + Clone + Debug + Send + 'static,
    H: Clone + Default + Debug + 'static,
{
    const MAX_DOCUMENTS: usize = 1024;
    type Client = Client<H>;

    fn make(client: &Self::Client, model: impl Into<String>, ndims: Option<usize>) -> Self {
        let model = model.into();
        let dims = ndims.unwrap_or_else(|| match model.as_str() {
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

        if self.ndims > 0 && self.model.as_str() != TEXT_EMBEDDING_ADA_002 {
            body["dimensions"] = json!(self.ndims);
        }
        if let Some(encoding_format) = &self.encoding_format {
            body["encoding_format"] = json!(encoding_format);
        }
        if let Some(user) = &self.user {
            body["user"] = json!(user);
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
            let body: openai::ApiResponse<openai::EmbeddingResponse> =
                serde_json::from_slice(&body)?;
            match body {
                openai::ApiResponse::Ok(response) => Ok(response
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
                    .collect()),
                openai::ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
            }
        } else {
            let text = http_client::text(response).await?;
            Err(EmbeddingError::ProviderError(text))
        }
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

    fn create_request(
        &self,
        request: completion::CompletionRequest,
    ) -> Result<ResponsesRequest, CompletionError> {
        let mut request = ResponsesRequest::try_from((self.model.clone(), request))?;
        request.tools.extend(self.tools.clone());
        Ok(request)
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
        let initiator = request_initiator(&completion_request);
        let has_vision = request_has_vision(&completion_request);
        let request = self.create_request(completion_request)?;
        let auth = self
            .client
            .ext()
            .auth
            .auth_context()
            .await
            .map_err(|err| CompletionError::ProviderError(err.to_string()))?;

        let headers = default_headers(&auth.api_key, initiator, has_vision);
        let req = apply_headers(
            post_with_auth_base(&self.client, &auth, "/responses", Transport::Http)?,
            &headers,
        )
        .body(serde_json::to_vec(&request)?)
        .map_err(|err| CompletionError::HttpError(err.into()))?;

        let response = self.client.send(req).await?;
        if response.status().is_success() {
            let body = http_client::text(response).await?;
            serde_json::from_str::<responses_api::CompletionResponse>(&body)?.try_into()
        } else {
            let text = http_client::text(response).await?;
            Err(CompletionError::ProviderError(text))
        }
    }

    async fn stream(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let initiator = request_initiator(&completion_request);
        let has_vision = request_has_vision(&completion_request);
        let request = self.create_request(completion_request)?;
        let auth = self
            .client
            .ext()
            .auth
            .auth_context()
            .await
            .map_err(|err| CompletionError::ProviderError(err.to_string()))?;

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
                gen_ai.provider.name = "github_copilot",
                gen_ai.request.model = self.model,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cached_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let client = self.client.clone();
        let mut event_source = crate::http_client::sse::GenericEventSource::new(client, req);

        let stream = tracing_futures::Instrument::instrument(
            stream! {
                let mut final_usage = responses_api::ResponsesUsage::new();
                let mut tool_calls: Vec<crate::streaming::RawStreamingChoice<responses_api::streaming::StreamingCompletionResponse>> = Vec::new();
                let mut tool_call_internal_ids: std::collections::HashMap<String, String> = std::collections::HashMap::new();
                let span = tracing::Span::current();

                while let Some(event_result) = event_source.next().await {
                    match event_result {
                        Ok(crate::http_client::sse::Event::Open) => continue,
                        Ok(crate::http_client::sse::Event::Message(evt)) => {
                            if evt.data.trim().is_empty() {
                                continue;
                            }

                            let data = serde_json::from_str::<responses_api::streaming::StreamingCompletionChunk>(&evt.data);
                            let Ok(data) = data else {
                                continue;
                            };

                            if let responses_api::streaming::StreamingCompletionChunk::Delta(chunk) = &data {
                                use crate::streaming::{RawStreamingChoice, RawStreamingToolCall, ToolCallDeltaContent};
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
                                                content: ToolCallDeltaContent::Name(func.name.clone()),
                                            });
                                        }
                                    }
                                    ItemChunkKind::OutputItemDone(message) => match message {
                                        StreamingItemDoneOutput { item: responses_api::Output::FunctionCall(func), .. } => {
                                            let internal_id = tool_call_internal_ids
                                                .entry(func.id.clone())
                                                .or_insert_with(|| nanoid::nanoid!())
                                                .clone();
                                            let raw_tool_call = RawStreamingToolCall::new(
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
                                                yield Ok(reasoning_choice);
                                            }
                                        }
                                        StreamingItemDoneOutput { item: responses_api::Output::Message(msg), .. } => {
                                            yield Ok(RawStreamingChoice::MessageId(msg.id.clone()));
                                        }
                                    },
                                    ItemChunkKind::OutputTextDelta(delta) => {
                                        yield Ok(crate::streaming::RawStreamingChoice::Message(delta.delta.clone()))
                                    }
                                    ItemChunkKind::ReasoningSummaryTextDelta(delta) => {
                                        yield Ok(crate::streaming::RawStreamingChoice::ReasoningDelta { id: None, reasoning: delta.delta.clone() })
                                    }
                                    ItemChunkKind::RefusalDelta(delta) => {
                                        yield Ok(crate::streaming::RawStreamingChoice::Message(delta.delta.clone()))
                                    }
                                    ItemChunkKind::FunctionCallArgsDelta(delta) => {
                                        let internal_call_id = tool_call_internal_ids
                                            .entry(delta.item_id.clone())
                                            .or_insert_with(|| nanoid::nanoid!())
                                            .clone();
                                        yield Ok(crate::streaming::RawStreamingChoice::ToolCallDelta {
                                            id: delta.item_id.clone(),
                                            internal_call_id,
                                            content: crate::streaming::ToolCallDeltaContent::Delta(delta.delta.clone())
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
                                            .unwrap_or_else(|| "GitHub Copilot response stream failed".into());
                                        yield Err(CompletionError::ProviderError(error));
                                        break;
                                    }
                                    _ => continue,
                                }
                            }
                        }
                        Err(crate::http_client::Error::StreamEnded) => {
                            event_source.close();
                        }
                        Err(error) => {
                            yield Err(CompletionError::ProviderError(error.to_string()));
                            break;
                        }
                    }
                }

                event_source.close();

                for tool_call in &tool_calls {
                    yield Ok(tool_call.to_owned())
                }

                span.record("gen_ai.usage.input_tokens", final_usage.input_tokens);
                span.record("gen_ai.usage.output_tokens", final_usage.output_tokens);
                span.record(
                    "gen_ai.usage.cached_tokens",
                    final_usage
                        .input_tokens_details
                        .as_ref()
                        .map(|d| d.cached_tokens)
                        .unwrap_or(0),
                );

                yield Ok(crate::streaming::RawStreamingChoice::FinalResponse(
                    responses_api::streaming::StreamingCompletionResponse { usage: final_usage }
                ));
            },
            span,
        );

        Ok(StreamingCompletionResponse::stream(Box::pin(stream)))
    }
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
    use super::requires_responses_api;

    #[test]
    fn test_client_initialization() {
        let _client = crate::providers::github_copilot::Client::builder()
            .oauth()
            .build()
            .expect("Client::builder()");
    }

    #[test]
    fn codex_models_require_responses_api() {
        assert!(requires_responses_api("gpt-5.3-codex"));
        assert!(requires_responses_api("gpt-5.1-CODEX-mini"));
        assert!(!requires_responses_api("gpt-5.2"));
        assert!(!requires_responses_api("claude-sonnet-4.5"));
    }
}
