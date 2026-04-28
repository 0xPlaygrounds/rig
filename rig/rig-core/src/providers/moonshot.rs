//! Moonshot AI (Kimi) API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig::providers::moonshot;
//! use rig::client::CompletionClient;
//!
//! let client = moonshot::Client::new("YOUR_API_KEY").expect("Failed to build client");
//!
//! let kimi_model = client.completion_model(moonshot::KIMI_K2_5);
//! ```
//!
//! # Custom base URL
//! The default base URL is `https://api.moonshot.ai/v1`. For China access,
//! use `https://api.moonshot.cn/v1`:
//! ```no_run
//! use rig::providers::moonshot;
//!
//! let client = moonshot::Client::builder()
//!     .api_key("YOUR_API_KEY")
//!     .base_url("https://api.moonshot.ai/v1")
//!     .build()
//!     .expect("Failed to build Moonshot client");
//! ```
use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
    ProviderClient,
};
use crate::http_client::HttpClientExt;
use crate::providers::anthropic::client::{
    AnthropicBuilder as AnthropicCompatBuilder, AnthropicKey, finish_anthropic_builder,
};
use crate::providers::openai::send_compatible_streaming_request;
use crate::streaming::StreamingCompletionResponse;
use crate::{
    completion::{self, CompletionError, CompletionRequest},
    json_utils,
    providers::openai,
};
use crate::{http_client, message};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::{Instrument, info_span};

// ================================================================
// Main Moonshot Client
// ================================================================
/// Global OpenAI-compatible base URL.
pub const GLOBAL_API_BASE_URL: &str = "https://api.moonshot.ai/v1";
/// China OpenAI-compatible base URL.
pub const CHINA_API_BASE_URL: &str = "https://api.moonshot.cn/v1";
/// Anthropic-compatible base URL.
pub const ANTHROPIC_API_BASE_URL: &str = "https://api.moonshot.ai/anthropic";

#[derive(Debug, Default, Clone, Copy)]
pub struct MoonshotExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct MoonshotBuilder;
#[derive(Debug, Default, Clone)]
pub struct MoonshotAnthropicBuilder {
    anthropic: AnthropicCompatBuilder,
}
#[derive(Debug, Default, Clone, Copy)]
pub struct MoonshotAnthropicExt;

type MoonshotApiKey = BearerAuth;

impl Provider for MoonshotExt {
    type Builder = MoonshotBuilder;

    const VERIFY_PATH: &'static str = "/models";
}

impl Provider for MoonshotAnthropicExt {
    type Builder = MoonshotAnthropicBuilder;

    const VERIFY_PATH: &'static str = "/v1/models";
}

impl DebugExt for MoonshotExt {}
impl DebugExt for MoonshotAnthropicExt {}

impl ProviderBuilder for MoonshotBuilder {
    type Extension<H>
        = MoonshotExt
    where
        H: HttpClientExt;
    type ApiKey = MoonshotApiKey;

    const BASE_URL: &'static str = GLOBAL_API_BASE_URL;

    fn build<H>(
        _builder: &crate::client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(MoonshotExt)
    }
}

impl ProviderBuilder for MoonshotAnthropicBuilder {
    type Extension<H>
        = MoonshotAnthropicExt
    where
        H: HttpClientExt;
    type ApiKey = AnthropicKey;

    const BASE_URL: &'static str = ANTHROPIC_API_BASE_URL;

    fn build<H>(
        _builder: &crate::client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(MoonshotAnthropicExt)
    }

    fn finish<H>(
        &self,
        builder: client::ClientBuilder<Self, AnthropicKey, H>,
    ) -> http_client::Result<client::ClientBuilder<Self, AnthropicKey, H>> {
        finish_anthropic_builder(&self.anthropic, builder)
    }
}

impl<H> Capabilities<H> for MoonshotExt {
    type Completion = Capable<CompletionModel<H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl<H> Capabilities<H> for MoonshotAnthropicExt {
    type Completion =
        Capable<super::anthropic::completion::GenericCompletionModel<MoonshotAnthropicExt, H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

pub type Client<H = reqwest::Client> = client::Client<MoonshotExt, H>;
pub type ClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<MoonshotBuilder, MoonshotApiKey, H>;
pub type AnthropicClient<H = reqwest::Client> = client::Client<MoonshotAnthropicExt, H>;
pub type AnthropicClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<MoonshotAnthropicBuilder, AnthropicKey, H>;

impl ProviderClient for Client {
    type Input = String;
    type Error = crate::client::ProviderClientError;

    /// Create a new Moonshot client from the `MOONSHOT_API_KEY` environment variable.
    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("MOONSHOT_API_KEY")?;
        let mut builder = Self::builder().api_key(&api_key);
        if let Some(base_url) = crate::client::optional_env_var("MOONSHOT_API_BASE")? {
            builder = builder.base_url(base_url);
        }
        builder.build().map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(&input).map_err(Into::into)
    }
}

impl ProviderClient for AnthropicClient {
    type Input = String;
    type Error = crate::client::ProviderClientError;

    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("MOONSHOT_API_KEY")?;
        let mut builder = Self::builder().api_key(api_key);
        if let Some(base_url) =
            anthropic_base_override("MOONSHOT_ANTHROPIC_API_BASE", "MOONSHOT_API_BASE")?
        {
            builder = builder.base_url(base_url);
        }
        builder.build().map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::builder().api_key(input).build().map_err(Into::into)
    }
}

impl<H> ClientBuilder<H> {
    pub fn global(self) -> Self {
        self.base_url(GLOBAL_API_BASE_URL)
    }

    pub fn china(self) -> Self {
        self.base_url(CHINA_API_BASE_URL)
    }
}

impl<H> AnthropicClientBuilder<H> {
    pub fn global(self) -> Self {
        self.base_url(ANTHROPIC_API_BASE_URL)
    }

    pub fn anthropic_version(self, anthropic_version: &str) -> Self {
        self.over_ext(|mut ext| {
            ext.anthropic.anthropic_version = anthropic_version.into();
            ext
        })
    }

    pub fn anthropic_betas(self, anthropic_betas: &[&str]) -> Self {
        self.over_ext(|mut ext| {
            ext.anthropic
                .anthropic_betas
                .extend(anthropic_betas.iter().copied().map(String::from));
            ext
        })
    }

    pub fn anthropic_beta(self, anthropic_beta: &str) -> Self {
        self.over_ext(|mut ext| {
            ext.anthropic.anthropic_betas.push(anthropic_beta.into());
            ext
        })
    }
}

impl super::anthropic::completion::AnthropicCompatibleProvider for MoonshotAnthropicExt {
    const PROVIDER_NAME: &'static str = "moonshot";

    fn default_max_tokens(_model: &str) -> Option<u64> {
        Some(4096)
    }
}

fn anthropic_base_override(
    primary_env: &'static str,
    fallback_env: &'static str,
) -> crate::client::ProviderClientResult<Option<String>> {
    let primary = crate::client::optional_env_var(primary_env)?;
    let fallback = crate::client::optional_env_var(fallback_env)?;

    Ok(resolve_anthropic_base_override(
        primary.as_deref(),
        fallback.as_deref(),
    ))
}

fn resolve_anthropic_base_override(
    primary: Option<&str>,
    fallback: Option<&str>,
) -> Option<String> {
    primary
        .map(str::to_owned)
        .or_else(|| fallback.and_then(normalize_anthropic_base_url))
}

fn normalize_anthropic_base_url(base_url: &str) -> Option<String> {
    if base_url.contains("/anthropic") {
        return Some(base_url.to_owned());
    }

    let mut url = url::Url::parse(base_url).ok()?;
    if !matches!(url.path(), "/v1" | "/v1/") {
        return None;
    }
    url.set_path("/anthropic");
    Some(url.to_string())
}

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    error: MoonshotError,
}

#[derive(Debug, Deserialize)]
struct MoonshotError {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

// ================================================================
// Moonshot Completion API
// ================================================================

/// Moonshot v1 128K context model (legacy)
pub const MOONSHOT_CHAT: &str = "moonshot-v1-128k";

/// Kimi K2 — Mixture-of-Experts model (1T total params, 32B active)
pub const KIMI_K2: &str = "kimi-k2";

/// Kimi K2.5 — Native multimodal agentic model with 256K context
pub const KIMI_K2_5: &str = "kimi-k2.5";

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct MoonshotCompletionRequest {
    model: String,
    pub messages: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<openai::ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<crate::providers::openai::completion::ToolChoice>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

impl TryFrom<(&str, CompletionRequest)> for MoonshotCompletionRequest {
    type Error = CompletionError;

    fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        if req.output_schema.is_some() {
            tracing::warn!("Structured outputs currently not supported for Moonshot");
        }
        let model = req.model.clone().unwrap_or_else(|| model.to_string());
        // Build up the order of messages (context, chat_history, prompt)
        let mut partial_history = vec![];
        if let Some(docs) = req.normalized_documents() {
            partial_history.push(docs);
        }
        partial_history.extend(req.chat_history);

        let mut full_history: Vec<Value> = match &req.preamble {
            Some(preamble) => vec![serde_json::to_value(openai::Message::system(preamble))?],
            None => vec![],
        };

        full_history.extend(moonshot_history_values(partial_history)?);

        let mut tool_choice = None;
        let mut tool_choice_required = false;
        if let Some(choice) = req.tool_choice.clone() {
            match choice {
                message::ToolChoice::Required => {
                    tool_choice_required = true;
                    tool_choice = Some(crate::providers::openai::completion::ToolChoice::Auto);
                }
                other => {
                    tool_choice = Some(crate::providers::openai::ToolChoice::try_from(other)?);
                }
            }
        }

        if tool_choice_required {
            tracing::warn!(
                "Moonshot does not support tool_choice=required; coercing to auto with an additional steering message"
            );
            full_history.push(json!({
                "role": "user",
                "content": "Please select a tool to handle the current issue."
            }));
        }

        Ok(Self {
            model: model.to_string(),
            messages: full_history,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            tools: req
                .tools
                .clone()
                .into_iter()
                .map(openai::ToolDefinition::from)
                .collect::<Vec<_>>(),
            tool_choice,
            additional_params: req.additional_params,
        })
    }
}

fn moonshot_history_values(history: Vec<message::Message>) -> Result<Vec<Value>, CompletionError> {
    let mut result = Vec::new();

    for message in history {
        match message {
            message::Message::Assistant { id: _, content } => {
                if let Some(value) = moonshot_assistant_message_value(content)? {
                    result.push(value);
                }
            }
            other => {
                result.extend(
                    Vec::<openai::Message>::try_from(other)?
                        .into_iter()
                        .map(serde_json::to_value)
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
        }
    }

    Ok(result)
}

fn moonshot_assistant_message_value(
    content: crate::OneOrMany<message::AssistantContent>,
) -> Result<Option<Value>, CompletionError> {
    let mut text_content = Vec::new();
    let mut tool_calls = Vec::new();
    let mut reasoning_parts = Vec::new();

    for item in content {
        match item {
            message::AssistantContent::Text(text) => {
                text_content.push(openai::AssistantContent::Text { text: text.text });
            }
            message::AssistantContent::ToolCall(tool_call) => {
                tool_calls.push(openai::ToolCall::from(tool_call));
            }
            message::AssistantContent::Reasoning(reasoning) => {
                let display = reasoning.display_text();
                if !display.is_empty() {
                    reasoning_parts.push(display);
                }
            }
            message::AssistantContent::Image(_) => {
                return Err(CompletionError::ProviderError(
                    "Moonshot does not support assistant image content in chat history".into(),
                ));
            }
        }
    }

    if text_content.is_empty() && tool_calls.is_empty() && reasoning_parts.is_empty() {
        return Ok(None);
    }

    let content_value = if text_content.is_empty() {
        Value::String(String::new())
    } else {
        serde_json::to_value(text_content)?
    };

    let mut object = serde_json::Map::from_iter([
        ("role".to_string(), Value::String("assistant".to_string())),
        ("content".to_string(), content_value),
    ]);

    if !tool_calls.is_empty() {
        object.insert("tool_calls".to_string(), serde_json::to_value(tool_calls)?);
    }

    if !reasoning_parts.is_empty() {
        object.insert(
            "reasoning_content".to_string(),
            Value::String(reasoning_parts.join("\n")),
        );
    }

    Ok(Some(Value::Object(object)))
}

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
}

impl<T> CompletionModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type Response = openai::CompletionResponse;
    type StreamingResponse = openai::StreamingCompletionResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<openai::CompletionResponse>, CompletionError> {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "moonshot",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        span.record("gen_ai.system_instructions", &completion_request.preamble);

        let request =
            MoonshotCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "MoonShot completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post("/chat/completions")?
            .body(body)
            .map_err(http_client::Error::from)?;

        let async_block = async move {
            let response = self.client.send::<_, bytes::Bytes>(req).await?;

            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if status.is_success() {
                match serde_json::from_slice::<ApiResponse<openai::CompletionResponse>>(
                    &response_body,
                )? {
                    ApiResponse::Ok(response) => {
                        let span = tracing::Span::current();
                        span.record("gen_ai.response.id", response.id.clone());
                        span.record("gen_ai.response.model", response.model.clone());
                        if let Some(ref usage) = response.usage {
                            span.record("gen_ai.usage.input_tokens", usage.prompt_tokens);
                            span.record(
                                "gen_ai.usage.output_tokens",
                                usage.total_tokens - usage.prompt_tokens,
                            );
                        }
                        if tracing::enabled!(tracing::Level::TRACE) {
                            tracing::trace!(target: "rig::completions",
                                "MoonShot completion response: {}",
                                serde_json::to_string_pretty(&response)?
                            );
                        }
                        response.try_into()
                    }
                    ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.error.message)),
                }
            } else {
                Err(CompletionError::ProviderError(
                    String::from_utf8_lossy(&response_body).to_string(),
                ))
            }
        };

        async_block.instrument(span).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "moonshot",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        span.record("gen_ai.system_instructions", &request.preamble);
        let mut request = MoonshotCompletionRequest::try_from((self.model.as_ref(), request))?;

        let params = json_utils::merge(
            request.additional_params.unwrap_or(serde_json::json!({})),
            serde_json::json!({"stream": true, "stream_options": {"include_usage": true} }),
        );

        request.additional_params = Some(params);

        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "MoonShot streaming completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post("/chat/completions")?
            .body(body)
            .map_err(http_client::Error::from)?;

        send_compatible_streaming_request(self.client.clone(), req)
            .instrument(span)
            .await
    }
}

#[derive(Default, Debug, Deserialize, Serialize)]
pub enum ToolChoice {
    None,
    #[default]
    Auto,
}

impl TryFrom<message::ToolChoice> for ToolChoice {
    type Error = CompletionError;

    fn try_from(value: message::ToolChoice) -> Result<Self, Self::Error> {
        let res = match value {
            message::ToolChoice::None => Self::None,
            message::ToolChoice::Auto => Self::Auto,
            choice => {
                return Err(CompletionError::ProviderError(format!(
                    "Unsupported tool choice type: {choice:?}"
                )));
            }
        };

        Ok(res)
    }
}
#[cfg(test)]
mod tests {
    use super::{
        MoonshotCompletionRequest, normalize_anthropic_base_url, resolve_anthropic_base_override,
    };
    use crate::completion::CompletionRequest;
    use crate::message::{
        AssistantContent, Message, Reasoning, ToolCall, ToolChoice, ToolFunction,
    };

    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::moonshot::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::moonshot::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
        let _anthropic_client = crate::providers::moonshot::AnthropicClient::new("dummy-key")
            .expect("AnthropicClient::new() failed");
        let _anthropic_client_from_builder = crate::providers::moonshot::AnthropicClient::builder()
            .api_key("dummy-key")
            .build()
            .expect("AnthropicClient::builder() failed");
    }

    #[test]
    fn moonshot_preserves_reasoning_content_in_assistant_history() {
        let assistant = Message::Assistant {
            id: None,
            content: crate::OneOrMany::many(vec![
                AssistantContent::Reasoning(Reasoning::new("tool planning")),
                AssistantContent::ToolCall(ToolCall {
                    id: "call_1".to_string(),
                    call_id: None,
                    function: ToolFunction {
                        name: "lookup".to_string(),
                        arguments: serde_json::json!({}),
                    },
                    signature: None,
                    additional_params: None,
                }),
            ])
            .expect("assistant content"),
        };

        let request = CompletionRequest {
            model: Some("kimi-k2-thinking".to_string()),
            preamble: None,
            chat_history: crate::OneOrMany::one(assistant),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        };

        let converted =
            MoonshotCompletionRequest::try_from(("kimi-k2-thinking", request)).expect("convert");
        let assistant = converted
            .messages
            .first()
            .and_then(|value| value.as_object())
            .expect("assistant message");

        assert_eq!(
            assistant
                .get("reasoning_content")
                .and_then(|value| value.as_str()),
            Some("tool planning")
        );
    }

    #[test]
    fn moonshot_required_tool_choice_is_coerced() {
        let request = CompletionRequest {
            model: Some("kimi-k2.5".to_string()),
            preamble: None,
            chat_history: crate::OneOrMany::one(Message::user("Use a tool.")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: Some(ToolChoice::Required),
            additional_params: None,
            output_schema: None,
        };

        let converted =
            MoonshotCompletionRequest::try_from(("kimi-k2.5", request)).expect("convert");
        assert!(matches!(
            converted.tool_choice,
            Some(crate::providers::openai::completion::ToolChoice::Auto)
        ));
        assert_eq!(
            converted
                .messages
                .last()
                .and_then(|value| value.get("content"))
                .and_then(|value| value.as_str()),
            Some("Please select a tool to handle the current issue.")
        );
    }

    #[test]
    fn normalize_openai_style_base_to_anthropic_base() {
        assert_eq!(
            normalize_anthropic_base_url("https://api.moonshot.ai/v1").as_deref(),
            Some("https://api.moonshot.ai/anthropic")
        );
        assert_eq!(
            normalize_anthropic_base_url("https://api.moonshot.cn/v1").as_deref(),
            Some("https://api.moonshot.cn/anthropic")
        );
        assert_eq!(
            normalize_anthropic_base_url("https://proxy.example.com/v1").as_deref(),
            Some("https://proxy.example.com/anthropic")
        );
    }

    #[test]
    fn normalize_preserves_existing_anthropic_base() {
        assert_eq!(
            normalize_anthropic_base_url("https://proxy.example.com/anthropic").as_deref(),
            Some("https://proxy.example.com/anthropic")
        );
    }

    #[test]
    fn anthropic_primary_override_wins() {
        let override_url = resolve_anthropic_base_override(
            Some("https://primary.example.com/anthropic"),
            Some("https://api.moonshot.cn/v1"),
        );

        assert_eq!(
            override_url.as_deref(),
            Some("https://primary.example.com/anthropic")
        );
    }
}
