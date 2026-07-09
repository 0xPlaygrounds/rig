//! Moonshot AI (Kimi) API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig_core::providers::moonshot;
//! use rig_core::client::CompletionClient;
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
//! use rig_core::providers::moonshot;
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
use crate::http_client;
use crate::http_client::HttpClientExt;
use crate::providers::anthropic::client::{
    AnthropicBuilder as AnthropicCompatBuilder, AnthropicKey, finish_anthropic_builder,
};
use crate::{completion::CompletionError, providers::openai};

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
    type Rerank = Nothing;
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
    type Rerank = Nothing;
}

pub type Client<H = reqwest::Client> = client::Client<MoonshotExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<MoonshotBuilder, MoonshotApiKey, H>;
pub type AnthropicClient<H = reqwest::Client> = client::Client<MoonshotAnthropicExt, H>;
pub type AnthropicClientBuilder<H = crate::markers::Missing> =
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

// ================================================================
// Moonshot Completion API
// ================================================================

/// Moonshot v1 128K context model (legacy)
pub const MOONSHOT_CHAT: &str = "moonshot-v1-128k";

/// Kimi K2 — Mixture-of-Experts model (1T total params, 32B active)
pub const KIMI_K2: &str = "kimi-k2";

/// Kimi K2.5 — Native multimodal agentic model with 256K context
pub const KIMI_K2_5: &str = "kimi-k2.5";

/// Moonshot completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = reqwest::Client> =
    openai::completion::GenericCompletionModel<MoonshotExt, H>;

impl openai::completion::OpenAICompatibleProvider for MoonshotExt {
    const PROVIDER_NAME: &'static str = "moonshot";

    type StreamingUsage = openai::Usage;

    // Moonshot's API rejects the `json_schema` response format; keep the
    // pre-migration behavior of dropping `output_schema` with a warning.
    const SUPPORTS_RESPONSE_FORMAT: bool = false;

    type Response = openai::CompletionResponse;

    fn prepare_request(
        &self,
        request: &mut openai::completion::CompletionRequest,
    ) -> Result<(), CompletionError> {
        // Moonshot only supports `auto`/`none` tool choices. Forcing one
        // specific tool has no workaround; fail fast like the pre-migration
        // conversion did (on main, `openai::ToolChoice::try_from` returned
        // "Provider doesn't support only using specific tools" for every
        // `ToolChoice::Specific`, single- or multi-name).
        if matches!(
            request.tool_choice,
            Some(openai::completion::ToolChoice::Function { .. })
        ) {
            return Err(CompletionError::ProviderError(
                "Moonshot does not support forcing a specific tool".to_string(),
            ));
        }

        // Moonshot does not support `tool_choice: "required"`; coerce it to
        // `auto` and steer the model with an extra user message instead.
        if matches!(
            request.tool_choice,
            Some(openai::completion::ToolChoice::Required)
        ) {
            tracing::warn!(
                "Moonshot does not support tool_choice=required; coercing to auto with an additional steering message"
            );
            request.tool_choice = Some(openai::completion::ToolChoice::Auto);
            request.messages.push(openai::Message::User {
                content: crate::OneOrMany::one(openai::UserContent::Text {
                    text: "Please select a tool to handle the current issue.".to_string(),
                }),
                name: None,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{MoonshotExt, normalize_anthropic_base_url, resolve_anthropic_base_override};
    use crate::completion::CompletionRequest;
    use crate::message::{
        AssistantContent, Message, Reasoning, ToolCall, ToolChoice, ToolFunction,
    };
    use crate::providers::openai::completion::{
        CompletionRequest as OpenAICompletionRequest, OpenAICompatibleProvider, OpenAIRequestParams,
    };

    fn prepared_body(request: CompletionRequest, model: &str) -> serde_json::Value {
        let mut request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
            model: model.to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
            supports_response_format: MoonshotExt::SUPPORTS_RESPONSE_FORMAT,
            supports_tools: true,
        })
        .expect("request should convert");
        MoonshotExt
            .prepare_request(&mut request)
            .expect("prepare_request should succeed");
        serde_json::to_value(request).expect("request should serialize")
    }

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

        let body = prepared_body(request, "kimi-k2-thinking");
        assert_eq!(
            body["messages"][0]["reasoning_content"],
            serde_json::json!("tool planning")
        );
    }

    #[test]
    fn moonshot_joins_multiple_reasoning_blocks_with_newline() {
        // A replayed assistant turn carrying two distinct reasoning blocks must
        // keep them newline-separated on the wire, not glued together.
        let assistant = Message::Assistant {
            id: None,
            content: crate::OneOrMany::many(vec![
                AssistantContent::Reasoning(Reasoning::new("first thought")),
                AssistantContent::Reasoning(Reasoning::new("second thought")),
                AssistantContent::Text("done".into()),
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

        let body = prepared_body(request, "kimi-k2-thinking");
        assert_eq!(
            body["messages"][0]["reasoning_content"],
            serde_json::json!("first thought\nsecond thought")
        );
    }

    #[test]
    fn moonshot_specific_tool_choice_is_rejected() {
        let request = CompletionRequest {
            model: Some("kimi-k2.5".to_string()),
            preamble: None,
            chat_history: crate::OneOrMany::one(Message::user("Use a tool.")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: Some(ToolChoice::Specific {
                function_names: vec!["lookup".to_string()],
            }),
            additional_params: None,
            output_schema: None,
        };

        let mut request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
            model: "kimi-k2.5".to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
            supports_response_format: MoonshotExt::SUPPORTS_RESPONSE_FORMAT,
            supports_tools: true,
        })
        .expect("request should convert");

        let error = MoonshotExt
            .prepare_request(&mut request)
            .expect_err("specific tool choice should be rejected");
        assert!(error.to_string().contains("specific tool"));
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

        let body = prepared_body(request, "kimi-k2.5");
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(
            body["messages"]
                .as_array()
                .and_then(|messages| messages.last())
                .and_then(|message| message.get("content"))
                .and_then(|content| content.as_str()),
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
