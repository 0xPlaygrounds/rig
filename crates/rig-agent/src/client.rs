//! Concrete fluent facade over data-oriented provider configuration.
//!
//! Provider clients stay monomorphic and own only connection defaults. This
//! module binds their plain [`ProviderConfig`] values to a shared [`Runtime`]
//! for agent construction and one-shot completion requests.

use std::sync::Arc;

use rig_core::completion::{
    CompletionError, CompletionRequest, CompletionRequestBuilder, CompletionResponse, Document,
    Message, ProviderToolDefinition, ToolDefinition,
};
use rig_core::message::ToolChoice;
use rig_core::streaming::CompletionStream;

use crate::AgentBuilder;
use crate::provider::{self, ProviderConfig, Runtime};

/// Convert a concrete provider client into plain provider configuration and a runtime.
pub trait ToProviderConfig {
    /// Materialize provider configuration for `model`.
    fn provider_config(&self, model: &str) -> ProviderConfig;

    /// Runtime sharing this client's live transport.
    fn runtime(&self) -> Arc<Runtime>;
}

/// Fluent agent construction for concrete provider clients.
pub trait AgentClientExt: ToProviderConfig {
    /// Start an agent builder for `model` using this client's connection.
    fn agent(&self, model: &str) -> AgentBuilder {
        AgentBuilder::new(self.provider_config(model)).runtime(self.runtime())
    }
}

impl<T> AgentClientExt for T where T: ToProviderConfig + ?Sized {}

/// Fluent direct-completion construction for concrete provider clients.
pub trait CompletionClientExt: ToProviderConfig {
    /// Bind `model` and this client's runtime to a completion handle.
    fn completion(&self, model: &str) -> CompletionHandle {
        CompletionHandle::new(self.provider_config(model), self.runtime())
    }

    /// Familiar alias for [`Self::completion`].
    fn completion_model(&self, model: &str) -> CompletionHandle {
        self.completion(model)
    }
}

impl<T> CompletionClientExt for T where T: ToProviderConfig + ?Sized {}

/// Bind a customized concrete provider config to a runtime.
pub trait BindCompletionExt {
    /// Erase only the provider variant, returning one concrete execution handle.
    fn bind_completion(self, runtime: Arc<Runtime>) -> CompletionHandle;
}

/// A concrete provider/runtime pair used to start direct completion requests.
#[derive(Clone, Debug)]
pub struct CompletionHandle {
    provider: ProviderConfig,
    runtime: Arc<Runtime>,
}

impl CompletionHandle {
    /// Construct a bound completion handle.
    pub fn new(provider: ProviderConfig, runtime: Arc<Runtime>) -> Self {
        Self { provider, runtime }
    }

    /// Start a fluent request for `prompt`.
    pub fn completion_request(&self, prompt: impl Into<Message>) -> BoundCompletionRequest {
        BoundCompletionRequest {
            target: self.clone(),
            builder: CompletionRequest::builder(prompt),
        }
    }

    /// Execute already-built request data through this provider/runtime pair.
    pub async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        provider::complete(&self.provider, &self.runtime, request).await
    }

    /// Stream already-built request data through this provider/runtime pair.
    pub async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionStream, CompletionError> {
        provider::open_stream(&self.provider, &self.runtime, request).await
    }

    /// The selected plain provider configuration.
    pub fn provider_config(&self) -> &ProviderConfig {
        &self.provider
    }

    /// The shared execution runtime.
    pub fn runtime(&self) -> &Arc<Runtime> {
        &self.runtime
    }
}

/// A concrete execution target bound to the existing request-data builder.
#[derive(Clone, Debug)]
pub struct BoundCompletionRequest {
    target: CompletionHandle,
    builder: CompletionRequestBuilder,
}

impl BoundCompletionRequest {
    /// Canonicalize a system preamble into a leading system message at build time.
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.builder = self.builder.preamble(preamble);
        self
    }

    /// Clear the request preamble.
    pub fn without_preamble(mut self) -> Self {
        self.builder = self.builder.without_preamble();
        self
    }

    /// Override the request model.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.builder = self.builder.model(model);
        self
    }

    /// Set or clear the request model override.
    pub fn model_opt(mut self, model: Option<String>) -> Self {
        self.builder = self.builder.model_opt(model);
        self
    }

    /// Append a history message.
    pub fn message(mut self, message: Message) -> Self {
        self.builder = self.builder.message(message);
        self
    }

    /// Append history messages.
    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.builder = self.builder.messages(messages);
        self
    }

    /// Append a document.
    pub fn document(mut self, document: Document) -> Self {
        self.builder = self.builder.document(document);
        self
    }

    /// Append documents.
    pub fn documents(mut self, documents: impl IntoIterator<Item = Document>) -> Self {
        self.builder = self.builder.documents(documents);
        self
    }

    /// Append a portable tool definition.
    pub fn tool(mut self, tool: ToolDefinition) -> Self {
        self.builder = self.builder.tool(tool);
        self
    }

    /// Append portable tool definitions.
    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolDefinition>) -> Self {
        self.builder = self.builder.tools(tools);
        self
    }

    /// Append a provider-hosted tool definition.
    pub fn provider_tool(mut self, tool: ProviderToolDefinition) -> Self {
        self.builder = self.builder.provider_tool(tool);
        self
    }

    /// Append provider-hosted tool definitions.
    pub fn provider_tools(
        mut self,
        tools: impl IntoIterator<Item = ProviderToolDefinition>,
    ) -> Self {
        self.builder = self.builder.provider_tools(tools);
        self
    }

    /// Merge provider-specific request parameters.
    pub fn additional_params(mut self, additional_params: serde_json::Value) -> Self {
        self.builder = self.builder.additional_params(additional_params);
        self
    }

    /// Replace or clear provider-specific request parameters.
    pub fn additional_params_opt(mut self, additional_params: Option<serde_json::Value>) -> Self {
        self.builder = self.builder.additional_params_opt(additional_params);
        self
    }

    /// Set sampling temperature.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.builder = self.builder.temperature(temperature);
        self
    }

    /// Set or clear sampling temperature.
    pub fn temperature_opt(mut self, temperature: Option<f64>) -> Self {
        self.builder = self.builder.temperature_opt(temperature);
        self
    }

    /// Set the maximum output token count.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.builder = self.builder.max_tokens(max_tokens);
        self
    }

    /// Set or clear the maximum output token count.
    pub fn max_tokens_opt(mut self, max_tokens: Option<u64>) -> Self {
        self.builder = self.builder.max_tokens_opt(max_tokens);
        self
    }

    /// Set the tool-choice policy.
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.builder = self.builder.tool_choice(tool_choice);
        self
    }

    /// Set or clear the tool-choice policy.
    pub fn tool_choice_opt(mut self, tool_choice: Option<ToolChoice>) -> Self {
        self.builder = self.builder.tool_choice_opt(tool_choice);
        self
    }

    /// Set a structured-output JSON Schema.
    pub fn output_schema(mut self, schema: schemars::Schema) -> Self {
        self.builder = self.builder.output_schema(schema);
        self
    }

    /// Set or clear a structured-output JSON Schema.
    pub fn output_schema_opt(mut self, schema: Option<schemars::Schema>) -> Self {
        self.builder = self.builder.output_schema_opt(schema);
        self
    }

    /// Opt in or out of sensitive-content telemetry.
    pub fn record_content_telemetry(mut self, enabled: bool) -> Self {
        self.builder = self.builder.record_content_telemetry(enabled);
        self
    }

    /// Inspect normalized input messages without consuming the request.
    pub fn messages_for_telemetry(&self) -> Vec<Message> {
        self.builder.messages_for_telemetry()
    }

    /// Build request data without executing it.
    pub fn build(self) -> CompletionRequest {
        self.builder.build()
    }

    /// Explicit alias for [`Self::build`].
    pub fn build_request(self) -> CompletionRequest {
        self.build()
    }

    /// Execute a non-streaming completion through the bundled dispatcher.
    pub async fn send(self) -> Result<CompletionResponse, CompletionError> {
        let request = self.builder.build();
        provider::complete(&self.target.provider, &self.target.runtime, request).await
    }

    /// Open a normalized streaming completion through the bundled dispatcher.
    pub async fn stream(self) -> Result<CompletionStream, CompletionError> {
        let request = self.builder.build();
        provider::open_stream(&self.target.provider, &self.target.runtime, request).await
    }
}

macro_rules! impl_http_client_bridge {
    ($client:path, $variant:ident) => {
        impl ToProviderConfig for $client {
            fn provider_config(&self, model: &str) -> ProviderConfig {
                ProviderConfig::$variant(self.config(model))
            }

            fn runtime(&self) -> Arc<Runtime> {
                Arc::new(Runtime::with_http(self.http_runtime()))
            }
        }
    };
}

impl_http_client_bridge!(rig_core::providers::anthropic::Client, Anthropic);
impl_http_client_bridge!(rig_core::providers::azure::Client, Azure);
impl_http_client_bridge!(rig_core::providers::chatgpt::Client, ChatGpt);
impl_http_client_bridge!(rig_core::providers::cohere::Client, Cohere);
impl_http_client_bridge!(rig_core::providers::copilot::Client, Copilot);
impl_http_client_bridge!(rig_core::providers::deepseek::Client, DeepSeek);
impl_http_client_bridge!(rig_core::providers::doubleword::Client, Doubleword);
impl_http_client_bridge!(rig_core::providers::gemini::Client, Gemini);
impl_http_client_bridge!(
    rig_core::providers::gemini::InteractionsClient,
    GeminiInteractions
);
impl_http_client_bridge!(rig_core::providers::groq::Client, Groq);
impl_http_client_bridge!(rig_core::providers::huggingface::Client, HuggingFace);
impl_http_client_bridge!(rig_core::providers::hyperbolic::Client, Hyperbolic);
impl_http_client_bridge!(rig_core::providers::llamafile::Client, Llamafile);
impl_http_client_bridge!(rig_core::providers::minimax::Client, Minimax);
impl_http_client_bridge!(rig_core::providers::mira::Client, Mira);
impl_http_client_bridge!(rig_core::providers::mistral::Client, Mistral);
impl_http_client_bridge!(rig_core::providers::moonshot::Client, Moonshot);
impl_http_client_bridge!(rig_core::providers::ollama::Client, Ollama);
impl_http_client_bridge!(rig_core::providers::openai::Client, OpenAiResponses);
impl_http_client_bridge!(rig_core::providers::openai::CompletionsClient, OpenAi);
impl_http_client_bridge!(rig_core::providers::openrouter::Client, OpenRouter);
impl_http_client_bridge!(rig_core::providers::perplexity::Client, Perplexity);
impl_http_client_bridge!(rig_core::providers::together::Client, Together);
impl_http_client_bridge!(rig_core::providers::xai::Client, Xai);
impl_http_client_bridge!(rig_core::providers::xiaomimimo::Client, XiaomiMimo);
impl_http_client_bridge!(rig_core::providers::zai::Client, Zai);

#[cfg(feature = "bedrock")]
impl ToProviderConfig for rig_bedrock::Client {
    fn provider_config(&self, model: &str) -> ProviderConfig {
        ProviderConfig::Bedrock(self.config(model))
    }

    fn runtime(&self) -> Arc<Runtime> {
        Arc::new(Runtime::with_bedrock_provider_client(self.clone()))
    }
}

#[cfg(feature = "gemini-grpc")]
impl ToProviderConfig for rig_gemini_grpc::Client {
    fn provider_config(&self, model: &str) -> ProviderConfig {
        ProviderConfig::GeminiGrpc(self.config(model))
    }

    fn runtime(&self) -> Arc<Runtime> {
        Arc::new(Runtime::with_gemini_grpc_client(
            self.connection_config().clone(),
            self.clone(),
        ))
    }
}

macro_rules! impl_bind_completion {
    ($config:path, $variant:ident) => {
        impl BindCompletionExt for $config {
            fn bind_completion(self, runtime: Arc<Runtime>) -> CompletionHandle {
                CompletionHandle::new(ProviderConfig::$variant(self), runtime)
            }
        }
    };
}

impl_bind_completion!(rig_core::providers::anthropic::functions::Config, Anthropic);
impl_bind_completion!(rig_core::providers::azure::functions::Config, Azure);
impl_bind_completion!(rig_core::providers::chatgpt::functions::Config, ChatGpt);
impl_bind_completion!(rig_core::providers::cohere::functions::Config, Cohere);
impl_bind_completion!(rig_core::providers::copilot::functions::Config, Copilot);
impl_bind_completion!(rig_core::providers::deepseek::functions::Config, DeepSeek);
impl_bind_completion!(
    rig_core::providers::doubleword::functions::Config,
    Doubleword
);
impl_bind_completion!(rig_core::providers::gemini::functions::Config, Gemini);
impl_bind_completion!(
    rig_core::providers::gemini::interactions_api::functions::Config,
    GeminiInteractions
);
impl_bind_completion!(rig_core::providers::groq::functions::Config, Groq);
impl_bind_completion!(
    rig_core::providers::huggingface::functions::Config,
    HuggingFace
);
impl_bind_completion!(
    rig_core::providers::hyperbolic::functions::Config,
    Hyperbolic
);
impl_bind_completion!(rig_core::providers::llamafile::functions::Config, Llamafile);
impl_bind_completion!(rig_core::providers::minimax::functions::Config, Minimax);
impl_bind_completion!(rig_core::providers::mira::functions::Config, Mira);
impl_bind_completion!(rig_core::providers::mistral::functions::Config, Mistral);
impl_bind_completion!(rig_core::providers::moonshot::functions::Config, Moonshot);
impl_bind_completion!(rig_core::providers::ollama::functions::Config, Ollama);
impl_bind_completion!(rig_core::providers::openai::functions::Config, OpenAi);
impl_bind_completion!(
    rig_core::providers::openai::responses_api::functions::Config,
    OpenAiResponses
);
impl_bind_completion!(
    rig_core::providers::openrouter::functions::Config,
    OpenRouter
);
impl_bind_completion!(
    rig_core::providers::perplexity::functions::Config,
    Perplexity
);
impl_bind_completion!(rig_core::providers::together::functions::Config, Together);
impl_bind_completion!(rig_core::providers::xai::functions::Config, Xai);
impl_bind_completion!(
    rig_core::providers::xiaomimimo::functions::Config,
    XiaomiMimo
);
impl_bind_completion!(rig_core::providers::zai::functions::Config, Zai);

impl_bind_completion!(rig_core::providers::bedrock::Config, Bedrock);
impl_bind_completion!(rig_core::providers::gemini_grpc::Config, GeminiGrpc);

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::providers::{anthropic, openai};

    #[test]
    fn bound_builder_matches_plain_request_builder() {
        let runtime = Arc::new(Runtime::new());
        let document = Document {
            id: "doc-1".to_string(),
            text: "reference".to_string(),
            additional_props: std::collections::HashMap::new(),
        };
        let tool = ToolDefinition {
            name: "lookup".to_string(),
            description: "Look something up".to_string(),
            parameters: serde_json::json!({ "type": "object" }),
        };
        let provider_tool =
            ProviderToolDefinition::new("web_search").with_config("depth", serde_json::json!(2));
        let schema = schemars::schema_for!(String);
        let bound = openai::functions::Config::new(openai::GPT_4O)
            .bind_completion(runtime)
            .completion_request("hello")
            .preamble("be terse")
            .model("override-model")
            .message(Message::assistant("prior answer"))
            .document(document.clone())
            .tool(tool.clone())
            .provider_tool(provider_tool.clone())
            .additional_params(serde_json::json!({ "metadata": { "a": 1 } }))
            .temperature(0.2)
            .max_tokens(32)
            .tool_choice(ToolChoice::Required)
            .output_schema(schema.clone())
            .record_content_telemetry(true)
            .build();
        let plain = CompletionRequest::builder("hello")
            .preamble("be terse")
            .model("override-model")
            .message(Message::assistant("prior answer"))
            .document(document)
            .tool(tool)
            .provider_tool(provider_tool)
            .additional_params(serde_json::json!({ "metadata": { "a": 1 } }))
            .temperature(0.2)
            .max_tokens(32)
            .tool_choice(ToolChoice::Required)
            .output_schema(schema)
            .record_content_telemetry(true)
            .build();
        assert_eq!(
            serde_json::to_value(&bound).expect("bound request should serialize"),
            serde_json::to_value(&plain).expect("plain request should serialize")
        );
        assert!(bound.record_telemetry_content);
        assert!(plain.record_telemetry_content);
    }

    #[test]
    fn bound_and_plain_requests_produce_identical_openai_wire_bytes() {
        let config = openai::responses_api::functions::Config::new(openai::GPT_5_2);
        let bound = config
            .clone()
            .bind_completion(Arc::new(Runtime::new()))
            .completion_request("question")
            .preamble("system")
            .message(Message::assistant("acknowledged"))
            .temperature(0.0)
            .max_tokens(64)
            .build();
        let plain = CompletionRequest::builder("question")
            .preamble("system")
            .message(Message::assistant("acknowledged"))
            .temperature(0.0)
            .max_tokens(64)
            .build();

        assert_eq!(
            openai::responses_api::functions::build_request_body(&config, &bound, false)
                .expect("bound request should convert"),
            openai::responses_api::functions::build_request_body(&config, &plain, false)
                .expect("plain request should convert")
        );
        assert_eq!(
            openai::responses_api::functions::build_request_body(&config, &bound, true)
                .expect("bound stream request should convert"),
            openai::responses_api::functions::build_request_body(&config, &plain, true)
                .expect("plain stream request should convert")
        );
    }

    #[test]
    fn provider_specific_config_survives_binding() {
        let handle = anthropic::functions::Config::new("claude-test")
            .with_prompt_caching()
            .with_automatic_caching()
            .bind_completion(Arc::new(Runtime::new()));
        let ProviderConfig::Anthropic(config) = handle.provider_config() else {
            panic!("expected anthropic provider config");
        };
        assert!(config.prompt_caching);
        assert!(config.automatic_caching);
    }

    #[test]
    fn openai_surfaces_remain_explicit() {
        let client = openai::Client::new("test-key");
        assert!(matches!(
            client.provider_config(openai::GPT_5_2),
            ProviderConfig::OpenAiResponses(_)
        ));
        assert!(matches!(
            client.completions_api().provider_config(openai::GPT_4O),
            ProviderConfig::OpenAi(_)
        ));
    }

    #[cfg(feature = "bedrock")]
    #[tokio::test]
    async fn bedrock_agent_and_completion_handles_share_the_client_sdk_cache() {
        use aws_sdk_bedrockruntime::config::{BehaviorVersion, Region};

        let seeded = aws_sdk_bedrockruntime::Client::from_conf(
            aws_sdk_bedrockruntime::config::Builder::new()
                .behavior_version(BehaviorVersion::latest())
                .region(Region::new("shared-cache-marker"))
                .endpoint_url("http://bedrock-cache.invalid")
                .build(),
        );
        let client = rig_bedrock::Client::from(seeded);
        let direct = client.get_inner().await;
        let agent = client.agent("agent-model").build();
        let completion = client.completion("completion-model");
        let agent_cfg = client.config("agent-model");
        let completion_cfg = client.config("completion-model");

        let through_agent = agent.rt.bedrock_client(&agent_cfg).await;
        let through_completion = completion.runtime().bedrock_client(&completion_cfg).await;

        assert!(std::ptr::eq(direct.config(), through_agent.config()));
        assert!(std::ptr::eq(direct.config(), through_completion.config()));
    }
}
