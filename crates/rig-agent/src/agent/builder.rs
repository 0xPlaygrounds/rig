use std::{collections::HashMap, sync::Arc};

use schemars::{JsonSchema, Schema, schema_for};

use rig_core::message::ToolChoice;

use crate::{
    completion::Document,
    executor::ToolExecutor,
    hooks::{HookEntry, Hooks},
    provider::{ProviderConfig, Runtime},
    tool::{PortableDynamicTool, PortableTool},
};

use super::{Agent, OutputMode};

/// A builder for creating an agent.
///
/// Tools are plain [`PortableDynamicTool`] records: [`tool`](Self::tool)
/// erases a typed [`PortableTool`], while [`dynamic_tool`](Self::dynamic_tool)
/// registers a runtime-built record directly. The built agent holds the
/// records in a [`ToolExecutor`] and advertises their definitions as a
/// [`ToolCatalog`](crate::agent::prepare::ToolCatalog).
///
/// # Example
/// ```no_run
/// use rig_agent::AgentBuilder;
/// use rig_agent::provider::ProviderConfig;
/// use rig_core::providers::openai;
///
/// # fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let provider = ProviderConfig::OpenAi(openai::functions::Config::new(openai::GPT_5_2));
///
/// // Configure the agent
/// let agent = AgentBuilder::new(provider)
///     .preamble("System prompt")
///     .context("Context document 1")
///     .context("Context document 2")
///     .temperature(0.8)
///     .build();
/// # Ok(())
/// # }
/// ```
pub struct AgentBuilder {
    /// Name of the agent used for logging and debugging
    name: Option<String>,
    /// Agent description. Primarily useful when using sub-agents as part of an agent workflow and converting agents to other formats.
    description: Option<String>,
    /// Provider selection as plain configuration (which provider, base URL,
    /// credential location, and model identifier).
    provider: ProviderConfig,
    /// Live transport handles the built agent fulfils requests with. `None`
    /// creates a fresh [`Runtime`] at build time.
    runtime: Option<Arc<Runtime>>,
    /// System prompt
    preamble: Option<String>,
    /// Context documents always available to the agent
    static_context: Vec<Document>,
    /// Additional parameters to be passed to the model
    additional_params: Option<serde_json::Value>,
    /// Whether to record sensitive request, response, and tool content on telemetry spans.
    record_telemetry_content: bool,
    /// Maximum number of tokens for the completion
    max_tokens: Option<u64>,
    /// Temperature of the model
    temperature: Option<f64>,
    /// Whether or not the underlying LLM should be forced to use a tool before providing a response.
    tool_choice: Option<ToolChoice>,
    /// Default total model-call budget, including the initial call and retries.
    default_max_turns: Option<usize>,
    /// The executable tool records registered on this builder.
    executor: ToolExecutor,
    /// Default hook stack applied to every prompt request from the built agent.
    hooks: Hooks,
    /// Optional JSON Schema for structured output
    output_schema: Option<schemars::Schema>,
    /// How `output_schema` is enforced (tool vs native vs prompted; see #1928)
    output_mode: OutputMode,
}

impl AgentBuilder {
    /// Create a new agent builder for the given provider configuration.
    ///
    /// Accepts either a [`ProviderConfig`] or any provider's
    /// `functions::Config` directly, so the common path needs no enum
    /// wrapping:
    ///
    /// ```no_run
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// use rig_agent::AgentBuilder;
    /// use rig_core::providers::openai;
    ///
    /// let cfg = openai::functions::Config::from_env("gpt-4o")?;
    /// let agent = AgentBuilder::new(cfg).preamble("Be terse.").build();
    /// # let _ = agent;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(provider: impl Into<ProviderConfig>) -> Self {
        Self {
            name: None,
            description: None,
            provider: provider.into(),
            runtime: None,
            preamble: None,
            static_context: vec![],
            temperature: None,
            max_tokens: None,
            additional_params: None,
            record_telemetry_content: false,
            tool_choice: None,
            default_max_turns: None,
            executor: ToolExecutor::new(),
            hooks: Hooks::new(),
            output_schema: None,
            output_mode: OutputMode::default(),
        }
    }

    /// Share live transport handles (HTTP client and feature-gated provider
    /// clients) with other agents instead of building fresh ones at
    /// [`build`](Self::build) time.
    pub fn runtime(mut self, rt: Arc<Runtime>) -> Self {
        self.runtime = Some(rt);
        self
    }

    /// Set the name of the agent
    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the description of the agent
    pub fn description(mut self, description: &str) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the system prompt
    pub fn preamble(mut self, preamble: &str) -> Self {
        self.preamble = Some(preamble.into());
        self
    }

    /// Remove the system prompt
    pub fn without_preamble(mut self) -> Self {
        self.preamble = None;
        self
    }

    /// Append to the preamble of the agent
    pub fn append_preamble(mut self, doc: &str) -> Self {
        self.preamble = Some(format!("{}\n{}", self.preamble.unwrap_or_default(), doc));
        self
    }

    /// Add a static context document to the agent
    pub fn context(mut self, doc: &str) -> Self {
        self.static_context.push(Document {
            id: format!("static_doc_{}", self.static_context.len()),
            text: doc.into(),
            additional_props: HashMap::new(),
        });
        self
    }

    /// Set the tool choice for the agent
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Set the default total model-call budget, including the initial call and
    /// every retry or continuation. Zero permits no model calls.
    pub fn default_max_turns(mut self, default_max_turns: usize) -> Self {
        self.default_max_turns = Some(default_max_turns);
        self
    }

    /// Set the temperature of the model
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the maximum number of tokens for the completion
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set additional parameters to be passed to the model
    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.additional_params = Some(params);
        self
    }

    /// Opt in or out of recording sensitive request, response, and tool content
    /// on GenAI telemetry spans for requests made by this agent.
    ///
    /// Defaults to `false`. Enabling this can expose prompts, retrieved context,
    /// tool results, model responses, and other sensitive or high-cardinality data
    /// through OpenTelemetry span attributes, which can increase observability
    /// backend storage and query costs. Only enable it when content telemetry is
    /// acceptable for this agent. Structural metadata and token usage remain
    /// available when this is disabled.
    pub fn record_content_telemetry(mut self, enabled: bool) -> Self {
        self.record_telemetry_content = enabled;
        self
    }

    /// Set the output schema for structured output. When set, providers that support
    /// native structured outputs will constrain the model's response to match this schema.
    pub fn output_schema<T>(mut self) -> Self
    where
        T: JsonSchema,
    {
        self.output_schema = Some(schema_for!(T));
        self
    }

    /// Set the output schema for structured output. In comparison to `AgentBuilder::schema()` which requires type annotation, you can put in any schema you'd like here.
    pub fn output_schema_raw(mut self, schema: Schema) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Set how `output_schema` is enforced — [`OutputMode::Tool`] (output as a
    /// tool call, the default when the agent has tools), [`OutputMode::Native`]
    /// (provider structured output), or [`OutputMode::Prompted`] (see #1928).
    /// Has no effect unless `output_schema`/`output_schema_raw` is also set.
    pub fn output_mode(mut self, mode: OutputMode) -> Self {
        self.output_mode = mode;
        self
    }

    /// Attach a default hook to the agent. Each call appends to the agent's hook
    /// stack; hooks run for every prompt request (unless more are added per
    /// request) in registration order. How their results compose is
    /// event-dependent: `CompletionCall` request patches accumulate and merge,
    /// `ToolCall`/`ToolResult` rewrites chain, while model-turn steering and
    /// observe-only/recovery events use first-non-`Continue`-wins. See the
    /// [`hook`](crate::agent::hook) module docs.
    pub fn add_hook(mut self, hook: HookEntry) -> Self {
        self.hooks.add(hook);
        self
    }

    /// Add a typed static tool to the agent. The tool is erased into a
    /// [`PortableDynamicTool`] record; registering the same name again
    /// replaces the earlier registration in place.
    pub fn tool<T>(mut self, tool: T) -> Self
    where
        T: PortableTool + 'static,
    {
        self.executor = self
            .executor
            .register(PortableDynamicTool::from_portable(tool));
        self
    }

    /// Add one runtime-defined tool record to the agent.
    pub fn dynamic_tool(mut self, tool: PortableDynamicTool) -> Self {
        self.executor = self.executor.register(tool);
        self
    }

    /// Add runtime-defined tool records to the agent.
    ///
    /// This is useful when tool definitions and callbacks are constructed at runtime.
    pub fn dynamic_tools(mut self, tools: impl IntoIterator<Item = PortableDynamicTool>) -> Self {
        for tool in tools {
            self.executor = self.executor.register(tool);
        }
        self
    }

    /// Build the agent with the configured tools.
    pub fn build(self) -> Agent {
        let config = crate::agent::AgentConfig {
            name: self.name,
            description: self.description,
            preamble: self.preamble,
            static_context: self.static_context,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            additional_params: self.additional_params,
            tool_choice: self.tool_choice,
            max_turns: self.default_max_turns,
            output_schema: self.output_schema,
            output_mode: self.output_mode,
            record_telemetry_content: self.record_telemetry_content,
            ..crate::agent::AgentConfig::new()
        };
        Agent {
            config,
            provider: self.provider,
            rt: self.runtime.unwrap_or_else(|| Arc::new(Runtime::new())),
            tools: self.executor.catalog(),
            executor: Some(self.executor),
            hooks: self.hooks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::HookDecision;
    use crate::provider::MockScript;
    use crate::test_utils::MockAddTool;
    use crate::tool::{ToolExecutionError, ToolOutput};
    use rig_core::OneOrMany;
    use rig_core::completion::{CompletionResponse, Usage};
    use rig_core::message::AssistantContent;

    /// A mock provider config scripted to answer one call with `text`.
    fn mock_text(text: &str) -> ProviderConfig {
        ProviderConfig::Mock(MockScript::from_responses(vec![CompletionResponse::new(
            OneOrMany::one(AssistantContent::text(text)),
            Usage::new(),
            "mock",
        )]))
    }

    /// A no-op hook entry: registration must survive tool configuration.
    fn builder_hook() -> HookEntry {
        HookEntry::new("builder-hook", |_| {
            Box::pin(async { HookDecision::Continue })
        })
    }

    #[test]
    fn hook_can_be_set_after_tool_configuration() {
        let _agent = AgentBuilder::new(mock_text("ok"))
            .tool(MockAddTool)
            .add_hook(builder_hook())
            .build();
    }

    struct NamedTool;

    impl NamedTool {
        fn new() -> Self {
            Self
        }
    }

    impl PortableTool for NamedTool {
        const NAME: &'static str = "registered_named";
        type Error = ToolExecutionError;
        type Args = serde_json::Value;
        type Output = String;

        fn description(&self) -> String {
            "uses its canonical name".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, ToolExecutionError> {
            Ok("ok".to_string())
        }
    }

    #[tokio::test]
    async fn typed_tool_builder_paths_advertise_canonical_name() {
        for agent in [
            AgentBuilder::new(mock_text("ok"))
                .tool(NamedTool::new())
                .build(),
            AgentBuilder::new(mock_text("ok"))
                .tool(MockAddTool)
                .tool(NamedTool::new())
                .build(),
        ] {
            let definitions = agent.tool_definitions().await;
            assert!(
                definitions
                    .iter()
                    .any(|definition| definition.name == NamedTool::NAME),
                "the provider definitions dropped the canonical tool name"
            );

            let tool = agent
                .executor
                .as_ref()
                .expect("the builder always attaches an executor")
                .get(NamedTool::NAME)
                .expect("the tool record is registered");
            let result = tool
                .execute(serde_json::json!({}))
                .await
                .expect("execution succeeds");
            assert_eq!(result, ToolOutput::text("ok"));
        }
    }
}
