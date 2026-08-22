use std::{collections::HashMap, sync::Arc};

use schemars::{JsonSchema, Schema, schema_for};

use rig_core::{
    memory::ConversationMemory,
    message::ToolChoice,
    vector_store::{VectorSearchRequest, VectorStoreIndexDyn},
};

use crate::{
    agent::hook::{AgentHook, CompletionCall, CompletionCallAction, HookContext, RequestPatch},
    completion::{CompletionModel, Document},
    tool::{
        DynamicTool, PortableDynamicTool, Tool, ToolSet,
        server::{ToolServer, ToolServerHandle},
    },
};

use super::{Agent, ModelHandle, OutputMode, completion::AgentConfig};

struct DynamicContext<I> {
    samples: usize,
    index: I,
}

impl<I> AgentHook for DynamicContext<I>
where
    I: VectorStoreIndexDyn,
{
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        event: CompletionCall<'_>,
    ) -> CompletionCallAction {
        let query = event.prompt.rag_text().or_else(|| {
            event
                .history
                .iter()
                .rev()
                .find_map(|message| message.rag_text())
        });
        let Some(query) = query else {
            return CompletionCallAction::continue_run();
        };

        let request = VectorSearchRequest::builder()
            .query(query)
            .samples(self.samples as u64)
            .build();
        match self.index.top_n(request).await {
            Ok(results) => CompletionCallAction::patch(RequestPatch::new().extra_context(
                results.into_iter().map(|(_, id, value)| Document {
                    id,
                    text:
                        serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string()),
                    additional_props: Default::default(),
                }),
            )),
            Err(error) => {
                CompletionCallAction::stop(format!("failed to retrieve dynamic context: {error}"))
            }
        }
    }
}

/// Marker type indicating no tool configuration has been set yet.
///
/// This is the default state for a new `AgentBuilder`. From this state,
/// you can either:
/// - Add tools via `.tool()`, `.dynamic_tool()`, `.dynamic_tools()`, or
///   `.retrieved_tools()` (transitions to `WithBuilderTools`)
/// - Set a pre-existing `ToolServerHandle` via `.tool_server_handle()` (transitions to `WithToolServerHandle`)
/// - Call `.build()` to create an agent with no tools
#[derive(Default)]
pub struct NoToolConfig;

/// Typestate indicating a pre-existing `ToolServerHandle` has been provided.
///
/// In this state, tool-adding methods (`.tool()`, `.dynamic_tool()`, etc.) are not available.
/// The provided handle will be used directly when building the agent.
pub struct WithToolServerHandle {
    handle: ToolServerHandle,
}

/// Typestate indicating tools are being configured via the builder API.
///
/// In this state, you can continue adding tools via `.tool()`,
/// `.dynamic_tool()`, `.dynamic_tools()`, and `.retrieved_tools()`. When
/// `.build()` is called, a new `ToolServer`
/// will be created with all the configured tools.
pub struct WithBuilderTools(ToolServer);

/// A builder for creating an agent
///
/// The builder uses a typestate pattern to enforce that tool configuration
/// is done in a mutually exclusive way: either provide a pre-existing
/// `ToolServerHandle`, or add tools via the builder API, but not both.
///
/// # Example
/// ```no_run
/// use rig_agent::AgentBuilder;
/// use rig_core::{client::CompletionClient, providers::openai};
/// use rig_reqwest::prelude::*;
///
/// # fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let openai = openai::Client::from_env()?;
///
/// let model = openai.completion_model(openai::GPT_5_2);
///
/// // Configure the agent
/// let agent = AgentBuilder::new(model)
///     .preamble("System prompt")
///     .context("Context document 1")
///     .context("Context document 2")
///     .temperature(0.8)
///     .build();
/// # Ok(())
/// # }
/// ```
pub struct AgentBuilder<ToolState = NoToolConfig> {
    /// Everything the built [`Agent`] carries unchanged.
    config: AgentConfig,
    /// Tool configuration state (typestate pattern)
    tool_state: ToolState,
}

impl<ToolState> AgentBuilder<ToolState> {
    /// Set the name of the agent
    pub fn name(mut self, name: &str) -> Self {
        self.config.name = Some(name.into());
        self
    }

    /// Set the description of the agent
    pub fn description(mut self, description: &str) -> Self {
        self.config.description = Some(description.into());
        self
    }

    /// Set the system prompt
    pub fn preamble(mut self, preamble: &str) -> Self {
        self.config.preamble = Some(preamble.into());
        self
    }

    /// Remove the system prompt
    pub fn without_preamble(mut self) -> Self {
        self.config.preamble = None;
        self
    }

    /// Append to the preamble of the agent
    pub fn append_preamble(mut self, doc: &str) -> Self {
        self.config.preamble = Some(format!(
            "{}\n{}",
            self.config.preamble.unwrap_or_default(),
            doc
        ));
        self
    }

    /// Add a static context document to the agent
    pub fn context(mut self, doc: &str) -> Self {
        self.config.static_context.push(Document {
            id: format!("static_doc_{}", self.config.static_context.len()),
            text: doc.into(),
            additional_props: HashMap::new(),
        });
        self
    }

    /// Add dynamic context retrieved from a vector store on every model call.
    ///
    /// This is a convenience wrapper around an internal completion-call hook.
    /// The hook searches with the current prompt's first text part, falling back
    /// to the latest textual history message, and appends the retrieved documents
    /// to the request after static context. Retrieval and injected documents
    /// follow registration order relative to application hooks, so register a
    /// stop policy before this helper when it should prevent retrieval. A
    /// retrieval failure stops the run before provider I/O.
    pub fn dynamic_context<I>(self, samples: usize, index: I) -> Self
    where
        I: VectorStoreIndexDyn + 'static,
    {
        self.add_hook(DynamicContext { samples, index })
    }

    /// Set the tool choice for the agent
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.config.tool_choice = Some(tool_choice);
        self
    }

    /// Set the default total model-call budget, including the initial call and
    /// every retry or continuation. Zero permits no model calls.
    pub fn default_max_turns(mut self, default_max_turns: usize) -> Self {
        self.config.max_turns = default_max_turns;
        self
    }

    /// Set the temperature of the model
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    /// Set the maximum number of tokens for the completion
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    /// Set additional parameters to be passed to the model
    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.config.additional_params = Some(params);
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
        self.config.record_telemetry_content = enabled;
        self
    }

    /// Set the output schema for structured output. When set, providers that support
    /// native structured outputs will constrain the model's response to match this schema.
    pub fn output_schema<T>(mut self) -> Self
    where
        T: JsonSchema,
    {
        self.config.output_schema = Some(schema_for!(T));
        self
    }

    /// Set the output schema for structured output. In comparison to `AgentBuilder::schema()` which requires type annotation, you can put in any schema you'd like here.
    pub fn output_schema_raw(mut self, schema: Schema) -> Self {
        self.config.output_schema = Some(schema);
        self
    }

    /// Set how `output_schema` is enforced — [`OutputMode::Tool`] (output as a
    /// tool call, the default when the agent has tools), [`OutputMode::Native`]
    /// (provider structured output), or [`OutputMode::Prompted`] (see #1928).
    /// Has no effect unless `output_schema`/`output_schema_raw` is also set.
    pub fn output_mode(mut self, mode: OutputMode) -> Self {
        self.config.output_mode = mode;
        self
    }

    /// Attach a [`ConversationMemory`] backend.
    ///
    /// When set, the agent will automatically load prior conversation history before
    /// each prompt and append the new turn after a successful response. A
    /// `conversation_id` must be supplied either via [`AgentBuilder::conversation`]
    /// or per-request via [`crate::agent::prompt_request::PromptRequest::conversation`].
    /// If neither is set, memory is silently bypassed.
    pub fn memory<B>(mut self, memory: B) -> Self
    where
        B: ConversationMemory + 'static,
    {
        self.config.memory = Some(Arc::new(memory));
        self
    }

    /// Set a default conversation id used when none is provided per-request.
    ///
    /// Most agents are reused across users or threads; prefer setting the id
    /// per-request via [`crate::agent::prompt_request::PromptRequest::conversation`].
    pub fn conversation(mut self, id: impl Into<String>) -> Self {
        self.config.conversation_id = Some(id.into());
        self
    }

    /// Attach a default hook to the agent. Each call appends to the agent's hook
    /// stack; hooks run for every prompt request (unless more are added per
    /// request) in registration order. How their results compose is
    /// event-dependent: model selections and `ToolCall`/`ToolResult` rewrites
    /// chain, `CompletionCall` request patches accumulate and merge, while
    /// model-turn steering and observe-only/recovery events use
    /// first-non-`Continue`-wins. See the [`hook`](crate::agent::hook) module
    /// docs.
    pub fn add_hook<H>(mut self, hook: H) -> Self
    where
        H: AgentHook + 'static,
    {
        self.config.hooks.push(hook);
        self
    }

    /// Carry the configuration into a builder with a new tool state.
    fn with_tool_state<S>(self, tool_state: S) -> AgentBuilder<S> {
        AgentBuilder {
            config: self.config,
            tool_state,
        }
    }

    /// Assemble the [`Agent`], resolving the tool server handle from the final
    /// tool state.
    fn build_agent(self, handle: impl FnOnce(ToolState) -> ToolServerHandle) -> Agent {
        Agent {
            tool_server_handle: handle(self.tool_state),
            config: self.config,
        }
    }
}

impl AgentBuilder<NoToolConfig> {
    /// Create a new agent builder with the given model.
    ///
    /// The typed model is erased once, here, into a [`ModelHandle`]; the built
    /// [`Agent`] carries no model type parameter.
    pub fn new<M>(model: M) -> Self
    where
        M: CompletionModel + 'static,
    {
        Self::from_model_handle(ModelHandle::new(model))
    }

    /// Create an agent builder from an already-erased runtime model handle.
    pub fn from_model_handle(model: ModelHandle) -> Self {
        Self {
            config: AgentConfig::new(model),
            tool_state: NoToolConfig,
        }
    }
}

impl AgentBuilder<NoToolConfig> {
    /// Set a pre-existing ToolServerHandle for the agent.
    ///
    /// After calling this method, tool-adding methods (`.tool()`, `.dynamic_tool()`, etc.)
    /// will not be available. Use this when you want to share a `ToolServer`
    /// between multiple agents or have pre-configured tools.
    pub fn tool_server_handle(
        self,
        handle: ToolServerHandle,
    ) -> AgentBuilder<WithToolServerHandle> {
        self.with_tool_state(WithToolServerHandle { handle })
    }

    /// Transition into the `WithBuilderTools` state with no tools yet; every
    /// tool-adding method below is the `WithBuilderTools` method after this
    /// one-way step.
    fn into_tool_builder(self) -> AgentBuilder<WithBuilderTools> {
        self.with_tool_state(WithBuilderTools(ToolServer::new()))
    }

    /// Add a static tool to the agent.
    ///
    /// This transitions the builder to the `WithBuilderTools` state, where
    /// additional tools can be added but `tool_server_handle()` is no longer available.
    pub fn tool<T>(self, tool: T) -> AgentBuilder<WithBuilderTools>
    where
        T: Tool + 'static,
    {
        self.into_tool_builder().tool(tool)
    }

    /// Add a pre-erased tool — the extension point for adapters that implement
    /// [`ErasedTool`](crate::tool::ErasedTool) directly (remote tool protocols
    /// such as MCP, provided by companion crates). Transitions the builder to
    /// the `WithBuilderTools` state.
    pub fn erased_tool(
        self,
        tool: std::sync::Arc<dyn crate::tool::ErasedTool>,
    ) -> AgentBuilder<WithBuilderTools> {
        self.into_tool_builder().erased_tool(tool)
    }

    /// Build the agent with no tools configured.
    ///
    /// An empty `ToolServer` will be created for the agent.
    pub fn build(self) -> Agent {
        self.build_agent(|_| ToolServer::new().run())
    }
}

/// Generate the `NoToolConfig` tool methods that transition into the
/// `WithBuilderTools` state by forwarding verbatim through
/// [`AgentBuilder::into_tool_builder`] to the `WithBuilderTools` method of the
/// same name. Doc comments live at each invocation; `tool` (generic over the
/// tool type) and the single-tool `erased_tool` helper stay hand-written above.
macro_rules! forward_into_tool_builder {
    ($( $(#[$attr:meta])* $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) );* $(;)?) => {
        impl AgentBuilder<NoToolConfig> {
            $(
                $(#[$attr])*
                pub fn $name(self, $($arg: $ty),*) -> AgentBuilder<WithBuilderTools> {
                    self.into_tool_builder().$name($($arg),*)
                }
            )*
        }
    };
}

forward_into_tool_builder! {
    /// Add one runtime-defined tool to the agent.
    dynamic_tool(tool: DynamicTool);

    /// Add one context-free dynamic tool through the classic registry adapter.
    portable_dynamic_tool(tool: PortableDynamicTool);

    /// Add runtime-defined tools to the agent.
    ///
    /// This is useful when tool definitions and callbacks are constructed at runtime.
    /// Transitions the builder to the `WithBuilderTools` state.
    dynamic_tools(tools: Vec<DynamicTool>);


    /// Configure tools retrieved from a vector index for each prompt.
    ///
    /// Transitions the builder to the `WithBuilderTools` state.
    retrieved_tools(
        sample: usize,
        index: impl VectorStoreIndexDyn + Send + Sync + 'static,
        toolset: ToolSet
    );
}

impl AgentBuilder<WithToolServerHandle> {
    /// Build the agent using the pre-configured ToolServerHandle.
    pub fn build(self) -> Agent {
        self.build_agent(|state| state.handle)
    }
}

impl AgentBuilder<WithBuilderTools> {
    /// Configure the [`ToolServer`] the builder is accumulating tools into. Every
    /// tool-adding method here is one of its registrations, so registration
    /// semantics live in exactly one place.
    fn map_server(self, register: impl FnOnce(ToolServer) -> ToolServer) -> Self {
        let Self { config, tool_state } = self;
        Self {
            config,
            tool_state: WithBuilderTools(register(tool_state.0)),
        }
    }

    /// Add another static tool to the agent.
    pub fn tool<T>(self, tool: T) -> Self
    where
        T: Tool + 'static,
    {
        self.map_server(|server| server.tool(tool))
    }

    /// Add one runtime-defined tool to the agent.
    pub fn dynamic_tool(self, tool: DynamicTool) -> Self {
        self.map_server(|server| server.dynamic_tool(tool))
    }

    /// Add one context-free dynamic tool through the classic registry adapter.
    pub fn portable_dynamic_tool(self, tool: PortableDynamicTool) -> Self {
        self.map_server(|server| server.portable_dynamic_tool(tool))
    }

    /// Add runtime-defined tools to the agent.
    pub fn dynamic_tools(self, tools: Vec<DynamicTool>) -> Self {
        self.map_server(|server| server.dynamic_tools(tools))
    }

    /// Add a pre-erased tool — the extension point for adapters that implement
    /// [`ErasedTool`](crate::tool::ErasedTool) directly (remote tool protocols
    /// such as MCP, provided by companion crates).
    pub fn erased_tool(self, tool: std::sync::Arc<dyn crate::tool::ErasedTool>) -> Self {
        self.map_server(|server| server.erased_tool(tool))
    }

    /// Configure tools retrieved from a vector index for each prompt.
    pub fn retrieved_tools(
        self,
        sample: usize,
        index: impl VectorStoreIndexDyn + Send + Sync + 'static,
        toolset: ToolSet,
    ) -> Self {
        self.map_server(|server| server.retrieved_tools(sample, index, toolset))
    }

    /// Build the agent with the configured tools.
    ///
    /// A new `ToolServer` will be created containing all tools added via
    /// `.tool()`, `.dynamic_tool()`, `.dynamic_tools()`, and
    /// `.retrieved_tools()`.
    pub fn build(self) -> Agent {
        self.build_agent(|state| state.0.run())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{MockAddTool, MockCompletionModel, MockSubtractTool, MockToolIndex};
    use crate::tool::{ToolContext, ToolExecutionError};

    #[derive(Clone)]
    struct BuilderHook;

    impl AgentHook for BuilderHook {}

    #[test]
    fn hook_can_be_set_after_tool_configuration() {
        let _agent = AgentBuilder::new(MockCompletionModel::text("ok"))
            .tool(MockAddTool)
            .add_hook(BuilderHook)
            .build();
    }

    struct NamedTool;

    impl NamedTool {
        fn new() -> Self {
            Self
        }
    }

    impl Tool for NamedTool {
        const NAME: &'static str = "registered_named";
        type Error = rig::tool::ToolExecutionError;
        type Args = serde_json::Value;
        type Output = String;

        fn description(&self) -> String {
            "uses its canonical name".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, ToolExecutionError> {
            Ok("ok".to_string())
        }
    }

    #[tokio::test]
    async fn typed_tool_builder_paths_advertise_canonical_name() {
        for agent in [
            AgentBuilder::new(MockCompletionModel::text("ok"))
                .tool(NamedTool::new())
                .build(),
            AgentBuilder::new(MockCompletionModel::text("ok"))
                .tool(MockAddTool)
                .tool(NamedTool::new())
                .build(),
        ] {
            let definitions = agent.tool_server_handle.get_tool_defs(None).await.unwrap();
            assert!(
                definitions
                    .iter()
                    .any(|definition| definition.name == NamedTool::NAME),
                "the provider definitions dropped the canonical tool name"
            );

            let mut context = ToolContext::new();
            let result = agent
                .tool_server_handle
                .execute(NamedTool::NAME, "{}", &mut context)
                .await;
            assert!(result.is_success());
            assert_eq!(result.output().as_text(), Some("ok"));
        }
    }

    #[tokio::test]
    async fn retrieved_tools_are_exposed_only_for_prompted_retrieval() {
        let retrieval_only = AgentBuilder::new(MockCompletionModel::text("ok"))
            .retrieved_tools(
                1,
                MockToolIndex::new(["add"]),
                ToolSet::from_tools(vec![MockAddTool]),
            )
            .build();
        assert!(
            retrieval_only
                .tool_server_handle
                .get_tool_defs(None)
                .await
                .unwrap()
                .is_empty()
        );

        let agent = AgentBuilder::new(MockCompletionModel::text("ok"))
            .tool(MockSubtractTool)
            .retrieved_tools(
                1,
                MockToolIndex::new(["add"]),
                ToolSet::from_tools(vec![MockAddTool]),
            )
            .build();

        let always = agent.tool_server_handle.get_tool_defs(None).await.unwrap();
        assert_eq!(
            always
                .iter()
                .map(|definition| definition.name.as_str())
                .collect::<Vec<_>>(),
            vec!["subtract"]
        );

        let with_retrieval = agent
            .tool_server_handle
            .get_tool_defs(Some("add two numbers".to_string()))
            .await
            .unwrap();
        assert_eq!(
            with_retrieval
                .iter()
                .map(|definition| definition.name.as_str())
                .collect::<Vec<_>>(),
            vec!["add", "subtract"]
        );
    }
}
