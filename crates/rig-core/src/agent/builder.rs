use std::{collections::HashMap, sync::Arc};

use schemars::{JsonSchema, Schema, schema_for};

use crate::{
    agent::prompt_request::hooks::PromptHook,
    completion::{CompletionModel, Document},
    memory::ConversationMemory,
    message::ToolChoice,
    tool::{
        Tool, ToolDyn, ToolSet,
        server::{ToolServer, ToolServerHandle},
    },
    vector_store::VectorStoreIndexDyn,
};

#[cfg(feature = "rmcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
use crate::tool::rmcp::McpTool as RmcpTool;

use super::{Agent, OutputMode};

/// Build [`RmcpTool`]s from MCP tool definitions, applying the given per-call
/// timeout to each (`None` disables it; see issue #1914). Returns
/// `(tool_name, tool)` pairs.
#[cfg(feature = "rmcp")]
fn build_rmcp_tools(
    tools: Vec<rmcp::model::Tool>,
    client: rmcp::service::ServerSink,
    timeout: Option<std::time::Duration>,
) -> Vec<(String, RmcpTool)> {
    tools
        .into_iter()
        .map(|tool| {
            let name = tool.name.to_string();
            let rmcp_tool = RmcpTool::from_mcp_server(tool, client.clone()).with_timeout(timeout);
            (name, rmcp_tool)
        })
        .collect()
}

/// Marker type indicating no tool configuration has been set yet.
///
/// This is the default state for a new `AgentBuilder`. From this state,
/// you can either:
/// - Add tools via `.tool()`, `.tools()`, `.dynamic_tools()`, etc. (transitions to `WithBuilderTools`)
/// - Set a pre-existing `ToolServerHandle` via `.tool_server_handle()` (transitions to `WithToolServerHandle`)
/// - Call `.build()` to create an agent with no tools
#[derive(Default)]
pub struct NoToolConfig;

/// Typestate indicating a pre-existing `ToolServerHandle` has been provided.
///
/// In this state, tool-adding methods (`.tool()`, `.tools()`, etc.) are not available.
/// The provided handle will be used directly when building the agent.
pub struct WithToolServerHandle {
    handle: ToolServerHandle,
}

/// Typestate indicating tools are being configured via the builder API.
///
/// In this state, you can continue adding tools via `.tool()`, `.tools()`,
/// `.dynamic_tools()`, etc. When `.build()` is called, a new `ToolServer`
/// will be created with all the configured tools.
pub struct WithBuilderTools {
    static_tools: Vec<String>,
    tools: ToolSet,
    dynamic_tools: Vec<(usize, Arc<dyn VectorStoreIndexDyn + Send + Sync>)>,
}

/// A builder for creating an agent
///
/// The builder uses a typestate pattern to enforce that tool configuration
/// is done in a mutually exclusive way: either provide a pre-existing
/// `ToolServerHandle`, or add tools via the builder API, but not both.
///
/// # Example
/// ```no_run
/// use rig_core::{agent::AgentBuilder, client::{CompletionClient, ProviderClient}, providers::openai};
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
pub struct AgentBuilder<M, P = (), ToolState = NoToolConfig>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    /// Name of the agent used for logging and debugging
    name: Option<String>,
    /// Agent description. Primarily useful when using sub-agents as part of an agent workflow and converting agents to other formats.
    description: Option<String>,
    /// Completion model (e.g.: OpenAI's gpt-3.5-turbo-1106, Cohere's command-r)
    model: M,
    /// System prompt
    preamble: Option<String>,
    /// Context documents always available to the agent
    static_context: Vec<Document>,
    /// Additional parameters to be passed to the model
    additional_params: Option<serde_json::Value>,
    /// Maximum number of tokens for the completion
    max_tokens: Option<u64>,
    /// List of vector store, with the sample number
    dynamic_context: Vec<(usize, Arc<dyn VectorStoreIndexDyn + Send + Sync>)>,
    /// Temperature of the model
    temperature: Option<f64>,
    /// Whether or not the underlying LLM should be forced to use a tool before providing a response.
    tool_choice: Option<ToolChoice>,
    /// Default maximum depth for multi-turn agent calls
    default_max_turns: Option<usize>,
    /// Tool configuration state (typestate pattern)
    tool_state: ToolState,
    /// Prompt hook
    hook: Option<P>,
    /// Optional JSON Schema for structured output
    output_schema: Option<schemars::Schema>,
    /// How `output_schema` is enforced (tool vs native vs prompted; see #1928)
    output_mode: OutputMode,
    /// Optional conversation memory backend that loads/saves history per conversation id.
    memory: Option<Arc<dyn ConversationMemory>>,
    /// Optional default conversation id used when none is set per-request.
    default_conversation_id: Option<String>,
}

impl<M, P, ToolState> AgentBuilder<M, P, ToolState>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
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

    /// Add some dynamic context to the agent. On each prompt, `sample` documents from the
    /// dynamic context will be inserted in the request.
    pub fn dynamic_context(
        mut self,
        sample: usize,
        dynamic_context: impl VectorStoreIndexDyn + Send + Sync + 'static,
    ) -> Self {
        self.dynamic_context
            .push((sample, Arc::new(dynamic_context)));
        self
    }

    /// Set the tool choice for the agent
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Set the default maximum depth that an agent will use for multi-turn.
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

    /// Attach a [`ConversationMemory`] backend.
    ///
    /// When set, the agent will automatically load prior conversation history before
    /// each prompt and append the new turn after a successful response. A
    /// `conversation_id` must be supplied either via [`AgentBuilder::conversation_id`]
    /// or per-request via [`crate::agent::prompt_request::PromptRequest::conversation`].
    /// If neither is set, memory is silently bypassed.
    pub fn memory<B>(mut self, memory: B) -> Self
    where
        B: ConversationMemory + 'static,
    {
        self.memory = Some(Arc::new(memory));
        self
    }

    /// Set a default conversation id used when none is provided per-request.
    ///
    /// Most agents are reused across users or threads; prefer setting the id
    /// per-request via [`crate::agent::prompt_request::PromptRequest::conversation`].
    pub fn conversation_id(mut self, id: impl Into<String>) -> Self {
        self.default_conversation_id = Some(id.into());
        self
    }

    /// Set the default hook for the agent.
    ///
    /// This hook will be used for all prompt requests unless overridden
    /// via `.with_hook()` on the request.
    pub fn hook<P2>(self, hook: P2) -> AgentBuilder<M, P2, ToolState>
    where
        P2: PromptHook<M>,
    {
        AgentBuilder {
            name: self.name,
            description: self.description,
            model: self.model,
            preamble: self.preamble,
            static_context: self.static_context,
            additional_params: self.additional_params,
            max_tokens: self.max_tokens,
            dynamic_context: self.dynamic_context,
            temperature: self.temperature,
            tool_choice: self.tool_choice,
            default_max_turns: self.default_max_turns,
            tool_state: self.tool_state,
            hook: Some(hook),
            output_schema: self.output_schema,
            output_mode: self.output_mode,
            memory: self.memory,
            default_conversation_id: self.default_conversation_id,
        }
    }
}

impl<M> AgentBuilder<M, (), NoToolConfig>
where
    M: CompletionModel,
{
    /// Create a new agent builder with the given model
    pub fn new(model: M) -> Self {
        Self {
            name: None,
            description: None,
            model,
            preamble: None,
            static_context: vec![],
            temperature: None,
            max_tokens: None,
            additional_params: None,
            dynamic_context: vec![],
            tool_choice: None,
            default_max_turns: None,
            tool_state: NoToolConfig,
            hook: None,
            output_schema: None,
            output_mode: OutputMode::default(),
            memory: None,
            default_conversation_id: None,
        }
    }
}

impl<M, P> AgentBuilder<M, P, NoToolConfig>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    /// Set a pre-existing ToolServerHandle for the agent.
    ///
    /// After calling this method, tool-adding methods (`.tool()`, `.tools()`, etc.)
    /// will not be available. Use this when you want to share a `ToolServer`
    /// between multiple agents or have pre-configured tools.
    pub fn tool_server_handle(
        self,
        handle: ToolServerHandle,
    ) -> AgentBuilder<M, P, WithToolServerHandle> {
        AgentBuilder {
            name: self.name,
            description: self.description,
            model: self.model,
            preamble: self.preamble,
            static_context: self.static_context,
            additional_params: self.additional_params,
            max_tokens: self.max_tokens,
            dynamic_context: self.dynamic_context,
            temperature: self.temperature,
            tool_choice: self.tool_choice,
            default_max_turns: self.default_max_turns,
            tool_state: WithToolServerHandle { handle },
            hook: self.hook,
            output_schema: self.output_schema,
            output_mode: self.output_mode,
            memory: self.memory,
            default_conversation_id: self.default_conversation_id,
        }
    }

    /// Add a static tool to the agent.
    ///
    /// This transitions the builder to the `WithBuilderTools` state, where
    /// additional tools can be added but `tool_server_handle()` is no longer available.
    pub fn tool(self, tool: impl Tool + 'static) -> AgentBuilder<M, P, WithBuilderTools> {
        let toolname = tool.name();
        AgentBuilder {
            name: self.name,
            description: self.description,
            model: self.model,
            preamble: self.preamble,
            static_context: self.static_context,
            additional_params: self.additional_params,
            max_tokens: self.max_tokens,
            dynamic_context: self.dynamic_context,
            temperature: self.temperature,
            tool_choice: self.tool_choice,
            default_max_turns: self.default_max_turns,
            tool_state: WithBuilderTools {
                static_tools: vec![toolname],
                tools: ToolSet::from_tools(vec![tool]),
                dynamic_tools: vec![],
            },
            hook: self.hook,
            output_schema: self.output_schema,
            output_mode: self.output_mode,
            memory: self.memory,
            default_conversation_id: self.default_conversation_id,
        }
    }

    /// Add a vector of boxed static tools to the agent.
    ///
    /// This is useful when you need to dynamically add static tools to the agent.
    /// Transitions the builder to the `WithBuilderTools` state.
    pub fn tools(self, tools: Vec<Box<dyn ToolDyn>>) -> AgentBuilder<M, P, WithBuilderTools> {
        let static_tools = tools.iter().map(|tool| tool.name()).collect();
        let tools = ToolSet::from_tools_boxed(tools);

        AgentBuilder {
            name: self.name,
            description: self.description,
            model: self.model,
            preamble: self.preamble,
            static_context: self.static_context,
            additional_params: self.additional_params,
            max_tokens: self.max_tokens,
            dynamic_context: self.dynamic_context,
            temperature: self.temperature,
            tool_choice: self.tool_choice,
            default_max_turns: self.default_max_turns,
            hook: self.hook,
            output_schema: self.output_schema,
            output_mode: self.output_mode,
            memory: self.memory,
            default_conversation_id: self.default_conversation_id,
            tool_state: WithBuilderTools {
                static_tools,
                tools,
                dynamic_tools: vec![],
            },
        }
    }

    /// Add an MCP tool (from `rmcp`) to the agent, bounded by
    /// [`DEFAULT_MCP_TOOL_TIMEOUT`](crate::tool::rmcp::DEFAULT_MCP_TOOL_TIMEOUT)
    /// (see issue #1914). Use [`rmcp_tool_with_timeout`](Self::rmcp_tool_with_timeout)
    /// to change or disable it.
    ///
    /// Transitions the builder to the `WithBuilderTools` state.
    #[cfg(feature = "rmcp")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    pub fn rmcp_tool(
        self,
        tool: rmcp::model::Tool,
        client: rmcp::service::ServerSink,
    ) -> AgentBuilder<M, P, WithBuilderTools> {
        self.rmcp_tool_with_timeout(tool, client, crate::tool::rmcp::DEFAULT_MCP_TOOL_TIMEOUT)
    }

    /// Add an MCP tool (from `rmcp`) with a per-call timeout (see issue #1914).
    ///
    /// Pass a [`Duration`](std::time::Duration) to bound the call, or `None` to
    /// disable the timeout (unbounded). On timeout the call resolves to a tool
    /// error the agent can recover from instead of blocking forever.
    /// Transitions the builder to the `WithBuilderTools` state.
    #[cfg(feature = "rmcp")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    pub fn rmcp_tool_with_timeout(
        self,
        tool: rmcp::model::Tool,
        client: rmcp::service::ServerSink,
        timeout: impl Into<Option<std::time::Duration>>,
    ) -> AgentBuilder<M, P, WithBuilderTools> {
        self.with_rmcp_toolset(build_rmcp_tools(vec![tool], client, timeout.into()))
    }

    /// Add an array of MCP tools (from `rmcp`) to the agent, each bounded by
    /// [`DEFAULT_MCP_TOOL_TIMEOUT`](crate::tool::rmcp::DEFAULT_MCP_TOOL_TIMEOUT)
    /// (see issue #1914). Use [`rmcp_tools_with_timeout`](Self::rmcp_tools_with_timeout)
    /// to change or disable it.
    ///
    /// Transitions the builder to the `WithBuilderTools` state.
    #[cfg(feature = "rmcp")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    pub fn rmcp_tools(
        self,
        tools: Vec<rmcp::model::Tool>,
        client: rmcp::service::ServerSink,
    ) -> AgentBuilder<M, P, WithBuilderTools> {
        self.rmcp_tools_with_timeout(tools, client, crate::tool::rmcp::DEFAULT_MCP_TOOL_TIMEOUT)
    }

    /// Add an array of MCP tools (from `rmcp`) with a per-call timeout (see
    /// issue #1914).
    ///
    /// Pass a [`Duration`](std::time::Duration) to bound calls, or `None` to
    /// disable the timeout (unbounded). On timeout a call resolves to a tool
    /// error the agent can recover from instead of blocking forever.
    /// Transitions the builder to the `WithBuilderTools` state.
    #[cfg(feature = "rmcp")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    pub fn rmcp_tools_with_timeout(
        self,
        tools: Vec<rmcp::model::Tool>,
        client: rmcp::service::ServerSink,
        timeout: impl Into<Option<std::time::Duration>>,
    ) -> AgentBuilder<M, P, WithBuilderTools> {
        self.with_rmcp_toolset(build_rmcp_tools(tools, client, timeout.into()))
    }

    /// Transition into the `WithBuilderTools` state carrying the given built
    /// MCP tools.
    #[cfg(feature = "rmcp")]
    fn with_rmcp_toolset(
        self,
        built: Vec<(String, RmcpTool)>,
    ) -> AgentBuilder<M, P, WithBuilderTools> {
        let (static_tools, toolset): (Vec<String>, Vec<RmcpTool>) = built.into_iter().unzip();

        AgentBuilder {
            name: self.name,
            description: self.description,
            model: self.model,
            preamble: self.preamble,
            static_context: self.static_context,
            additional_params: self.additional_params,
            max_tokens: self.max_tokens,
            dynamic_context: self.dynamic_context,
            temperature: self.temperature,
            tool_choice: self.tool_choice,
            default_max_turns: self.default_max_turns,
            hook: self.hook,
            output_schema: self.output_schema,
            output_mode: self.output_mode,
            memory: self.memory,
            default_conversation_id: self.default_conversation_id,
            tool_state: WithBuilderTools {
                static_tools,
                tools: ToolSet::from_tools(toolset),
                dynamic_tools: vec![],
            },
        }
    }

    /// Add some dynamic tools to the agent. On each prompt, `sample` tools from the
    /// dynamic toolset will be inserted in the request.
    ///
    /// Transitions the builder to the `WithBuilderTools` state.
    pub fn dynamic_tools(
        self,
        sample: usize,
        dynamic_tools: impl VectorStoreIndexDyn + Send + Sync + 'static,
        toolset: ToolSet,
    ) -> AgentBuilder<M, P, WithBuilderTools> {
        AgentBuilder {
            name: self.name,
            description: self.description,
            model: self.model,
            preamble: self.preamble,
            static_context: self.static_context,
            additional_params: self.additional_params,
            max_tokens: self.max_tokens,
            dynamic_context: self.dynamic_context,
            temperature: self.temperature,
            tool_choice: self.tool_choice,
            default_max_turns: self.default_max_turns,
            hook: self.hook,
            output_schema: self.output_schema,
            output_mode: self.output_mode,
            memory: self.memory,
            default_conversation_id: self.default_conversation_id,
            tool_state: WithBuilderTools {
                static_tools: vec![],
                tools: toolset,
                dynamic_tools: vec![(sample, Arc::new(dynamic_tools))],
            },
        }
    }

    /// Build the agent with no tools configured.
    ///
    /// An empty `ToolServer` will be created for the agent.
    pub fn build(self) -> Agent<M, P> {
        let tool_server_handle = ToolServer::new().run();

        Agent {
            name: self.name,
            description: self.description,
            model: Arc::new(self.model),
            preamble: self.preamble,
            static_context: self.static_context,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            additional_params: self.additional_params,
            tool_choice: self.tool_choice,
            dynamic_context: Arc::new(self.dynamic_context),
            tool_server_handle,
            default_max_turns: self.default_max_turns,
            hook: self.hook,
            output_schema: self.output_schema,
            output_mode: self.output_mode,
            memory: self.memory,
            default_conversation_id: self.default_conversation_id,
        }
    }
}

impl<M, P> AgentBuilder<M, P, WithToolServerHandle>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    /// Build the agent using the pre-configured ToolServerHandle.
    pub fn build(self) -> Agent<M, P> {
        Agent {
            name: self.name,
            description: self.description,
            model: Arc::new(self.model),
            preamble: self.preamble,
            static_context: self.static_context,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            additional_params: self.additional_params,
            tool_choice: self.tool_choice,
            dynamic_context: Arc::new(self.dynamic_context),
            tool_server_handle: self.tool_state.handle,
            default_max_turns: self.default_max_turns,
            hook: self.hook,
            output_schema: self.output_schema,
            output_mode: self.output_mode,
            memory: self.memory,
            default_conversation_id: self.default_conversation_id,
        }
    }
}

impl<M, P> AgentBuilder<M, P, WithBuilderTools>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    /// Add another static tool to the agent.
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        let toolname = tool.name();
        self.tool_state.tools.add_tool(tool);
        self.tool_state.static_tools.push(toolname);
        self
    }

    /// Add a vector of boxed static tools to the agent.
    pub fn tools(mut self, tools: Vec<Box<dyn ToolDyn>>) -> Self {
        let toolnames: Vec<String> = tools.iter().map(|tool| tool.name()).collect();
        let tools = ToolSet::from_tools_boxed(tools);
        self.tool_state.tools.add_tools(tools);
        self.tool_state.static_tools.extend(toolnames);
        self
    }

    /// Add an array of MCP tools (from `rmcp`) to the agent, each bounded by
    /// [`DEFAULT_MCP_TOOL_TIMEOUT`](crate::tool::rmcp::DEFAULT_MCP_TOOL_TIMEOUT)
    /// (see issue #1914). Use [`rmcp_tools_with_timeout`](Self::rmcp_tools_with_timeout)
    /// to change or disable it.
    #[cfg(feature = "rmcp")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    pub fn rmcp_tools(
        self,
        tools: Vec<rmcp::model::Tool>,
        client: rmcp::service::ServerSink,
    ) -> Self {
        self.rmcp_tools_with_timeout(tools, client, crate::tool::rmcp::DEFAULT_MCP_TOOL_TIMEOUT)
    }

    /// Add an array of MCP tools (from `rmcp`) with a per-call timeout (see
    /// issue #1914).
    ///
    /// Pass a [`Duration`](std::time::Duration) to bound calls, or `None` to
    /// disable the timeout (unbounded). On timeout a call resolves to a tool
    /// error the agent can recover from instead of blocking forever.
    #[cfg(feature = "rmcp")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    pub fn rmcp_tools_with_timeout(
        self,
        tools: Vec<rmcp::model::Tool>,
        client: rmcp::service::ServerSink,
        timeout: impl Into<Option<std::time::Duration>>,
    ) -> Self {
        self.add_rmcp_tools(build_rmcp_tools(tools, client, timeout.into()))
    }

    #[cfg(feature = "rmcp")]
    fn add_rmcp_tools(mut self, built: Vec<(String, RmcpTool)>) -> Self {
        for (name, tool) in built {
            self.tool_state.static_tools.push(name);
            self.tool_state.tools.add_tool(tool);
        }

        self
    }

    /// Add some dynamic tools to the agent. On each prompt, `sample` tools from the
    /// dynamic toolset will be inserted in the request.
    pub fn dynamic_tools(
        mut self,
        sample: usize,
        dynamic_tools: impl VectorStoreIndexDyn + Send + Sync + 'static,
        toolset: ToolSet,
    ) -> Self {
        self.tool_state
            .dynamic_tools
            .push((sample, Arc::new(dynamic_tools)));
        self.tool_state.tools.add_tools(toolset);
        self
    }

    /// Build the agent with the configured tools.
    ///
    /// A new `ToolServer` will be created containing all tools added via
    /// `.tool()`, `.tools()`, `.dynamic_tools()`, etc.
    pub fn build(self) -> Agent<M, P> {
        let tool_server_handle = ToolServer::new()
            .static_tool_names(self.tool_state.static_tools)
            .add_tools(self.tool_state.tools)
            .add_dynamic_tools(self.tool_state.dynamic_tools)
            .run();

        Agent {
            name: self.name,
            description: self.description,
            model: Arc::new(self.model),
            preamble: self.preamble,
            static_context: self.static_context,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            additional_params: self.additional_params,
            tool_choice: self.tool_choice,
            dynamic_context: Arc::new(self.dynamic_context),
            tool_server_handle,
            default_max_turns: self.default_max_turns,
            hook: self.hook,
            output_schema: self.output_schema,
            output_mode: self.output_mode,
            memory: self.memory,
            default_conversation_id: self.default_conversation_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{MockAddTool, MockCompletionModel};

    #[derive(Clone)]
    struct BuilderHook;

    impl PromptHook<MockCompletionModel> for BuilderHook {}

    #[test]
    fn hook_can_be_set_after_tool_configuration() {
        let _agent = AgentBuilder::new(MockCompletionModel::text("ok"))
            .tool(MockAddTool)
            .hook(BuilderHook)
            .build();
    }

    /// The builder's shared MCP helper threads the configured timeout (default,
    /// explicit, or `None`/disabled) onto every built tool, and the threaded
    /// timeout actually bounds a hanging call. This covers the plumbing behind
    /// `rmcp_tool[s]` / `rmcp_tool[s]_with_timeout` (see issue #1914).
    #[cfg(feature = "rmcp")]
    #[tokio::test]
    async fn build_rmcp_tools_threads_timeout_into_built_tools() {
        use crate::tool::ToolDyn;
        use crate::tool::rmcp::DEFAULT_MCP_TOOL_TIMEOUT;
        use rmcp::model::{
            CallToolRequestParams, CallToolResult, ClientInfo, ErrorData, Implementation,
            ProtocolVersion, ServerCapabilities, ServerInfo, Tool,
        };
        use rmcp::service::RequestContext;
        use rmcp::{RoleServer, ServerHandler, ServiceExt};
        use std::sync::Arc;
        use std::time::Duration;

        #[derive(Clone)]
        struct HangingServer;
        impl ServerHandler for HangingServer {
            fn get_info(&self) -> ServerInfo {
                ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
                    .with_protocol_version(ProtocolVersion::LATEST)
                    .with_server_info(Implementation::new("builder-timeout-test", "0.1.0"))
            }
            async fn call_tool(
                &self,
                _request: CallToolRequestParams,
                _context: RequestContext<RoleServer>,
            ) -> Result<CallToolResult, ErrorData> {
                std::future::pending::<Result<CallToolResult, ErrorData>>().await
            }
        }

        fn tool(name: &str) -> Tool {
            Tool::new(
                name.to_string(),
                String::new(),
                Arc::new(serde_json::Map::new()),
            )
        }

        let (c2s, sfc) = tokio::io::duplex(8192);
        let (s2c, cfs) = tokio::io::duplex(8192);
        let server_task = tokio::spawn(async move {
            let running = HangingServer.serve((sfc, s2c)).await.expect("server start");
            running.waiting().await.expect("server error");
        });
        let client = ClientInfo::default()
            .serve((cfs, c2s))
            .await
            .expect("client connect");
        let peer = client.peer().clone();

        // The configured timeout (default, explicit, or disabled) is threaded
        // onto each built tool.
        let built_default = build_rmcp_tools(
            vec![tool("a")],
            peer.clone(),
            Some(DEFAULT_MCP_TOOL_TIMEOUT),
        );
        assert_eq!(built_default[0].1.timeout(), Some(DEFAULT_MCP_TOOL_TIMEOUT));
        let built_none = build_rmcp_tools(vec![tool("b")], peer.clone(), None);
        assert_eq!(built_none[0].1.timeout(), None);

        // ...and the threaded timeout actually bounds a hanging call.
        let built = build_rmcp_tools(
            vec![tool("hang_forever")],
            peer,
            Some(Duration::from_millis(200)),
        );
        assert_eq!(built.len(), 1);
        assert_eq!(built[0].0, "hang_forever");
        let timed =
            tokio::time::timeout(Duration::from_secs(5), built[0].1.call("{}".to_string())).await;
        let err = timed
            .expect("built tool hung past the safety timeout")
            .expect_err("call should time out");
        assert!(err.to_string().contains("timed out"), "got: {err}");

        drop(client);
        server_task.abort();
    }
}
