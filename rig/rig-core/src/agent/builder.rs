use std::{collections::HashMap, sync::Arc};

use tokio::sync::RwLock;

use crate::{
    completion::{CompletionModel, Document},
    message::ToolChoice,
    tool::{
        server::{ToolServer, ToolServerHandle},
        Tool, ToolDyn, ToolSet,
    },
    vector_store::VectorStoreIndexDyn,
};

#[cfg(feature = "rmcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
use crate::tool::rmcp::McpTool as RmcpTool;

use super::Agent;

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
    dynamic_tools: Vec<(usize, Box<dyn VectorStoreIndexDyn + Send + Sync>)>,
}

/// A builder for creating an agent
///
/// The builder uses a typestate pattern to enforce that tool configuration
/// is done in a mutually exclusive way: either provide a pre-existing
/// `ToolServerHandle`, or add tools via the builder API, but not both.
///
/// # Example
/// ```
/// use rig::{providers::openai, agent::AgentBuilder};
///
/// let openai = openai::Client::from_env();
///
/// let gpt4o = openai.completion_model("gpt-4o");
///
/// // Configure the agent
/// let agent = AgentBuilder::new(gpt4o)
///     .preamble("System prompt")
///     .context("Context document 1")
///     .context("Context document 2")
///     .tool(tool1)
///     .tool(tool2)
///     .temperature(0.8)
///     .additional_params(json!({"foo": "bar"}))
///     .build();
/// ```
pub struct AgentBuilder<M, ToolState = NoToolConfig>
where
    M: CompletionModel,
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
    dynamic_context: Vec<(usize, Box<dyn VectorStoreIndexDyn + Send + Sync>)>,
    /// Temperature of the model
    temperature: Option<f64>,
    /// Whether or not the underlying LLM should be forced to use a tool before providing a response.
    tool_choice: Option<ToolChoice>,
    /// Default maximum depth for multi-turn agent calls
    default_max_turns: Option<usize>,
    /// Tool configuration state (typestate pattern)
    tool_state: ToolState,
}

impl<M, ToolState> AgentBuilder<M, ToolState>
where
    M: CompletionModel,
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
            .push((sample, Box::new(dynamic_context)));
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
}

impl<M> AgentBuilder<M, NoToolConfig>
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
        }
    }

    /// Set a pre-existing ToolServerHandle for the agent.
    ///
    /// After calling this method, tool-adding methods (`.tool()`, `.tools()`, etc.)
    /// will not be available. Use this when you want to share a `ToolServer`
    /// between multiple agents or have pre-configured tools.
    pub fn tool_server_handle(
        self,
        handle: ToolServerHandle,
    ) -> AgentBuilder<M, WithToolServerHandle> {
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
        }
    }

    /// Add a static tool to the agent.
    ///
    /// This transitions the builder to the `WithBuilderTools` state, where
    /// additional tools can be added but `tool_server_handle()` is no longer available.
    pub fn tool(self, tool: impl Tool + 'static) -> AgentBuilder<M, WithBuilderTools> {
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
        }
    }

    /// Add a vector of boxed static tools to the agent.
    ///
    /// This is useful when you need to dynamically add static tools to the agent.
    /// Transitions the builder to the `WithBuilderTools` state.
    pub fn tools(self, tools: Vec<Box<dyn ToolDyn>>) -> AgentBuilder<M, WithBuilderTools> {
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
            tool_state: WithBuilderTools {
                static_tools,
                tools,
                dynamic_tools: vec![],
            },
        }
    }

    /// Add an MCP tool (from `rmcp`) to the agent.
    ///
    /// Transitions the builder to the `WithBuilderTools` state.
    #[cfg(feature = "rmcp")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    pub fn rmcp_tool(
        self,
        tool: rmcp::model::Tool,
        client: rmcp::service::ServerSink,
    ) -> AgentBuilder<M, WithBuilderTools> {
        let toolname = tool.name.clone().to_string();
        let tools = ToolSet::from_tools(vec![RmcpTool::from_mcp_server(tool, client)]);

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
                tools,
                dynamic_tools: vec![],
            },
        }
    }

    /// Add an array of MCP tools (from `rmcp`) to the agent.
    ///
    /// Transitions the builder to the `WithBuilderTools` state.
    #[cfg(feature = "rmcp")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    pub fn rmcp_tools(
        self,
        tools: Vec<rmcp::model::Tool>,
        client: rmcp::service::ServerSink,
    ) -> AgentBuilder<M, WithBuilderTools> {
        let (static_tools, tools) = tools.into_iter().fold(
            (Vec::new(), Vec::new()),
            |(mut toolnames, mut toolset), tool| {
                let tool_name = tool.name.to_string();
                let tool = RmcpTool::from_mcp_server(tool, client.clone());
                toolnames.push(tool_name);
                toolset.push(tool);
                (toolnames, toolset)
            },
        );

        let tools = ToolSet::from_tools(tools);

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
                static_tools,
                tools,
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
    ) -> AgentBuilder<M, WithBuilderTools> {
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
                static_tools: vec![],
                tools: toolset,
                dynamic_tools: vec![(sample, Box::new(dynamic_tools))],
            },
        }
    }

    /// Build the agent with no tools configured.
    ///
    /// An empty `ToolServer` will be created for the agent.
    pub fn build(self) -> Agent<M> {
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
            dynamic_context: Arc::new(RwLock::new(self.dynamic_context)),
            tool_server_handle,
            default_max_turns: self.default_max_turns,
        }
    }
}

impl<M> AgentBuilder<M, WithToolServerHandle>
where
    M: CompletionModel,
{
    /// Build the agent using the pre-configured ToolServerHandle.
    pub fn build(self) -> Agent<M> {
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
            dynamic_context: Arc::new(RwLock::new(self.dynamic_context)),
            tool_server_handle: self.tool_state.handle,
            default_max_turns: self.default_max_turns,
        }
    }
}

impl<M> AgentBuilder<M, WithBuilderTools>
where
    M: CompletionModel,
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

    /// Add an array of MCP tools (from `rmcp`) to the agent.
    #[cfg(feature = "rmcp")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    pub fn rmcp_tools(
        mut self,
        tools: Vec<rmcp::model::Tool>,
        client: rmcp::service::ServerSink,
    ) -> Self {
        for tool in tools {
            let tool_name = tool.name.to_string();
            let tool = RmcpTool::from_mcp_server(tool, client.clone());
            self.tool_state.static_tools.push(tool_name);
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
            .push((sample, Box::new(dynamic_tools)));
        self.tool_state.tools.add_tools(toolset);
        self
    }

    /// Build the agent with the configured tools.
    ///
    /// A new `ToolServer` will be created containing all tools added via
    /// `.tool()`, `.tools()`, `.dynamic_tools()`, etc.
    pub fn build(self) -> Agent<M> {
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
            dynamic_context: Arc::new(RwLock::new(self.dynamic_context)),
            tool_server_handle,
            default_max_turns: self.default_max_turns,
        }
    }
}
