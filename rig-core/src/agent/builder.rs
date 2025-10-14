use std::{collections::HashMap, sync::Arc};

use tokio::sync::RwLock;

use crate::{
    completion::{CompletionModel, Document},
    message::ToolChoice,
    tool::{
        Tool, ToolSet,
        server::{ToolServer, ToolServerHandle},
    },
    vector_store::VectorStoreIndexDyn,
};

#[cfg(feature = "rmcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
use crate::tool::rmcp::McpTool as RmcpTool;

use super::Agent;

/// A builder for creating an agent
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
/// let agent = AgentBuilder::new(model)
///     .preamble("System prompt")
///     .context("Context document 1")
///     .context("Context document 2")
///     .tool(tool1)
///     .tool(tool2)
///     .temperature(0.8)
///     .additional_params(json!({"foo": "bar"}))
///     .build();
/// ```
pub struct AgentBuilder<M>
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
    dynamic_context: Vec<(usize, Box<dyn VectorStoreIndexDyn>)>,
    /// Temperature of the model
    temperature: Option<f64>,
    /// Tool server handle
    tool_server_handle: Option<ToolServerHandle>,
    /// Whether or not the underlying LLM should be forced to use a tool before providing a response.
    tool_choice: Option<ToolChoice>,
}

impl<M> AgentBuilder<M>
where
    M: CompletionModel,
{
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
            tool_server_handle: None,
            tool_choice: None,
        }
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
        self.preamble = Some(format!(
            "{}\n{}",
            self.preamble.unwrap_or_else(|| "".into()),
            doc
        ));
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

    /// Add a static tool to the agent
    pub fn tool(self, tool: impl Tool + 'static) -> AgentBuilderSimple<M> {
        let toolname = tool.name();
        let tools = ToolSet::from_tools(vec![tool]);
        let static_tools = vec![toolname];

        AgentBuilderSimple {
            name: self.name,
            description: self.description,
            model: self.model,
            preamble: self.preamble,
            static_context: self.static_context,
            static_tools,
            additional_params: self.additional_params,
            max_tokens: self.max_tokens,
            dynamic_context: vec![],
            dynamic_tools: vec![],
            temperature: self.temperature,
            tools,
            tool_choice: self.tool_choice,
        }
    }

    pub fn tool_server_handle(mut self, handle: ToolServerHandle) -> Self {
        self.tool_server_handle = Some(handle);
        self
    }

    /// Add an MCP tool (from `rmcp`) to the agent
    #[cfg(feature = "rmcp")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    pub fn rmcp_tool(
        self,
        tool: rmcp::model::Tool,
        client: rmcp::service::ServerSink,
    ) -> AgentBuilderSimple<M> {
        let toolname = tool.name.clone().to_string();
        let tools = ToolSet::from_tools(vec![RmcpTool::from_mcp_server(tool, client)]);
        let static_tools = vec![toolname];

        AgentBuilderSimple {
            name: self.name,
            description: self.description,
            model: self.model,
            preamble: self.preamble,
            static_context: self.static_context,
            static_tools,
            additional_params: self.additional_params,
            max_tokens: self.max_tokens,
            dynamic_context: vec![],
            dynamic_tools: vec![],
            temperature: self.temperature,
            tools,
            tool_choice: self.tool_choice,
        }
    }

    /// Add an array of MCP tools (from `rmcp`) to the agent
    #[cfg(feature = "rmcp")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    pub fn rmcp_tools(
        self,
        tools: Vec<rmcp::model::Tool>,
        client: rmcp::service::ServerSink,
    ) -> AgentBuilderSimple<M> {
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

        AgentBuilderSimple {
            name: self.name,
            description: self.description,
            model: self.model,
            preamble: self.preamble,
            static_context: self.static_context,
            static_tools,
            additional_params: self.additional_params,
            max_tokens: self.max_tokens,
            dynamic_context: vec![],
            dynamic_tools: vec![],
            temperature: self.temperature,
            tools,
            tool_choice: self.tool_choice,
        }
    }

    /// Add some dynamic context to the agent. On each prompt, `sample` documents from the
    /// dynamic context will be inserted in the request.
    pub fn dynamic_context(
        mut self,
        sample: usize,
        dynamic_context: impl VectorStoreIndexDyn + 'static,
    ) -> Self {
        self.dynamic_context
            .push((sample, Box::new(dynamic_context)));
        self
    }

    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Add some dynamic tools to the agent. On each prompt, `sample` tools from the
    /// dynamic toolset will be inserted in the request.
    pub fn dynamic_tools(
        self,
        sample: usize,
        dynamic_tools: impl VectorStoreIndexDyn + 'static,
        toolset: ToolSet,
    ) -> AgentBuilderSimple<M> {
        let thing: Box<dyn VectorStoreIndexDyn + 'static> = Box::new(dynamic_tools);
        let dynamic_tools = vec![(sample, thing)];

        AgentBuilderSimple {
            name: self.name,
            description: self.description,
            model: self.model,
            preamble: self.preamble,
            static_context: self.static_context,
            static_tools: vec![],
            additional_params: self.additional_params,
            max_tokens: self.max_tokens,
            dynamic_context: vec![],
            dynamic_tools,
            temperature: self.temperature,
            tools: toolset,
            tool_choice: self.tool_choice,
        }
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

    /// Build the agent
    pub fn build(self) -> Agent<M> {
        let tool_server_handle = if let Some(handle) = self.tool_server_handle {
            handle
        } else {
            ToolServer::new().run()
        };

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
        }
    }
}

/// A fluent builder variation of `AgentBuilder`. Allows adding tools directly to the builder rather than using the tool server handle.
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
/// let agent = AgentBuilder::new(model)
///     .preamble("System prompt")
///     .context("Context document 1")
///     .context("Context document 2")
///     .tool(tool1)
///     .tool(tool2)
///     .temperature(0.8)
///     .additional_params(json!({"foo": "bar"}))
///     .build();
/// ```
pub struct AgentBuilderSimple<M>
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
    /// Tools that are always available to the agent (by name)
    static_tools: Vec<String>,
    /// Additional parameters to be passed to the model
    additional_params: Option<serde_json::Value>,
    /// Maximum number of tokens for the completion
    max_tokens: Option<u64>,
    /// List of vector store, with the sample number
    dynamic_context: Vec<(usize, Box<dyn VectorStoreIndexDyn>)>,
    /// Dynamic tools
    dynamic_tools: Vec<(usize, Box<dyn VectorStoreIndexDyn>)>,
    /// Temperature of the model
    temperature: Option<f64>,
    /// Actual tool implementations
    tools: ToolSet,
    /// Whether or not the underlying LLM should be forced to use a tool before providing a response.
    tool_choice: Option<ToolChoice>,
}

impl<M> AgentBuilderSimple<M>
where
    M: CompletionModel,
{
    pub fn new(model: M) -> Self {
        Self {
            name: None,
            description: None,
            model,
            preamble: None,
            static_context: vec![],
            static_tools: vec![],
            temperature: None,
            max_tokens: None,
            additional_params: None,
            dynamic_context: vec![],
            dynamic_tools: vec![],
            tools: ToolSet::default(),
            tool_choice: None,
        }
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
        self.preamble = Some(format!(
            "{}\n{}",
            self.preamble.unwrap_or_else(|| "".into()),
            doc
        ));
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

    /// Add a static tool to the agent
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        let toolname = tool.name();
        self.tools.add_tool(tool);
        self.static_tools.push(toolname);
        self
    }

    /// Add an array of MCP tools (from `rmcp`) to the agent
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
            self.static_tools.push(tool_name);
            self.tools.add_tool(tool);
        }

        self
    }

    /// Add some dynamic context to the agent. On each prompt, `sample` documents from the
    /// dynamic context will be inserted in the request.
    pub fn dynamic_context(
        mut self,
        sample: usize,
        dynamic_context: impl VectorStoreIndexDyn + 'static,
    ) -> Self {
        self.dynamic_context
            .push((sample, Box::new(dynamic_context)));
        self
    }

    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Add some dynamic tools to the agent. On each prompt, `sample` tools from the
    /// dynamic toolset will be inserted in the request.
    pub fn dynamic_tools(
        mut self,
        sample: usize,
        dynamic_tools: impl VectorStoreIndexDyn + 'static,
        toolset: ToolSet,
    ) -> Self {
        self.dynamic_tools.push((sample, Box::new(dynamic_tools)));
        self.tools.add_tools(toolset);
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

    /// Build the agent
    pub fn build(self) -> Agent<M> {
        let tool_server_handle = ToolServer::new()
            .static_tool_names(self.static_tools)
            .add_tools(self.tools)
            .add_dynamic_tools(self.dynamic_tools)
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
        }
    }
}
