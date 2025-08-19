use std::collections::HashMap;

use crate::{
    completion::{CompletionModel, Document},
    tool::{Tool, ToolSet},
    vector_store::VectorStoreIndexDyn,
};

#[cfg(feature = "rmcp")]
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
pub struct AgentBuilder<M: CompletionModel> {
    /// Name of the agent used for logging and debugging
    name: Option<String>,
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
}

impl<M: CompletionModel> AgentBuilder<M> {
    pub fn new(model: M) -> Self {
        Self {
            name: None,
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
        }
    }

    /// Set the name of the agent
    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the system prompt
    pub fn preamble(mut self, preamble: &str) -> Self {
        self.preamble = Some(preamble.into());
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

    // Add an MCP tool (from `rmcp`) to the agent
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    #[cfg(feature = "rmcp")]
    pub fn rmcp_tool(mut self, tool: rmcp::model::Tool, client: rmcp::service::ServerSink) -> Self {
        let toolname = tool.name.clone();
        self.tools.add_tool(RmcpTool::from_mcp_server(tool, client));
        self.static_tools.push(toolname.to_string());
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
        Agent {
            name: self.name,
            model: self.model,
            preamble: self.preamble.unwrap_or_default(),
            static_context: self.static_context,
            static_tools: self.static_tools,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            additional_params: self.additional_params,
            dynamic_context: self.dynamic_context,
            dynamic_tools: self.dynamic_tools,
            tools: self.tools,
        }
    }
}
