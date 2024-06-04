use std::collections::HashMap;

use anyhow::Result;
use futures::{stream, StreamExt};

use crate::{
    completion::{
        Completion, CompletionError, CompletionModel, CompletionRequestBuilder, CompletionResponse,
        Document, Message, ModelChoice, Prompt, PromptError,
    },
    tool::{Tool, ToolSet},
};

/// Struct reprensenting an LLM agent. An agent is an LLM model
/// combined with static context (i.e.: always inserted at the top
/// of the chat history before any use prompts) and static tools.
pub struct Agent<M: CompletionModel> {
    /// Completion model (e.g.: OpenAI's gpt-3.5-turbo-1106, Cohere's command-r)
    model: M,
    /// System prompt
    preamble: String,
    /// Context documents always available to the agent
    context: Vec<Document>,
    /// Tools that are always available to the agent (identified by their name)
    static_tools: Vec<String>,
    /// Temperature of the model
    temperature: Option<f64>,
    /// Additional parameters to be passed to the model
    additional_params: Option<serde_json::Value>,
    /// Actual tool implementations
    tools: ToolSet,
}

impl<M: CompletionModel> Agent<M> {
    /// Create a new Agent
    pub fn new(
        model: M,
        preamble: String,
        static_context: Vec<String>,
        static_tools: Vec<impl Tool + Sync + 'static>,
        temperature: Option<f64>,
        additional_params: Option<serde_json::Value>,
    ) -> Self {
        let static_tools_ids = static_tools.iter().map(|tool| tool.name()).collect();

        Self {
            model,
            preamble,
            context: static_context
                .into_iter()
                .enumerate()
                .map(|(i, doc)| Document {
                    id: format!("static_doc_{}", i),
                    text: doc,
                    additional_props: HashMap::new(),
                })
                .collect(),
            tools: ToolSet::new(static_tools),
            static_tools: static_tools_ids,
            temperature,
            additional_params,
        }
    }
}

impl<M: CompletionModel> Completion<M> for Agent<M> {
    async fn completion(
        &self,
        prompt: &str,
        chat_history: Vec<Message>,
    ) -> Result<CompletionRequestBuilder<M>, CompletionError> {
        let tool_definitions = stream::iter(self.static_tools.iter())
            .filter_map(|toolname| async move {
                if let Some(tool) = self.tools.get(toolname) {
                    Some(tool.definition(prompt.into()).await)
                } else {
                    tracing::error!(target: "ai", "Agent static tool {} not found", toolname);
                    None
                }
            })
            .collect::<Vec<_>>()
            .await;

        Ok(self
            .model
            .completion_request(prompt)
            .preamble(self.preamble.clone())
            .messages(chat_history)
            .documents(self.context.clone())
            .tools(tool_definitions.clone())
            .temperature_opt(self.temperature)
            .additional_params_opt(self.additional_params.clone()))
    }
}

impl<M: CompletionModel> Prompt for Agent<M> {
    async fn prompt(
        &self,
        prompt: &str,
        chat_history: Vec<Message>,
    ) -> Result<String, PromptError> {
        match self.completion(prompt, chat_history).await?.send().await? {
            CompletionResponse {
                choice: ModelChoice::Message(msg),
                ..
            } => Ok(msg),
            CompletionResponse {
                choice: ModelChoice::ToolCall(toolname, args),
                ..
            } => self
                .tools
                .call(&toolname, args.to_string())
                .await
                .map_err(|e| PromptError::ToolCallError(format!("{}", e))),
        }
    }
}

pub struct AgentBuilder<M: CompletionModel> {
    model: M,
    preamble: Option<String>,
    static_context: Vec<Document>,
    static_tools: Vec<String>,
    temperature: Option<f64>,
    additional_params: Option<serde_json::Value>,
    tools: ToolSet,
}

impl<M: CompletionModel> AgentBuilder<M> {
    pub fn new(model: M) -> Self {
        Self {
            model,
            preamble: None,
            static_context: vec![],
            static_tools: vec![],
            temperature: None,
            additional_params: None,
            tools: ToolSet::default(),
        }
    }

    /// Set the preamble of the agent
    pub fn preamble(mut self, doc: &str) -> Self {
        self.preamble = Some(doc.into());
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
    pub fn tool(mut self, tool: impl Tool + Sync + 'static) -> Self {
        let toolname = tool.name();
        self.tools.add_tool(tool);
        self.static_tools.push(toolname);
        self
    }

    /// Set the temperature of the model
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
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
            model: self.model,
            preamble: self.preamble.unwrap_or_else(|| "".into()),
            context: self.static_context,
            tools: self.tools,
            static_tools: self.static_tools,
            temperature: self.temperature,
            additional_params: self.additional_params,
        }
    }
}
