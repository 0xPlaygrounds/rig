//! This module contains the implementation of the [Agent] struct and its builder.
//!
//! The [Agent] struct represents an LLM agent, which combines an LLM model with a preamble (system prompt),
//! a set of context documents, and a set of static tools. The agent can be used to interact with the LLM model
//! by providing prompts and chat history without having to provide the preamble and other parameters everytime.
//!
//! The [AgentBuilder] implements the builder pattern for creating instances of [Agent].
//! It allows configuring the model, preamble, context documents, static tools, temperature, and additional parameters
//! before building the agent.
//!
//! # Example
//! ```rust
//! use rig::{
//!     completion::{Chat, Completion, Prompt},
//!     providers::openai,
//! };
//!
//! let openai = openai::Client::from_env();
//!
//! // Configure the agent
//! let agent = openai.agent("gpt-4o")
//!     .preamble("System prompt")
//!     .context("Context document 1")
//!     .context("Context document 2")
//!     .tool(tool1)
//!     .tool(tool2)
//!     .temperature(0.8)
//!     .additional_params(json!({"foo": "bar"}))
//!     .build();
//!
//! // Use the agent for completions and prompts
//! // Generate a chat completion response from a prompt and chat history
//! let chat_response = agent.chat("Prompt", chat_history)
//!     .await
//!     .expect("Failed to chat with Agent");
//! 
//! // Generate a prompt completion response from a simple prompt
//! let chat_response = agent.prompt("Prompt")
//!     .await
//!     .expect("Failed to prompt the Agent");
//! 
//! // Generate a completion request builder from a prompt and chat history. The builder
//! // will contain the agent's configuration (i.e.: preamble, context documents, tools, 
//! // model parameters, etc.), but these can be overwritten.
//! let completion_req_builder = agent.completion("Prompt", chat_history)
//!     .await
//!     .expect("Failed to create completion request builder");
//! 
//! let response = completion_req_builder
//!     .temperature(0.9) // Overwrite the agent's temperature
//!     .send()
//!     .await
//!     .expect("Failed to send completion request");
//! ```
use std::collections::HashMap;

use futures::{stream, StreamExt};

use crate::{
    completion::{
        Chat, Completion, CompletionError, CompletionModel, CompletionRequestBuilder,
        CompletionResponse, Document, Message, ModelChoice, Prompt, PromptError,
    },
    tool::{Tool, ToolSet},
};

/// Struct reprensenting an LLM agent. An agent is an LLM model combined with a preamble
/// (i.e.: system prompt) and a static set of context documents and tools.
/// All context documents and tools are always provided to the agent when prompted.
///
/// # Example
/// ```
/// use rig::{completion::Prompt, providers::openai};
///
/// let openai = openai::Client::from_env();
///
/// let comedian_agent = openai
///     .agent("gpt-4o")
///     .preamble("You are a comedian here to entertain the user using humour and jokes.")
///     .temperature(0.9)
///     .build();
///
/// let response = comedian_agent.prompt("Entertain me!")
///     .await
///     .expect("Failed to prompt the agent");
/// ```
pub struct Agent<M: CompletionModel> {
    /// Completion model (e.g.: OpenAI's `gpt-3.5-turbo-1106`, Cohere's `command-r`)
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
        static_tools: Vec<impl Tool + 'static>,
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
            tools: ToolSet::from_tools(static_tools),
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
                    tracing::error!(target: "rig", "Agent static tool {} not found", toolname);
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
    async fn prompt(&self, prompt: &str) -> Result<String, PromptError> {
        self.chat(prompt, vec![]).await
    }
}

impl<M: CompletionModel> Chat for Agent<M> {
    async fn chat(&self, prompt: &str, chat_history: Vec<Message>) -> Result<String, PromptError> {
        match self.completion(prompt, chat_history).await?.send().await? {
            CompletionResponse {
                choice: ModelChoice::Message(msg),
                ..
            } => Ok(msg),
            CompletionResponse {
                choice: ModelChoice::ToolCall(toolname, args),
                ..
            } => Ok(self.tools.call(&toolname, args.to_string()).await?),
        }
    }
}

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
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
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
