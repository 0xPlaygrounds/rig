//! This module contains the implementation of the `Agent` struct and its builder.
//!
//! The `Agent` struct represents an LLM agent, which combines an LLM model with a preamble (system prompt),
//! a set of context documents, and a set of static tools. The agent can be used to interact with the LLM model
//! by providing prompts and chat history.
//!
//! The `AgentBuilder` struct provides a builder pattern for creating instances of the `Agent` struct.
//! It allows configuring the model, preamble, context documents, static tools, temperature, and additional parameters
//! before building the agent.
//!
//! # Example
//! ```rust
//! use rig::{completion::Prompt, providers::openai};
//!
//! let openai_client = openai::Client::from_env();
//!
//! // Configure the model
//! let model = client.model("gpt-4o")
//!     .temperature(0.8)
//!     .build();
//!
//! // Use the model for completions and prompts
//! let completion_req_builder = model.completion("Prompt", chat_history).await;
//! let chat_response = model.chat("Prompt", chat_history).await;
//! ```
//!
//! For more information on how to use the `Agent` struct and its builder, refer to the documentation of the respective structs and methods.
use crate::completion::{
    Chat, Completion, CompletionError, CompletionModel, CompletionRequestBuilder,
    CompletionResponse, Message, ModelChoice, Prompt, PromptError,
};

/// A model that can be used to prompt completions from a completion model.
/// This is the simplest building block for creating an LLM powered application.
pub struct Model<M: CompletionModel> {
    /// Completion model (e.g.: OpenAI's `gpt-3.5-turbo-1106`, Cohere's `command-r`)
    model: M,
    /// Temperature of the model
    temperature: Option<f64>,
}

impl<M: CompletionModel> Completion<M> for Model<M> {
    async fn completion(
        &self,
        prompt: &str,
        chat_history: Vec<Message>,
    ) -> Result<CompletionRequestBuilder<M>, CompletionError> {
        Ok(self
            .model
            .completion_request(prompt)
            .messages(chat_history)
            .temperature_opt(self.temperature))
    }
}

impl<M: CompletionModel> Prompt for Model<M> {
    async fn prompt(&self, prompt: &str) -> Result<String, PromptError> {
        self.chat(prompt, vec![]).await
    }
}

impl<M: CompletionModel> Chat for Model<M> {
    async fn chat(&self, prompt: &str, chat_history: Vec<Message>) -> Result<String, PromptError> {
        match self.completion(prompt, chat_history).await?.send().await? {
            CompletionResponse {
                choice: ModelChoice::Message(message),
                ..
            } => Ok(message),
            CompletionResponse {
                choice: ModelChoice::ToolCall(toolname, _),
                ..
            } => Err(PromptError::ToolError(
                crate::tool::ToolSetError::ToolNotFoundError(toolname),
            )),
        }
    }
}

/// A builder for creating a model
///
/// # Example
/// ```
/// use rig::{providers::openai, model::ModelBuilder};
///
/// let openai_client = openai::Client::from_env();
///
/// let gpt4 = openai_client.completion_model("gpt-4");
///
/// // Configure the model
/// let model = ModelBuilder::new(model)
///     .temperature(0.8)
///     .build();
/// ```
pub struct ModelBuilder<M: CompletionModel> {
    model: M,
    temperature: Option<f64>,
}

impl<M: CompletionModel> ModelBuilder<M> {
    /// Create a new model builder
    pub fn new(model: M) -> Self {
        Self {
            model,
            temperature: None,
        }
    }

    /// Set the temperature of the model
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the temperature of the model (set to None to use the default temperature of the model)
    pub fn temperature_opt(mut self, temperature: Option<f64>) -> Self {
        self.temperature = temperature;
        self
    }

    /// Build the model
    pub fn build(self) -> Model<M> {
        Model {
            model: self.model,
            temperature: self.temperature,
        }
    }
}
