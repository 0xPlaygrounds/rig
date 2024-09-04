//! This module contains the implementation of the [Model] struct and its builder.
//!
//! The [Model] type is the simplest building block for creating an LLM powered application
//! and can be used to prompt completions from a completion model. This struct acts as a
//! thin wrapper around a completion model (i.e.: a struct implementing the
//! [CompletionModel](crate::completion::CompletionModel) trait).
//!  
//! The [ModelBuilder] struct provides a builder interface for creating [Model] instances
//! and allows the user to set the underlying model and other common parameters such as
//! the temperature of the model.
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
//! // Configure the model
//! let model = openai.model("gpt-4o")
//!     .temperature(0.8)
//!     .build();
//!
//! // Use the model for completions and prompts
//! // Generate a chat completion response from a prompt and chat history
//! let chat_response = agent.chat("Prompt", chat_history)
//!     .await
//!     .expect("Failed to chat with model");
//!
//! // Generate a prompt completion response from a simple prompt
//! let chat_response = agent.prompt("Prompt")
//!     .await
//!     .expect("Failed to prompt the model");
//!
//! // Generate a completion request builder from a prompt and chat history. The builder
//! // will contain the model's configuration (i.e.: model parameters, etc.), but these
//! // can be overwritten.
//! let completion_req_builder = agent.completion("Prompt", chat_history)
//!     .await
//!     .expect("Failed to create completion request builder");
//!
//! let response = completion_req_builder
//!     .temperature(0.9) // Overwrite the model's temperature
//!     .send()
//!     .await
//!     .expect("Failed to send completion request");
//! ```
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

    /// Set the temperature of the model (set to `None` to use the default temperature of the model)
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
