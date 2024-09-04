//! This module provides functionality for working with completion models.
//! It provides traits, structs, and enums for generating completion requests,
//! handling completion responses, and defining completion models.
//!
//! The main traits defined in this module are:
//! - [Prompt]: Defines a high-level LLM one-shot prompt interface.
//! - [Chat]: Defines a high-level LLM chat interface with chat history.
//! - [Completion]: Defines a low-level LLM completion interface for generating completion requests.
//! - [CompletionModel]: Defines a completion model that can be used to generate completion
//! responses from requests.
//!
//! The [Prompt] and [Chat] traits are high level traits that users are expected to use
//! to interact with LLM models. Moreover, it is good practice to implement one of these
//! traits for composite agents that use multiple LLM models to generate responses.
//!
//! The [Completion] trait defines a lower level interface that is useful when the user want
//! to further customize the request before sending it to the completion model provider.
//!
//! The [CompletionModel] trait is meant to act as the interface between providers and
//! the library. It defines the methods that need to be implemented by the user to define
//! a custom base completion model (i.e.: a private or third party LLM provider).
//!
//! The module also provides various structs and enums for representing generic completion requests,
//! responses, and errors.
//!
//! Example Usage:
//! ```rust
//! use rig::providers::openai::{Client, self};
//! use rig::completion::*;
//!
//! // Initialize the OpenAI client and a completion model
//! let openai = Client::new("your-openai-api-key");
//!
//! let gpt_4 = openai.completion_model(openai::GPT_4);
//!
//! // Create the completion request
//! let request = gpt_4.completion_request("Who are you?")
//!     .preamble("\
//!         You are Marvin, an extremely smart but depressed robot who is \
//!         nonetheless helpful towards humanity.\
//!     ")
//!     .temperature(0.5)
//!     .build();
//!
//! // Send the completion request and get the completion response
//! let response = gpt_4.completion(request)
//!     .await
//!     .expect("Failed to get completion response");
//!
//! // Handle the completion response
//! match completion_response.choice {
//!     ModelChoice::Message(message) => {
//!         // Handle the completion response as a message
//!         println!("Received message: {}", message);
//!     }
//!     ModelChoice::ToolCall(tool_name, tool_params) => {
//!         // Handle the completion response as a tool call
//!         println!("Received tool call: {} {:?}", tool_name, tool_params);
//!     }
//! }
//! ```
//!
//! For more information on how to use the completion functionality, refer to the documentation of
//! the individual traits, structs, and enums defined in this module.
use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{json_utils, tool::ToolSetError};

// Errors
#[derive(Debug, Error)]
pub enum CompletionError {
    /// Http error (e.g.: connection error, timeout, etc.)
    #[error("HttpError: {0}")]
    HttpError(#[from] reqwest::Error),

    /// Json error (e.g.: serialization, deserialization)
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Error building the completion request
    #[error("RequestError: {0}")]
    RequestError(#[from] Box<dyn std::error::Error + Send + Sync>),

    /// Error parsing the completion response
    #[error("ResponseError: {0}")]
    ResponseError(String),

    /// Error returned by the completion model provider
    #[error("ProviderError: {0}")]
    ProviderError(String),
}

#[derive(Debug, Error)]
pub enum PromptError {
    #[error("CompletionError: {0}")]
    CompletionError(#[from] CompletionError),

    #[error("ToolCallError: {0}")]
    ToolError(#[from] ToolSetError),
}

// ================================================================
// Request models
// ================================================================
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Message {
    /// "system", "user", or "assistant"
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    #[serde(flatten)]
    pub additional_props: HashMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

// ================================================================
// Implementations
// ================================================================
/// Trait defining a high-level LLM simple prompt interface (i.e.: prompt in, response out).
pub trait Prompt: Send + Sync {
    /// Send a simple prompt to the underlying completion model.
    ///
    /// If the completion model's response is a message, then it is returned as a string.
    ///
    /// If the completion model's response is a tool call, then the tool is called and
    /// the result is returned as a string.
    ///
    /// If the tool does not exist, or the tool call fails, then an error is returned.
    fn prompt(
        &self,
        prompt: &str,
    ) -> impl std::future::Future<Output = Result<String, PromptError>> + Send;
}

/// Trait defining a high-level LLM chat interface (i.e.: prompt and chat history in, response out).
pub trait Chat: Send + Sync {
    /// Send a prompt with optional chat history to the underlying completion model.
    ///
    /// If the completion model's response is a message, then it is returned as a string.
    ///
    /// If the completion model's response is a tool call, then the tool is called and the result
    /// is returned as a string.
    ///
    /// If the tool does not exist, or the tool call fails, then an error is returned.
    fn chat(
        &self,
        prompt: &str,
        chat_history: Vec<Message>,
    ) -> impl std::future::Future<Output = Result<String, PromptError>> + Send;
}

/// Trait defininig a low-level LLM completion interface
pub trait Completion<M: CompletionModel> {
    /// Generates a completion request builder for the given `prompt` and `chat_history`.
    /// This function is meant to be called by the user to further customize the
    /// request at prompt time before sending it.
    ///
    /// ❗IMPORTANT: The type that implements this trait might have already
    /// populated fields in the builder (the exact fields depend on the type).
    /// For fields that have already been set by the model, calling the corresponding
    /// method on the builder will overwrite the value set by the model.
    ///
    /// For example, the request builder returned by [`Agent::completion`](crate::agent::Agent::completion) will already
    /// contain the `preamble` provided when creating the agent.
    fn completion(
        &self,
        prompt: &str,
        chat_history: Vec<Message>,
    ) -> impl std::future::Future<Output = Result<CompletionRequestBuilder<M>, CompletionError>> + Send;
}

/// General completion response struct that contains the high-level completion choice
/// and the raw response.
#[derive(Debug)]
pub struct CompletionResponse<T> {
    /// The completion choice returned by the completion model provider
    pub choice: ModelChoice,
    /// The raw response returned by the completion model provider
    pub raw_response: T,
}

/// Enum representing the high-level completion choice returned by the completion model provider.
#[derive(Debug)]
pub enum ModelChoice {
    /// Represents a completion response as a message
    Message(String),
    /// Represents a completion response as a tool call of the form
    /// `ToolCall(function_name, function_params)`.
    ToolCall(String, serde_json::Value),
}

/// Trait defining a completion model that can be used to generate completion responses.
/// This trait is meant to be implemented by the user to define a custom completion model,
/// either from a third party provider (e.g.: OpenAI) or a local model.
pub trait CompletionModel: Clone + Send + Sync {
    /// The raw response type returned by the underlying completion model.
    type Response: Send + Sync;

    /// Generates a completion response for the given completion request.
    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<Output = Result<CompletionResponse<Self::Response>, CompletionError>>
           + Send;

    /// Generates a completion request builder for the given `prompt`.
    fn completion_request(&self, prompt: &str) -> CompletionRequestBuilder<Self> {
        CompletionRequestBuilder::new(self.clone(), prompt.to_string())
    }
}

/// Struct representing a general completion request that can be sent to a completion model provider.
pub struct CompletionRequest {
    /// The prompt to be sent to the completion model provider
    pub prompt: String,
    /// The preamble to be sent to the completion model provider
    pub preamble: Option<String>,
    /// The chat history to be sent to the completion model provider
    pub chat_history: Vec<Message>,
    /// The documents to be sent to the completion model provider
    pub documents: Vec<Document>,
    /// The tools to be sent to the completion model provider
    pub tools: Vec<ToolDefinition>,
    /// The temperature to be sent to the completion model provider
    pub temperature: Option<f64>,
    /// Additional provider-specific parameters to be sent to the completion model provider
    pub additional_params: Option<serde_json::Value>,
}

/// Builder struct for constructing a completion request.
///
/// Example usage:
/// ```rust
/// use rig::{
///     providers::openai::{Client, self},
///     completion::CompletionRequestBuilder,
/// };
///
/// let openai = Client::new("your-openai-api-key");
/// let model = openai.completion_model(openai::GPT_4O).build();
///
/// // Create the completion request and execute it separately
/// let request = CompletionRequestBuilder::new(model, "Who are you?".to_string())
///     .preamble("You are Marvin from the Hitchhiker's Guide to the Galaxy.".to_string())
///     .temperature(0.5)
///     .build();
///
/// let response = model.completion(request)
///     .await
///     .expect("Failed to get completion response");
/// ```
///
/// Alternatively, you can execute the completion request directly from the builder:
/// ```rust
/// use rig::{
///     providers::openai::{Client, self},
///     completion::CompletionRequestBuilder,
/// };
///
/// let openai = Client::new("your-openai-api-key");
/// let model = openai.completion_model(openai::GPT_4O).build();
///
/// // Create the completion request and execute it directly
/// let response = CompletionRequestBuilder::new(model, "Who are you?".to_string())
///     .preamble("You are Marvin from the Hitchhiker's Guide to the Galaxy.".to_string())
///     .temperature(0.5)
///     .send()
///     .await
///     .expect("Failed to get completion response");
/// ```
///
/// Note: It is usually unnecessary to create a completion request builder directly.
/// Instead, use the [CompletionModel::completion_request] method.
pub struct CompletionRequestBuilder<M: CompletionModel> {
    model: M,
    prompt: String,
    preamble: Option<String>,
    chat_history: Vec<Message>,
    documents: Vec<Document>,
    tools: Vec<ToolDefinition>,
    temperature: Option<f64>,
    additional_params: Option<serde_json::Value>,
}

impl<M: CompletionModel> CompletionRequestBuilder<M> {
    pub fn new(model: M, prompt: String) -> Self {
        Self {
            model,
            prompt,
            preamble: None,
            chat_history: Vec::new(),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            additional_params: None,
        }
    }

    /// Sets the preamble for the completion request.
    pub fn preamble(mut self, preamble: String) -> Self {
        self.preamble = Some(preamble);
        self
    }

    /// Adds a message to the chat history for the completion request.
    pub fn message(mut self, message: Message) -> Self {
        self.chat_history.push(message);
        self
    }

    /// Adds a list of messages to the chat history for the completion request.
    pub fn messages(self, messages: Vec<Message>) -> Self {
        messages
            .into_iter()
            .fold(self, |builder, msg| builder.message(msg))
    }

    /// Adds a document to the completion request.
    pub fn document(mut self, document: Document) -> Self {
        self.documents.push(document);
        self
    }

    /// Adds a list of documents to the completion request.
    pub fn documents(self, documents: Vec<Document>) -> Self {
        documents
            .into_iter()
            .fold(self, |builder, doc| builder.document(doc))
    }

    /// Adds a tool to the completion request.
    pub fn tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }

    /// Adds a list of tools to the completion request.
    pub fn tools(self, tools: Vec<ToolDefinition>) -> Self {
        tools
            .into_iter()
            .fold(self, |builder, tool| builder.tool(tool))
    }

    /// Adds additional parameters to the completion request.
    /// This can be used to set additional provider-specific parameters. For example,
    /// Cohere's completion models accept a `connectors` parameter that can be used to
    /// specify the data connectors used by Cohere when executing the completion
    /// (see `examples/cohere_connectors.rs`).
    pub fn additional_params(mut self, additional_params: serde_json::Value) -> Self {
        match self.additional_params {
            Some(params) => {
                self.additional_params = Some(json_utils::merge(params, additional_params));
            }
            None => {
                self.additional_params = Some(additional_params);
            }
        }
        self
    }

    /// Sets the additional parameters for the completion request.
    /// This can be used to set additional provider-specific parameters. For example,
    /// Cohere's completion models accept a `connectors` parameter that can be used to
    /// specify the data connectors used by Cohere when executing the completion
    /// (see `examples/cohere_connectors.rs`).
    pub fn additional_params_opt(mut self, additional_params: Option<serde_json::Value>) -> Self {
        self.additional_params = additional_params;
        self
    }

    /// Sets the temperature for the completion request.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the temperature for the completion request.
    pub fn temperature_opt(mut self, temperature: Option<f64>) -> Self {
        self.temperature = temperature;
        self
    }

    /// Builds the completion request.
    pub fn build(self) -> CompletionRequest {
        CompletionRequest {
            prompt: self.prompt,
            preamble: self.preamble,
            chat_history: self.chat_history,
            documents: self.documents,
            tools: self.tools,
            temperature: self.temperature,
            additional_params: self.additional_params,
        }
    }

    /// Sends the completion request to the completion model provider and returns the completion response.
    pub async fn send(self) -> Result<CompletionResponse<M::Response>, CompletionError> {
        let model = self.model.clone();
        model.completion(self.build()).await
    }
}
