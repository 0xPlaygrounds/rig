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
#[deprecated(
    since = "0.2.0",
    note = "Please use the `Agent` type directly instead of the `Model` type."
)]
pub type Model<M> = crate::agent::Agent<M>;

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

#[deprecated(
    since = "0.2.0",
    note = "Please use the `AgentBuilder` type directly instead of the `ModelBuilder` type."
)]
pub type ModelBuilder<M> = crate::agent::AgentBuilder<M>;
