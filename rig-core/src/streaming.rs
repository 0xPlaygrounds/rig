//! This module provides functionality for working with streaming completion models.
//! It provides traits and types for generating streaming completion requests and
//! handling streaming completion responses.
//!
//! The main traits defined in this module are:
//! - [StreamingPrompt]: Defines a high-level streaming LLM one-shot prompt interface
//! - [StreamingChat]: Defines a high-level streaming LLM chat interface with history
//! - [StreamingCompletion]: Defines a low-level streaming LLM completion interface
//! - [StreamingCompletionModel]: Defines a streaming completion model interface
//!
//! Example Usage:
//! ```rust
//! use rig::providers::openai::{Client, self};
//! use rig::streaming::*;
//!
//! let openai = Client::new("your-openai-api-key");
//! let gpt_4 = openai.streaming_completion_model(openai::GPT_4);
//!
//! let mut stream = gpt_4.stream_prompt("Tell me a story")
//!     .await
//!     .expect("Failed to create stream");
//!
//! while let Some(chunk) = stream.next().await {
//!     match chunk {
//!         Ok(StreamingChoice::Message(text)) => {
//!             print!("{}", text);
//!         }
//!         Ok(StreamingChoice::ToolCall(name, id, params)) => {
//!             println!("Tool call: {} {} {:?}", name, id, params);
//!         }
//!         Err(e) => {
//!             eprintln!("Error: {}", e);
//!             break;
//!         }
//!     }
//! }
//! ```

use crate::completion::{CompletionError, CompletionModel, CompletionRequest, Message};
use futures::Stream;
use std::pin::Pin;

/// Enum representing a streaming chunk from the model
#[derive(Debug)]
pub enum StreamingChoice {
    /// A text chunk from a message response
    Message(String),

    /// A tool call response chunk
    ToolCall(String, String, serde_json::Value),
}

type StreamingResult = Pin<Box<dyn Stream<Item = Result<StreamingChoice, CompletionError>> + Send>>;

/// Trait for high-level streaming prompt interface
pub trait StreamingPrompt: Send + Sync {
    /// Stream a simple prompt to the model
    fn stream_prompt(
        &self,
        prompt: &str,
    ) -> impl std::future::Future<Output = Result<StreamingResult, CompletionError>> + Send;
}

/// Trait for high-level streaming chat interface
pub trait StreamingChat: Send + Sync {
    /// Stream a chat with history to the model
    fn stream_chat(
        &self,
        prompt: &str,
        chat_history: Vec<Message>,
    ) -> impl std::future::Future<Output = Result<StreamingResult, CompletionError>> + Send;
}

/// Trait for low-level streaming completion interface
pub trait StreamingCompletion<M: StreamingCompletionModel>: Send + Sync {
    /// Generate a streaming completion from a request
    fn streaming_completion(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<Output = Result<StreamingResult, CompletionError>> + Send;
}

/// Trait defining a streaming completion model
pub trait StreamingCompletionModel: CompletionModel {
    /// Stream a completion response for the given request
    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<Output = Result<StreamingResult, CompletionError>> + Send;
}
