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
use futures::{Stream, StreamExt};
use std::fmt::{Display, Formatter};
use std::future::Future;
use std::pin::Pin;

/// Enum representing a streaming chunk from the model
#[derive(Debug)]
pub enum StreamingChoice {
    /// A text chunk from a message response
    Message(String),

    /// A tool call response chunk
    ToolCall(String, String, serde_json::Value),
}

impl Display for StreamingChoice {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamingChoice::Message(text) => write!(f, "{}", text),
            StreamingChoice::ToolCall(name, id, params) => {
                write!(f, "Tool call: {} {} {:?}", name, id, params)
            }
        }
    }
}

type StreamingResult = Pin<Box<dyn Stream<Item = Result<StreamingChoice, CompletionError>> + Send>>;

/// Trait for high-level streaming prompt interface
pub trait StreamingPrompt: Send + Sync {
    /// Stream a simple prompt to the model
    fn stream_prompt(
        &self,
        prompt: &str,
    ) -> impl Future<Output = Result<StreamingResult, CompletionError>> + Send;
}

/// Trait for high-level streaming chat interface
pub trait StreamingChat: Send + Sync {
    /// Stream a chat with history to the model
    fn stream_chat(
        &self,
        prompt: &str,
        chat_history: Vec<Message>,
    ) -> impl Future<Output = Result<StreamingResult, CompletionError>> + Send;
}

/// Trait for low-level streaming completion interface
pub trait StreamingCompletion<M: StreamingCompletionModel>: Send + Sync {
    /// Generate a streaming completion from a request
    fn streaming_completion(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<StreamingResult, CompletionError>> + Send;
}

/// Trait defining a streaming completion model
pub trait StreamingCompletionModel: CompletionModel {
    /// Stream a completion response for the given request
    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<StreamingResult, CompletionError>> + Send;
}

/// helper function to stream a completion request to stdout
pub async fn stream_to_stdout(stream: &mut StreamingResult) -> Result<(), std::io::Error> {
    print!("Response: ");
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(chunk) => {
                print!("{}", chunk);
                // Flush stdout to ensure immediate printing
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Err(e) => eprintln!("Error receiving chunk: {}", e),
        }
    }
    println!(); // New line after streaming completes

    Ok(())
}
