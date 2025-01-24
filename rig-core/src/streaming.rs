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

use crate::agent::Agent;
use crate::completion::{CompletionError, CompletionModel, CompletionRequest, Message};
use futures::stream::BoxStream;
use futures::StreamExt;
use std::fmt::{Display, Formatter};
use std::future::Future;

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

type StreamingResult = BoxStream<'static, Result<StreamingChoice, CompletionError>>;

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
pub async fn stream_to_stdout<M: StreamingCompletionModel>(
    agent: Agent<M>,
    stream: &mut StreamingResult,
) -> Result<(), std::io::Error> {
    print!("Response: ");
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(StreamingChoice::Message(text)) => {
                print!("{}", text);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Ok(StreamingChoice::ToolCall(name, _, params)) => {
                let res = agent
                    .tools
                    .call(&name, params.to_string())
                    .await
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                println!("\nResult: {}", res);
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        }
    }
    println!(); // New line after streaming completes

    Ok(())
}
