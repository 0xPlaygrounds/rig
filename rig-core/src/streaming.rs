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
use crate::completion::{
    CompletionError, CompletionModel, CompletionRequest, CompletionRequestBuilder, Message,
};
use crate::message::ToolCall;
use futures::{Stream, StreamExt};
use std::boxed::Box;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Enum representing a streaming chunk from the model
#[derive(Debug)]
pub enum StreamingChoice {
    /// A text chunk from a message response
    Message(String),

    /// A tool call response chunk
    ToolCall(String, String, serde_json::Value),

    /// A parallel tool call response chunk
    ParToolCall(HashMap<usize, ToolCall>), // index to tool call
}

impl Display for StreamingChoice {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamingChoice::Message(text) => write!(f, "{}", text),
            StreamingChoice::ToolCall(name, id, params) => {
                write!(f, "Tool call: {} {} {:?}", name, id, params)
            }
            StreamingChoice::ParToolCall(tool_calls) => {
                write!(f, "Tool calls: {:?}", tool_calls)
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub type StreamingResult =
    Pin<Box<dyn Stream<Item = Result<StreamingChoice, CompletionError>> + Send>>;

#[cfg(target_arch = "wasm32")]
pub type StreamingResult = Pin<Box<dyn Stream<Item = Result<StreamingChoice, CompletionError>>>>;

/// Trait for high-level streaming prompt interface
pub trait StreamingPrompt: Send + Sync {
    /// Stream a simple prompt to the model
    fn stream_prompt(
        &self,
        prompt: &str,
    ) -> impl Future<Output = Result<StreamingResult, CompletionError>>;
}

/// Trait for high-level streaming chat interface
pub trait StreamingChat: Send + Sync {
    /// Stream a chat with history to the model
    fn stream_chat(
        &self,
        prompt: &str,
        chat_history: Vec<Message>,
    ) -> impl Future<Output = Result<StreamingResult, CompletionError>>;
}

/// Trait for low-level streaming completion interface
pub trait StreamingCompletion<M: StreamingCompletionModel> {
    /// Generate a streaming completion from a request
    fn stream_completion(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> impl Future<Output = Result<CompletionRequestBuilder<M>, CompletionError>>;
}

/// Trait defining a streaming completion model
pub trait StreamingCompletionModel: CompletionModel {
    /// Stream a completion response for the given request
    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<StreamingResult, CompletionError>>;
}

/// helper function to stream a completion request to stdout
pub async fn stream_to_stdout<M: StreamingCompletionModel>(
    agent: Arc<Agent<M>>,
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
            Ok(StreamingChoice::ParToolCall(tool_calls)) => {
                let mut futures = Vec::new();
                for (_index, tool_call) in tool_calls {
                    let name = tool_call.function.name.clone();
                    let params = tool_call.function.arguments;
                    let agent = agent.clone();
                    let future = async move {
                        agent
                            .tools
                            .call(&name, serde_json::to_string(&params).unwrap_or_default())
                            .await
                    };
                    futures.push(future);
                }

                let results = futures::future::try_join_all(futures)
                    .await
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

                for res in results {
                    println!("\nResult: {}", res);
                }
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
