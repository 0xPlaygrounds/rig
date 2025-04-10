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
use crate::message::AssistantContent;
use crate::OneOrMany;
use futures::{Stream, StreamExt};
use std::boxed::Box;
use std::fmt::{Display, Formatter};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Enum representing a streaming chunk from the model
#[derive(Debug, Clone)]
pub enum RawStreamingChoice<R: Clone> {
    /// A text chunk from a message response
    Message(String),

    /// A tool call response chunk
    ToolCall(String, String, serde_json::Value),

    /// The final response object
    FinalResponse(R),
}

#[cfg(not(target_arch = "wasm32"))]
pub type StreamingResult<R> =
    Pin<Box<dyn Stream<Item = Result<RawStreamingChoice<R>, CompletionError>> + Send>>;

#[cfg(target_arch = "wasm32")]
pub type StreamingResult<R> =
    Pin<Box<dyn Stream<Item = Result<RawStreamingChoice<R>, CompletionError>>>>;

pub struct StreamingCompletionResponse<R: Clone + Unpin> {
    inner: StreamingResult<R>,
    text: String,
    tool_calls: Vec<(String, String, serde_json::Value)>,
    pub message: Message,
    pub response: Option<R>,
}

impl<R: Clone + Unpin> StreamingCompletionResponse<R> {
    pub fn new(inner: StreamingResult<R>) -> StreamingCompletionResponse<R> {
        Self {
            inner,
            text: "".to_string(),
            tool_calls: vec![],
            message: Message::assistant(""),
            response: None,
        }
    }
}

impl<R: Clone + Unpin> Stream for StreamingCompletionResponse<R> {
    type Item = Result<AssistantContent, CompletionError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let stream = self.get_mut();

        match stream.inner.as_mut().poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                let mut content = vec![];

                stream.tool_calls.iter().for_each(|(n, d, a)| {
                    content.push(AssistantContent::tool_call(n, d, a.clone()));
                });

                if content.is_empty() || !stream.text.is_empty() {
                    content.insert(0, AssistantContent::text(stream.text.clone()));
                }

                stream.message = Message::Assistant {
                    content: OneOrMany::many(content)
                        .expect("There should be at least one assistant message"),
                };

                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(err))) => Poll::Ready(Some(Err(err))),
            Poll::Ready(Some(Ok(choice))) => match choice {
                RawStreamingChoice::Message(text) => {
                    stream.text = format!("{}{}", stream.text, text.clone());
                    Poll::Ready(Some(Ok(AssistantContent::text(text))))
                }
                RawStreamingChoice::ToolCall(id, name, args) => {
                    stream
                        .tool_calls
                        .push((id.clone(), name.clone(), args.clone()));
                    Poll::Ready(Some(Ok(AssistantContent::tool_call(id, name, args))))
                }
                RawStreamingChoice::FinalResponse(response) => {
                    stream.response = Some(response);

                    stream.poll_next_unpin(cx)
                }
            },
        }
    }
}

/// Trait for high-level streaming prompt interface
pub trait StreamingPrompt<R: Clone + Unpin>: Send + Sync {
    /// Stream a simple prompt to the model
    fn stream_prompt(
        &self,
        prompt: &str,
    ) -> impl Future<Output = Result<StreamingCompletionResponse<R>, CompletionError>>;
}

/// Trait for high-level streaming chat interface
pub trait StreamingChat<R: Clone + Unpin>: Send + Sync {
    /// Stream a chat with history to the model
    fn stream_chat(
        &self,
        prompt: &str,
        chat_history: Vec<Message>,
    ) -> impl Future<Output = Result<StreamingCompletionResponse<R>, CompletionError>>;
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
    type StreamingResponse: Clone + Unpin;
    /// Stream a completion response for the given request
    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Future<
        Output = Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>,
    >;
}

/// helper function to stream a completion request to stdout
pub async fn stream_to_stdout<M: StreamingCompletionModel>(
    agent: Agent<M>,
    stream: &mut StreamingCompletionResponse<M::StreamingResponse>,
) -> Result<(), std::io::Error> {
    print!("Response: ");
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(AssistantContent::Text(text)) => {
                print!("{}", text.text);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Ok(AssistantContent::ToolCall(tool_call)) => {
                let res = agent
                    .tools
                    .call(
                        &tool_call.function.name,
                        tool_call.function.arguments.to_string(),
                    )
                    .await
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
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
