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
use crate::message::{AssistantContent, ToolCall, ToolFunction};
use crate::OneOrMany;
use futures::{Stream, StreamExt};
use std::boxed::Box;
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

    /// The final response object, must be yielded if you want the
    /// `response` field to be populated on the `StreamingCompletionResponse`
    FinalResponse(R),
}

#[cfg(not(target_arch = "wasm32"))]
pub type StreamingResult<R> =
    Pin<Box<dyn Stream<Item = Result<RawStreamingChoice<R>, CompletionError>> + Send>>;

#[cfg(target_arch = "wasm32")]
pub type StreamingResult<R> =
    Pin<Box<dyn Stream<Item = Result<RawStreamingChoice<R>, CompletionError>>>>;

/// The response from a streaming completion request;
/// message and response are populated at the end of the
/// `inner` stream.
pub struct StreamingCompletionResponse<R: Clone + Unpin> {
    inner: StreamingResult<R>,
    text: String,
    tool_calls: Vec<ToolCall>,
    /// The final aggregated message from the stream
    /// contains all text and tool calls generated
    pub message: Message,
    /// The final response from the stream, may be `None`
    /// if the provider didn't yield it during the stream
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
                // This is run at the end of the inner stream to collect all tokens into
                // a single unified `Message`.
                let mut content = vec![];

                stream.tool_calls.iter().for_each(|tc| {
                    content.push(AssistantContent::ToolCall(tc.clone()));
                });

                // This is required to ensure there's always at least one item in the content
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
                    // Forward the streaming tokens to the outer stream
                    // and concat the text together
                    stream.text = format!("{}{}", stream.text, text.clone());
                    Poll::Ready(Some(Ok(AssistantContent::text(text))))
                }
                RawStreamingChoice::ToolCall(id, name, args) => {
                    // Keep track of each tool call to aggregate the final message later
                    // and pass it to the outer stream
                    stream.tool_calls.push(ToolCall {
                        id: id.clone(),
                        function: ToolFunction {
                            name: name.clone(),
                            arguments: args.clone(),
                        },
                    });
                    Poll::Ready(Some(Ok(AssistantContent::tool_call(id, name, args))))
                }
                RawStreamingChoice::FinalResponse(response) => {
                    // Set the final response field and return the next item in the stream
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
