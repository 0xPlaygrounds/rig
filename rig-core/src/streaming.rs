//! This module provides functionality for working with streaming completion models.
//! It provides traits and types for generating streaming completion requests and
//! handling streaming completion responses.
//!
//! The main traits defined in this module are:
//! - [StreamingPrompt]: Defines a high-level streaming LLM one-shot prompt interface
//! - [StreamingChat]: Defines a high-level streaming LLM chat interface with history
//! - [StreamingCompletion]: Defines a low-level streaming LLM completion interface
//!

use crate::OneOrMany;
use crate::agent::Agent;
use crate::agent::prompt_request::streaming::StreamingPromptRequest;
use crate::client::builder::FinalCompletionResponse;
use crate::completion::{
    CompletionError, CompletionModel, CompletionRequestBuilder, CompletionResponse, GetTokenUsage,
    Message, Usage,
};
use crate::message::{AssistantContent, Reasoning, Text, ToolCall, ToolFunction};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use futures::stream::{AbortHandle, Abortable};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::task::{Context, Poll};
use tokio::sync::watch;

/// Control for pausing and resuming a streaming response
pub struct PauseControl {
    pub(crate) paused_tx: watch::Sender<bool>,
    pub(crate) paused_rx: watch::Receiver<bool>,
}

impl PauseControl {
    pub fn new() -> Self {
        let (paused_tx, paused_rx) = watch::channel(false);
        Self {
            paused_tx,
            paused_rx,
        }
    }

    pub fn pause(&self) {
        self.paused_tx.send(true).unwrap();
    }

    pub fn resume(&self) {
        self.paused_tx.send(false).unwrap();
    }

    pub fn is_paused(&self) -> bool {
        *self.paused_rx.borrow()
    }
}

impl Default for PauseControl {
    fn default() -> Self {
        Self::new()
    }
}

/// Enum representing a streaming chunk from the model
#[derive(Debug, Clone)]
pub enum RawStreamingChoice<R>
where
    R: Clone,
{
    /// A text chunk from a message response
    Message(String),

    /// A tool call response chunk
    ToolCall {
        id: String,
        call_id: Option<String>,
        name: String,
        arguments: serde_json::Value,
    },
    /// A reasoning chunk
    Reasoning {
        id: Option<String>,
        reasoning: String,
        signature: Option<String>,
    },

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
pub struct StreamingCompletionResponse<R>
where
    R: Clone + Unpin + GetTokenUsage,
{
    pub(crate) inner: Abortable<StreamingResult<R>>,
    pub(crate) abort_handle: AbortHandle,
    pub(crate) pause_control: PauseControl,
    text: String,
    reasoning: String,
    tool_calls: Vec<ToolCall>,
    /// The final aggregated message from the stream
    /// contains all text and tool calls generated
    pub choice: OneOrMany<AssistantContent>,
    /// The final response from the stream, may be `None`
    /// if the provider didn't yield it during the stream
    pub response: Option<R>,
    pub final_response_yielded: AtomicBool,
}

impl<R> StreamingCompletionResponse<R>
where
    R: Clone + Unpin + GetTokenUsage,
{
    pub fn stream(inner: StreamingResult<R>) -> StreamingCompletionResponse<R> {
        let (abort_handle, abort_registration) = AbortHandle::new_pair();
        let abortable_stream = Abortable::new(inner, abort_registration);
        let pause_control = PauseControl::new();
        Self {
            inner: abortable_stream,
            abort_handle,
            pause_control,
            reasoning: String::new(),
            text: "".to_string(),
            tool_calls: vec![],
            choice: OneOrMany::one(AssistantContent::text("")),
            response: None,
            final_response_yielded: AtomicBool::new(false),
        }
    }

    pub fn cancel(&self) {
        self.abort_handle.abort();
    }

    pub fn pause(&self) {
        self.pause_control.pause();
    }

    pub fn resume(&self) {
        self.pause_control.resume();
    }

    pub fn is_paused(&self) -> bool {
        self.pause_control.is_paused()
    }
}

impl<R> From<StreamingCompletionResponse<R>> for CompletionResponse<Option<R>>
where
    R: Clone + Unpin + GetTokenUsage,
{
    fn from(value: StreamingCompletionResponse<R>) -> CompletionResponse<Option<R>> {
        CompletionResponse {
            choice: value.choice,
            usage: Usage::new(), // Usage is not tracked in streaming responses
            raw_response: value.response,
        }
    }
}

impl<R> Stream for StreamingCompletionResponse<R>
where
    R: Clone + Unpin + GetTokenUsage,
{
    type Item = Result<StreamedAssistantContent<R>, CompletionError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let stream = self.get_mut();

        if stream.is_paused() {
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }

        match Pin::new(&mut stream.inner).poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                // This is run at the end of the inner stream to collect all tokens into
                // a single unified `Message`.
                let mut choice = vec![];

                stream.tool_calls.iter().for_each(|tc| {
                    choice.push(AssistantContent::ToolCall(tc.clone()));
                });

                // This is required to ensure there's always at least one item in the content
                if choice.is_empty() || !stream.text.is_empty() {
                    choice.insert(0, AssistantContent::text(stream.text.clone()));
                }

                stream.choice = OneOrMany::many(choice)
                    .expect("There should be at least one assistant message");

                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(err))) => {
                if matches!(err, CompletionError::ProviderError(ref e) if e.to_string().contains("aborted"))
                {
                    return Poll::Ready(None); // Treat cancellation as stream termination
                }
                Poll::Ready(Some(Err(err)))
            }
            Poll::Ready(Some(Ok(choice))) => match choice {
                RawStreamingChoice::Message(text) => {
                    // Forward the streaming tokens to the outer stream
                    // and concat the text together
                    stream.text = format!("{}{}", stream.text, text.clone());
                    Poll::Ready(Some(Ok(StreamedAssistantContent::text(&text))))
                }
                RawStreamingChoice::Reasoning {
                    id,
                    reasoning,
                    signature,
                } => {
                    // Forward the streaming tokens to the outer stream
                    // and concat the text together
                    stream.reasoning = format!("{}{}", stream.reasoning, reasoning.clone());
                    Poll::Ready(Some(Ok(StreamedAssistantContent::Reasoning(Reasoning {
                        id,
                        reasoning: vec![stream.reasoning.clone()],
                        signature,
                    }))))
                }
                RawStreamingChoice::ToolCall {
                    id,
                    name,
                    arguments,
                    call_id,
                } => {
                    // Keep track of each tool call to aggregate the final message later
                    // and pass it to the outer stream
                    stream.tool_calls.push(ToolCall {
                        id: id.clone(),
                        call_id: call_id.clone(),
                        function: ToolFunction {
                            name: name.clone(),
                            arguments: arguments.clone(),
                        },
                    });
                    if let Some(call_id) = call_id {
                        Poll::Ready(Some(Ok(StreamedAssistantContent::tool_call_with_call_id(
                            id, call_id, name, arguments,
                        ))))
                    } else {
                        Poll::Ready(Some(Ok(StreamedAssistantContent::tool_call(
                            id, name, arguments,
                        ))))
                    }
                }
                RawStreamingChoice::FinalResponse(response) => {
                    if stream
                        .final_response_yielded
                        .load(std::sync::atomic::Ordering::SeqCst)
                    {
                        stream.poll_next_unpin(cx)
                    } else {
                        // Set the final response field and return the next item in the stream
                        stream.response = Some(response.clone());
                        stream
                            .final_response_yielded
                            .store(true, std::sync::atomic::Ordering::SeqCst);
                        let final_response = StreamedAssistantContent::final_response(response);
                        Poll::Ready(Some(Ok(final_response)))
                    }
                }
            },
        }
    }
}

/// Trait for high-level streaming prompt interface
pub trait StreamingPrompt<M, R>
where
    M: CompletionModel + 'static,
    <M as CompletionModel>::StreamingResponse: WasmCompatSend,
    R: Clone + Unpin + GetTokenUsage,
{
    /// Stream a simple prompt to the model
    fn stream_prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> StreamingPromptRequest<M, ()>;
}

/// Trait for high-level streaming chat interface
pub trait StreamingChat<M, R>: WasmCompatSend + WasmCompatSync
where
    M: CompletionModel + 'static,
    <M as CompletionModel>::StreamingResponse: WasmCompatSend,
    R: Clone + Unpin + GetTokenUsage,
{
    /// Stream a chat with history to the model
    fn stream_chat(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: Vec<Message>,
    ) -> StreamingPromptRequest<M, ()>;
}

/// Trait for low-level streaming completion interface
pub trait StreamingCompletion<M: CompletionModel> {
    /// Generate a streaming completion from a request
    fn stream_completion(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: Vec<Message>,
    ) -> impl Future<Output = Result<CompletionRequestBuilder<M>, CompletionError>>;
}

pub(crate) struct StreamingResultDyn<R: Clone + Unpin + GetTokenUsage> {
    pub(crate) inner: StreamingResult<R>,
}

impl<R: Clone + Unpin + GetTokenUsage> Stream for StreamingResultDyn<R> {
    type Item = Result<RawStreamingChoice<FinalCompletionResponse>, CompletionError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let stream = self.get_mut();

        match stream.inner.as_mut().poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Ready(Some(Err(err))) => Poll::Ready(Some(Err(err))),
            Poll::Ready(Some(Ok(chunk))) => match chunk {
                RawStreamingChoice::FinalResponse(res) => Poll::Ready(Some(Ok(
                    RawStreamingChoice::FinalResponse(FinalCompletionResponse {
                        usage: res.token_usage(),
                    }),
                ))),
                RawStreamingChoice::Message(m) => {
                    Poll::Ready(Some(Ok(RawStreamingChoice::Message(m))))
                }
                RawStreamingChoice::Reasoning {
                    id,
                    reasoning,
                    signature,
                } => Poll::Ready(Some(Ok(RawStreamingChoice::Reasoning {
                    id,
                    reasoning,
                    signature,
                }))),
                RawStreamingChoice::ToolCall {
                    id,
                    name,
                    arguments,
                    call_id,
                } => Poll::Ready(Some(Ok(RawStreamingChoice::ToolCall {
                    id,
                    name,
                    arguments,
                    call_id,
                }))),
            },
        }
    }
}

/// helper function to stream a completion request to stdout
pub async fn stream_to_stdout<M>(
    agent: &'static Agent<M>,
    stream: &mut StreamingCompletionResponse<M::StreamingResponse>,
) -> Result<(), std::io::Error>
where
    M: CompletionModel,
{
    let mut is_reasoning = false;
    print!("Response: ");
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(StreamedAssistantContent::Text(text)) => {
                if is_reasoning {
                    is_reasoning = false;
                    println!("\n---\n");
                }
                print!("{}", text.text);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Ok(StreamedAssistantContent::ToolCall(tool_call)) => {
                let res = agent
                    .tool_server_handle
                    .call_tool(
                        &tool_call.function.name,
                        &tool_call.function.arguments.to_string(),
                    )
                    .await
                    .map_err(|x| std::io::Error::other(x.to_string()))?;
                println!("\nResult: {res}");
            }
            Ok(StreamedAssistantContent::Final(res)) => {
                let json_res = serde_json::to_string_pretty(&res).unwrap();
                println!();
                tracing::info!("Final result: {json_res}");
            }
            Ok(StreamedAssistantContent::Reasoning(Reasoning { reasoning, .. })) => {
                if !is_reasoning {
                    is_reasoning = true;
                    println!();
                    println!("Thinking: ");
                }
                let reasoning = reasoning.into_iter().collect::<Vec<String>>().join("");

                print!("{reasoning}");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Err(e) => {
                if e.to_string().contains("aborted") {
                    println!("\nStream cancelled.");
                    break;
                }
                eprintln!("Error: {e}");
                break;
            }
        }
    }

    println!(); // New line after streaming completes

    Ok(())
}

// Test module
#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use async_stream::stream;
    use tokio::time::sleep;

    #[derive(Debug, Clone)]
    pub struct MockResponse {
        #[allow(dead_code)]
        token_count: u32,
    }

    impl GetTokenUsage for MockResponse {
        fn token_usage(&self) -> Option<crate::completion::Usage> {
            let mut usage = Usage::new();
            usage.total_tokens = 15;
            Some(usage)
        }
    }

    fn create_mock_stream() -> StreamingCompletionResponse<MockResponse> {
        let stream = stream! {
            yield Ok(RawStreamingChoice::Message("hello 1".to_string()));
            sleep(Duration::from_millis(100)).await;
            yield Ok(RawStreamingChoice::Message("hello 2".to_string()));
            sleep(Duration::from_millis(100)).await;
            yield Ok(RawStreamingChoice::Message("hello 3".to_string()));
            sleep(Duration::from_millis(100)).await;
            yield Ok(RawStreamingChoice::FinalResponse(MockResponse { token_count: 15 }));
        };

        #[cfg(not(target_arch = "wasm32"))]
        let pinned_stream: StreamingResult<MockResponse> = Box::pin(stream);
        #[cfg(target_arch = "wasm32")]
        let pinned_stream: StreamingResult<MockResponse> = Box::pin(stream);

        StreamingCompletionResponse::stream(pinned_stream)
    }

    #[tokio::test]
    async fn test_stream_cancellation() {
        let mut stream = create_mock_stream();

        println!("Response: ");
        let mut chunk_count = 0;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(StreamedAssistantContent::Text(text)) => {
                    print!("{}", text.text);
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                    chunk_count += 1;
                }
                Ok(StreamedAssistantContent::ToolCall(tc)) => {
                    println!("\nTool Call: {tc:?}");
                    chunk_count += 1;
                }
                Ok(StreamedAssistantContent::Final(res)) => {
                    println!("\nFinal response: {res:?}");
                }
                Ok(StreamedAssistantContent::Reasoning(Reasoning { reasoning, .. })) => {
                    let reasoning = reasoning.into_iter().collect::<Vec<String>>().join("");
                    print!("{reasoning}");
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                }
                Err(e) => {
                    eprintln!("Error: {e:?}");
                    break;
                }
            }

            if chunk_count >= 2 {
                println!("\nCancelling stream...");
                stream.cancel();
                println!("Stream cancelled.");
                break;
            }
        }

        let next_chunk = stream.next().await;
        assert!(
            next_chunk.is_none(),
            "Expected no further chunks after cancellation, got {next_chunk:?}"
        );
    }

    #[tokio::test]
    async fn test_stream_pause_resume() {
        let stream = create_mock_stream();

        // Test pause
        stream.pause();
        assert!(stream.is_paused());

        // Test resume
        stream.resume();
        assert!(!stream.is_paused());
    }
}

/// Describes responses from a streamed provider response which is either text, a tool call or a final usage response.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum StreamedAssistantContent<R> {
    Text(Text),
    ToolCall(ToolCall),
    Reasoning(Reasoning),
    Final(R),
}

impl<R> StreamedAssistantContent<R>
where
    R: Clone + Unpin,
{
    pub fn text(text: &str) -> Self {
        Self::Text(Text {
            text: text.to_string(),
        })
    }

    /// Helper constructor to make creating assistant tool call content easier.
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self::ToolCall(ToolCall {
            id: id.into(),
            call_id: None,
            function: ToolFunction {
                name: name.into(),
                arguments,
            },
        })
    }

    pub fn tool_call_with_call_id(
        id: impl Into<String>,
        call_id: String,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self::ToolCall(ToolCall {
            id: id.into(),
            call_id: Some(call_id),
            function: ToolFunction {
                name: name.into(),
                arguments,
            },
        })
    }

    pub fn final_response(res: R) -> Self {
        Self::Final(res)
    }
}
