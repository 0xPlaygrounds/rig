//! This module provides the concrete types for normalizing and consuming live
//! completion streams.
//!
//! Provider implementations use these types to expose raw streamed completion
//! events without depending on a runtime.

use crate::OneOrMany;
use crate::completion::{CompletionError, CompletionResponse, Usage};
use crate::message::{
    AssistantContent, Reasoning, ReasoningContent, Text, ToolCall, ToolFunction, ToolResult,
};
use crate::wasm_compat::WasmCompatSend;
use futures::stream::{AbortHandle, Abortable};
use futures::task::AtomicWaker;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};
use thiserror::Error;

/// Internal control for pausing and resuming a live completion stream.
struct PauseControl {
    paused: AtomicBool,
    waker: AtomicWaker,
}

/// Cloneable control handle for pausing and resuming a completion stream.
///
/// The handle can wake a task currently awaiting [`CompletionStream::next`],
/// so callers do not need to cancel and recreate the pending read in order to
/// resume delivery.
#[derive(Clone)]
pub struct CompletionStreamPauseHandle {
    control: Arc<PauseControl>,
}

impl CompletionStreamPauseHandle {
    /// Pause provider-source polling.
    pub fn pause(&self) {
        self.control.pause();
    }

    /// Resume provider-source polling and wake the parked consumer.
    pub fn resume(&self) {
        self.control.resume();
    }

    /// Return whether provider-source polling is paused.
    pub fn is_paused(&self) -> bool {
        self.control.is_paused()
    }
}

impl PauseControl {
    /// Create a pause controller in the running state.
    fn new() -> Self {
        Self {
            paused: AtomicBool::new(false),
            waker: AtomicWaker::new(),
        }
    }

    /// Pause polling of the public stream until [`PauseControl::resume`] is called.
    fn pause(&self) {
        self.paused.store(true, Ordering::Release);
    }

    /// Resume polling after a pause.
    fn resume(&self) {
        if self.paused.swap(false, Ordering::AcqRel) {
            self.waker.wake();
        }
    }

    /// Returns whether the stream is currently paused.
    fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Acquire)
    }

    /// Register the current task while paused without racing a concurrent resume.
    fn poll_ready(&self, cx: &mut Context<'_>) -> Poll<()> {
        if !self.is_paused() {
            return Poll::Ready(());
        }

        self.waker.register(cx.waker());
        if self.is_paused() {
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    }

    /// Wake a task parked by the pause gate when another terminal transition occurs.
    fn wake(&self) {
        self.waker.wake();
    }
}

impl Default for PauseControl {
    fn default() -> Self {
        Self::new()
    }
}

/// The content of a tool call delta - either the tool name or argument data
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum ToolCallDeltaContent {
    /// Tool/function name emitted by the provider.
    Name(String),
    /// Partial JSON argument data emitted by the provider.
    Delta(String),
}

/// Discriminant for [`StreamFinal`], required so the `#[serde(untagged)]`
/// [`StreamedAssistantContent`] enum can distinguish a final record from an
/// [`StreamedAssistantContent::Unknown`] payload on deserialize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamFinalKind {
    /// The provider's terminal stream event.
    Final,
}

/// The provider's terminal stream record, normalized.
///
/// Replaces the provider-typed `FinalResponse(R)` payload: usage becomes a
/// field (deleting the `GetTokenUsage` trait), and the finish reason is
/// normalized exactly as on the unary [`CompletionResponse`]
/// (`crate::completion::FinishReason`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StreamFinal {
    /// Discriminating field; always [`StreamFinalKind::Final`].
    pub kind: StreamFinalKind,
    /// Token usage reported by the provider for this streamed completion.
    /// Zero-valued usage is the documented sentinel for missing metrics.
    pub usage: Usage,
    /// Why the model stopped generating, when the provider reported it.
    pub finish_reason: Option<crate::completion::FinishReason>,
    /// Provider-assigned message ID, when available.
    pub message_id: Option<String>,
    /// Name of the provider that produced this stream (descriptor name).
    pub provider: String,
    /// Provider-reported model identifier, when available.
    pub model: Option<String>,
}

impl StreamFinal {
    /// Create a terminal record for `provider` with `usage`; optional
    /// metadata starts unset and is filled with the `with_*` helpers.
    pub fn new(provider: impl Into<String>, usage: Usage) -> Self {
        Self {
            kind: StreamFinalKind::Final,
            usage,
            finish_reason: None,
            message_id: None,
            provider: provider.into(),
            model: None,
        }
    }

    /// Attach the normalized finish reason.
    pub fn with_finish_reason(mut self, finish_reason: crate::completion::FinishReason) -> Self {
        self.finish_reason = Some(finish_reason);
        self
    }

    /// Attach the provider-assigned message ID.
    pub fn with_message_id(mut self, message_id: impl Into<String>) -> Self {
        self.message_id = Some(message_id.into());
        self
    }

    /// Attach the provider-reported model identifier.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

/// Enum representing a streaming chunk from the model
#[derive(Debug, Clone)]
pub enum RawStreamingChoice {
    /// A text chunk from a message response
    Message(String),

    /// Start a new text content block in the accumulated final choice.
    ///
    /// This is an internal provider-normalization event. It is not yielded to
    /// public stream consumers, but lets providers preserve metadata boundaries
    /// for final aggregated assistant text blocks.
    TextStart {
        /// Provider-specific metadata attached to this text block.
        additional_params: Option<serde_json::Value>,
    },

    /// Provider-specific metadata for the current text content block.
    ///
    /// This is not yielded to public stream consumers. The metadata is merged
    /// into the current aggregated [`Text`] block.
    TextAdditionalParams(serde_json::Value),

    /// A tool call response (in its entirety)
    ToolCall(RawStreamingToolCall),
    /// A tool call partial/delta
    ToolCallDelta {
        /// Provider-supplied tool call ID.
        id: String,
        /// Rig-generated unique identifier for this tool call.
        internal_call_id: String,
        content: ToolCallDeltaContent,
    },
    /// A reasoning (in its entirety)
    Reasoning {
        /// Provider-supplied reasoning block ID, when present.
        id: Option<String>,
        /// Complete reasoning content block.
        content: ReasoningContent,
    },
    /// A reasoning partial/delta
    ReasoningDelta {
        /// Provider-supplied reasoning block ID, when present.
        id: Option<String>,
        /// Partial reasoning text.
        reasoning: String,
    },

    /// The provider's normalized terminal record; must be yielded if you want
    /// [`CompletionStream::final_record`] to return a value.
    FinalResponse(StreamFinal),

    /// Provider-assigned message ID (e.g. OpenAI Responses API `msg_` ID).
    /// Captured silently and exposed through [`CompletionStream::message_id`].
    MessageId(String),

    /// A provider-native output item this version does not model — e.g. an
    /// OpenAI Responses hosted-tool result (`web_search_call`, `file_search_call`,
    /// `computer_call`, `code_interpreter_call`). Carries the raw item object
    /// verbatim. Forwarded to the stream consumer as
    /// [`StreamedAssistantContent::Unknown`] but not folded into the accumulated
    /// assistant message (there is no `AssistantContent::Unknown` history slot).
    Unknown(serde_json::Value),
}

/// Describes a streaming tool call response (in its entirety)
#[derive(Debug, Clone)]
pub struct RawStreamingToolCall {
    /// Provider-supplied tool call ID.
    pub id: String,
    /// Rig-generated unique identifier for this tool call.
    pub internal_call_id: String,
    /// Provider-specific call ID used by some APIs for tool result correlation.
    pub call_id: Option<String>,
    /// Tool/function name.
    pub name: String,
    /// Parsed tool arguments.
    pub arguments: serde_json::Value,
    /// Optional provider signature associated with the tool call.
    pub signature: Option<String>,
    /// Additional provider-specific tool call metadata.
    pub additional_params: Option<serde_json::Value>,
}

impl RawStreamingToolCall {
    /// Create an empty tool call accumulator for provider streaming parsers.
    pub fn empty() -> Self {
        Self {
            id: String::new(),
            internal_call_id: crate::id::generate(),
            call_id: None,
            name: String::new(),
            arguments: serde_json::Value::Null,
            signature: None,
            additional_params: None,
        }
    }

    /// Create a complete tool call with a generated internal call ID.
    pub fn new(id: String, name: String, arguments: serde_json::Value) -> Self {
        Self {
            id,
            internal_call_id: crate::id::generate(),
            call_id: None,
            name,
            arguments,
            signature: None,
            additional_params: None,
        }
    }

    /// Override the generated internal call ID.
    pub fn with_internal_call_id(mut self, internal_call_id: String) -> Self {
        self.internal_call_id = internal_call_id;
        self
    }

    /// Attach a provider-specific call ID.
    pub fn with_call_id(mut self, call_id: String) -> Self {
        self.call_id = Some(call_id);
        self
    }

    /// Attach or clear a provider signature.
    pub fn with_signature(mut self, signature: Option<String>) -> Self {
        self.signature = signature;
        self
    }

    /// Attach provider-specific metadata.
    pub fn with_additional_params(mut self, additional_params: Option<serde_json::Value>) -> Self {
        self.additional_params = additional_params;
        self
    }
}

impl From<RawStreamingToolCall> for ToolCall {
    fn from(tool_call: RawStreamingToolCall) -> Self {
        ToolCall {
            id: tool_call.id,
            call_id: tool_call.call_id,
            function: ToolFunction {
                name: tool_call.name,
                arguments: tool_call.arguments,
            },
            signature: tool_call.signature,
            additional_params: tool_call.additional_params,
        }
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
type ErasedRawStream =
    Pin<Box<dyn Stream<Item = Result<RawStreamingChoice, CompletionError>> + Send + 'static>>;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
type ErasedRawStream =
    Pin<Box<dyn Stream<Item = Result<RawStreamingChoice, CompletionError>> + 'static>>;

/// Why a [`CompletionStream`] stopped producing items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionStreamTermination {
    /// The provider source reached natural end-of-stream and aggregation completed.
    Exhausted,
    /// The caller explicitly cancelled the stream.
    Cancelled,
    /// The provider source yielded an error.
    Failed,
}

/// Error returned when a live or incomplete stream is finalized as a response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CompletionStreamFinalizationError {
    /// The provider source has not reached a terminal state.
    #[error("completion stream is still running")]
    Running,
    /// A cancelled stream contains only a provisional partial aggregate.
    #[error("cancelled completion stream cannot be finalized as a response")]
    Cancelled,
    /// A failed stream contains only a provisional partial aggregate.
    #[error("failed completion stream cannot be finalized as a response")]
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompletionStreamState {
    Running,
    Exhausted,
    Cancelled,
    Failed,
}

/// A live normalized completion stream.
///
/// The provider-specific source is pinned and erased privately while this
/// concrete handle aggregates its normalized [`AssistantContent`] and terminal
/// [`StreamFinal`]. Use [`CompletionStream::into_response`] after natural
/// exhaustion to obtain owned finished response data. Cancelled, failed, and
/// still-running streams cannot be finalized as successful responses.
pub struct CompletionStream {
    inner: Abortable<ErasedRawStream>,
    abort_handle: AbortHandle,
    pause_control: Arc<PauseControl>,
    state: CompletionStreamState,
    assistant_items: Vec<AssistantContent>,
    text_item_index: Option<usize>,
    reasoning_item_index: Option<usize>,
    choice: OneOrMany<AssistantContent>,
    final_record: Option<StreamFinal>,
    final_record_yielded: bool,
    message_id: Option<String>,
}

impl CompletionStream {
    /// Wrap an owned provider stream and initialize normalized aggregation.
    ///
    /// The handle owns the stream's pinning and type erasure. Provider
    /// implementations can pass their concrete stream directly; on browser
    /// wasm that stream may remain worker-local and non-`Send`.
    ///
    /// The `'static` bound means the returned handle owns its source and does
    /// not borrow stack data. It does not require the stream to live forever.
    pub fn from_stream(
        inner: impl Stream<Item = Result<RawStreamingChoice, CompletionError>>
        + WasmCompatSend
        + 'static,
    ) -> Self {
        let (abort_handle, abort_registration) = AbortHandle::new_pair();
        let inner: ErasedRawStream = Box::pin(inner);
        let abortable_stream = Abortable::new(inner, abort_registration);
        let pause_control = Arc::new(PauseControl::new());
        Self {
            inner: abortable_stream,
            abort_handle,
            pause_control,
            state: CompletionStreamState::Running,
            assistant_items: vec![],
            text_item_index: None,
            reasoning_item_index: None,
            choice: OneOrMany::one(AssistantContent::text("")),
            final_record: None,
            final_record_yielded: false,
            message_id: None,
        }
    }

    /// Pull the next normalized item without requiring [`StreamExt`] or caller
    /// pinning.
    pub async fn next(&mut self) -> Option<Result<StreamedAssistantContent, CompletionError>> {
        StreamExt::next(self).await
    }

    /// Cancel the stream and immediately drop the provider's inner stream.
    /// Cancellation is surfaced as normal stream termination.
    pub fn cancel(&mut self) {
        if self.state != CompletionStreamState::Running {
            return;
        }

        self.replace_source_with_empty();
        self.state = CompletionStreamState::Cancelled;
        self.pause_control.wake();
    }

    fn replace_source_with_empty(&mut self) {
        self.abort_handle.abort();
        let (abort_handle, abort_registration) = AbortHandle::new_pair();
        let empty: ErasedRawStream = Box::pin(futures::stream::empty::<
            Result<RawStreamingChoice, CompletionError>,
        >());
        self.inner = Abortable::new(empty, abort_registration);
        self.abort_handle = abort_handle;
    }

    /// Pause stream polling.
    pub fn pause(&self) {
        self.pause_control.pause();
    }

    /// Resume stream polling after a pause.
    pub fn resume(&self) {
        self.pause_control.resume();
    }

    /// Returns whether the stream is currently paused.
    pub fn is_paused(&self) -> bool {
        self.pause_control.is_paused()
    }

    /// Return a cloneable handle that can resume a task currently parked in
    /// [`Self::next`].
    pub fn pause_handle(&self) -> CompletionStreamPauseHandle {
        CompletionStreamPauseHandle {
            control: Arc::clone(&self.pause_control),
        }
    }

    /// Return the terminal outcome, or None while the stream is still running.
    pub fn termination(&self) -> Option<CompletionStreamTermination> {
        match self.state {
            CompletionStreamState::Running => None,
            CompletionStreamState::Exhausted => Some(CompletionStreamTermination::Exhausted),
            CompletionStreamState::Cancelled => Some(CompletionStreamTermination::Cancelled),
            CompletionStreamState::Failed => Some(CompletionStreamTermination::Failed),
        }
    }

    /// The aggregated assistant choice.
    ///
    /// The canonical aggregate is finalized when the raw source reaches EOF.
    /// Before then this remains the stream's current placeholder value.
    pub fn choice(&self) -> &OneOrMany<AssistantContent> {
        &self.choice
    }

    /// The provider's normalized terminal record, once yielded.
    pub fn final_record(&self) -> Option<&StreamFinal> {
        self.final_record.as_ref()
    }

    /// The provider-assigned message ID captured by the stream, when present.
    pub fn message_id(&self) -> Option<&str> {
        self.message_id.as_deref()
    }

    /// Token usage reported by the provider for this response.
    ///
    /// Returns the usage carried by the final response once the stream has
    /// produced it. Until then — or when the provider does not report streamed
    /// usage — this returns [`Usage::new`], the zero-valued sentinel for missing
    /// usage metrics.
    pub fn usage(&self) -> Usage {
        self.final_record
            .as_ref()
            .map(|record| record.usage)
            .unwrap_or_default()
    }

    /// Convert a naturally exhausted stream into its completed response.
    ///
    /// Content observed before exhaustion is provisional. Terminal metadata
    /// already yielded by the source remains inspectable through the read-only
    /// accessors, but no running, cancelled, or failed stream can be mistaken
    /// for a successfully completed response.
    pub fn into_response(self) -> Result<CompletionResponse, CompletionStreamFinalizationError> {
        match self.state {
            CompletionStreamState::Running => {
                return Err(CompletionStreamFinalizationError::Running);
            }
            CompletionStreamState::Cancelled => {
                return Err(CompletionStreamFinalizationError::Cancelled);
            }
            CompletionStreamState::Failed => {
                return Err(CompletionStreamFinalizationError::Failed);
            }
            CompletionStreamState::Exhausted => {}
        }

        let CompletionStream {
            choice,
            final_record,
            message_id: stream_message_id,
            ..
        } = self;
        let (usage, terminal_message_id, finish_reason, provider, model) = final_record
            .map_or_else(
                || (Usage::new(), None, None, String::new(), None),
                |record| {
                    (
                        record.usage,
                        record.message_id,
                        record.finish_reason,
                        record.provider,
                        record.model,
                    )
                },
            );

        Ok(CompletionResponse {
            choice,
            usage,
            message_id: stream_message_id.or(terminal_message_id),
            finish_reason,
            provider,
            model,
        })
    }

    fn append_text_chunk(&mut self, text: &str) {
        if let Some(index) = self.text_item_index
            && let Some(AssistantContent::Text(existing_text)) = self.assistant_items.get_mut(index)
        {
            existing_text.text.push_str(text);
            return;
        }

        self.assistant_items
            .push(AssistantContent::text(text.to_owned()));
        self.text_item_index = Some(self.assistant_items.len() - 1);
    }

    fn append_text_additional_params(&mut self, additional_params: serde_json::Value) {
        if additional_params.is_null() {
            return;
        }

        let index = if let Some(index) = self.text_item_index
            && matches!(
                self.assistant_items.get(index),
                Some(AssistantContent::Text(_))
            ) {
            index
        } else {
            self.assistant_items.push(AssistantContent::text(""));
            let index = self.assistant_items.len() - 1;
            self.text_item_index = Some(index);
            index
        };

        let Some(AssistantContent::Text(text)) = self.assistant_items.get_mut(index) else {
            return;
        };

        match text.additional_params.as_mut() {
            Some(existing) => merge_text_additional_params(existing, additional_params),
            None => text.additional_params = Some(additional_params),
        }
    }

    /// Accumulate streaming reasoning delta text into assistant_items.
    /// Providers that only emit ReasoningDelta (not full Reasoning blocks)
    /// need this so the aggregated response includes reasoning content.
    fn append_reasoning_chunk(&mut self, id: &Option<String>, text: &str) {
        if let Some(index) = self.reasoning_item_index
            && let Some(AssistantContent::Reasoning(existing)) = self.assistant_items.get_mut(index)
            && let Some(ReasoningContent::Text {
                text: existing_text,
                ..
            }) = existing.content.last_mut()
        {
            existing_text.push_str(text);
            return;
        }

        self.assistant_items
            .push(AssistantContent::Reasoning(Reasoning {
                id: id.clone(),
                content: vec![ReasoningContent::Text {
                    text: text.to_string(),
                    signature: None,
                }],
            }));
        self.reasoning_item_index = Some(self.assistant_items.len() - 1);
    }
}

fn merge_text_additional_params(existing: &mut serde_json::Value, incoming: serde_json::Value) {
    match (existing, incoming) {
        (serde_json::Value::Object(existing_map), serde_json::Value::Object(incoming_map)) => {
            for (key, incoming_value) in incoming_map {
                match existing_map.get_mut(&key) {
                    Some(existing_value) => match (existing_value, incoming_value) {
                        (
                            serde_json::Value::Array(existing_array),
                            serde_json::Value::Array(mut incoming_array),
                        ) => existing_array.append(&mut incoming_array),
                        (existing_value, incoming_value) => {
                            merge_text_additional_params(existing_value, incoming_value);
                        }
                    },
                    None => {
                        existing_map.insert(key, incoming_value);
                    }
                }
            }
        }
        (existing, incoming) => {
            *existing = incoming;
        }
    }
}

impl Stream for CompletionStream {
    type Item = Result<StreamedAssistantContent, CompletionError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let stream = self.get_mut();

        if stream.state != CompletionStreamState::Running {
            return Poll::Ready(None);
        }

        if stream.pause_control.poll_ready(cx).is_pending() {
            return Poll::Pending;
        }

        loop {
            match Pin::new(&mut stream.inner).poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    if stream.assistant_items.is_empty() {
                        stream.assistant_items.push(AssistantContent::text(""));
                    }

                    if let Some(choice) =
                        OneOrMany::from_iter_optional(std::mem::take(&mut stream.assistant_items))
                    {
                        stream.choice = choice;
                    }
                    stream.state = CompletionStreamState::Exhausted;
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(Err(err))) => {
                    stream.replace_source_with_empty();
                    stream.state = CompletionStreamState::Failed;
                    return Poll::Ready(Some(Err(err)));
                }
                Poll::Ready(Some(Ok(choice))) => match choice {
                    RawStreamingChoice::Message(text) => {
                        stream.reasoning_item_index = None;
                        stream.append_text_chunk(&text);
                        return Poll::Ready(Some(Ok(StreamedAssistantContent::text(&text))));
                    }
                    RawStreamingChoice::TextStart { additional_params } => {
                        stream.reasoning_item_index = None;
                        stream.text_item_index = None;
                        if let Some(additional_params) = additional_params {
                            stream.append_text_additional_params(additional_params);
                        }
                        continue;
                    }
                    RawStreamingChoice::TextAdditionalParams(additional_params) => {
                        stream.append_text_additional_params(additional_params);
                        continue;
                    }
                    RawStreamingChoice::ToolCallDelta {
                        id,
                        internal_call_id,
                        content,
                    } => {
                        return Poll::Ready(Some(Ok(StreamedAssistantContent::ToolCallDelta {
                            id,
                            internal_call_id,
                            content,
                        })));
                    }
                    RawStreamingChoice::Reasoning { id, content } => {
                        let reasoning = Reasoning {
                            id,
                            content: vec![content],
                        };
                        stream.text_item_index = None;
                        stream.reasoning_item_index = None;
                        stream
                            .assistant_items
                            .push(AssistantContent::Reasoning(reasoning.clone()));
                        return Poll::Ready(Some(Ok(StreamedAssistantContent::Reasoning(
                            reasoning,
                        ))));
                    }
                    RawStreamingChoice::ReasoningDelta { id, reasoning } => {
                        stream.text_item_index = None;
                        stream.append_reasoning_chunk(&id, &reasoning);
                        return Poll::Ready(Some(Ok(StreamedAssistantContent::ReasoningDelta {
                            id,
                            reasoning,
                        })));
                    }
                    RawStreamingChoice::ToolCall(raw_tool_call) => {
                        let internal_call_id = raw_tool_call.internal_call_id.clone();
                        let tool_call: ToolCall = raw_tool_call.into();
                        stream.text_item_index = None;
                        stream.reasoning_item_index = None;
                        stream
                            .assistant_items
                            .push(AssistantContent::ToolCall(tool_call.clone()));
                        return Poll::Ready(Some(Ok(StreamedAssistantContent::ToolCall {
                            tool_call,
                            internal_call_id,
                        })));
                    }
                    RawStreamingChoice::FinalResponse(response) => {
                        if stream.final_record_yielded {
                            continue;
                        }
                        stream.final_record = Some(response.clone());
                        stream.final_record_yielded = true;
                        let final_response = StreamedAssistantContent::final_response(response);
                        return Poll::Ready(Some(Ok(final_response)));
                    }
                    RawStreamingChoice::MessageId(id) => {
                        stream.message_id = Some(id);
                        continue;
                    }
                    RawStreamingChoice::Unknown(value) => {
                        return Poll::Ready(Some(Ok(StreamedAssistantContent::Unknown(value))));
                    }
                },
            }
        }
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[allow(dead_code)]
fn _assert_completion_stream_is_send() {
    fn assert_send<T: Send>() {}

    assert_send::<CompletionStream>();
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[allow(dead_code)]
fn _assert_completion_stream_accepts_worker_local_stream() {
    let worker_local = std::rc::Rc::new(());
    let stream = futures::stream::once(async move {
        std::future::pending::<()>().await;
        drop(worker_local);
        Ok(RawStreamingChoice::Message(String::new()))
    });

    let _stream = CompletionStream::from_stream(stream);
}

// Test module
#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use async_stream::stream;
    use tokio::time::sleep;

    /// Terminal record for mock streams with a known total-token count.
    fn mock_final(total_tokens: u64) -> StreamFinal {
        let mut usage = Usage::new();
        usage.total_tokens = total_tokens;
        StreamFinal::new("mock", usage)
    }

    fn create_mock_stream() -> CompletionStream {
        let stream = stream! {
            yield Ok(RawStreamingChoice::Message("hello 1".to_string()));
            sleep(Duration::from_millis(100)).await;
            yield Ok(RawStreamingChoice::Message("hello 2".to_string()));
            sleep(Duration::from_millis(100)).await;
            yield Ok(RawStreamingChoice::Message("hello 3".to_string()));
            sleep(Duration::from_millis(100)).await;
            yield Ok(RawStreamingChoice::FinalResponse(mock_final(15)));
        };

        CompletionStream::from_stream(stream)
    }

    fn create_reasoning_stream() -> CompletionStream {
        let stream = stream! {
            yield Ok(RawStreamingChoice::Reasoning {
                id: Some("rs_1".to_string()),
                content: ReasoningContent::Text {
                    text: "step one".to_string(),
                    signature: Some("sig_1".to_string()),
                },
            });
            yield Ok(RawStreamingChoice::Message("final answer".to_string()));
            yield Ok(RawStreamingChoice::FinalResponse(mock_final(5)));
        };

        CompletionStream::from_stream(stream)
    }

    fn create_reasoning_only_stream() -> CompletionStream {
        let stream = stream! {
            yield Ok(RawStreamingChoice::Reasoning {
                id: Some("rs_only".to_string()),
                content: ReasoningContent::Summary("hidden summary".to_string()),
            });
            yield Ok(RawStreamingChoice::FinalResponse(mock_final(2)));
        };

        CompletionStream::from_stream(stream)
    }

    fn create_interleaved_stream() -> CompletionStream {
        let stream = stream! {
            yield Ok(RawStreamingChoice::Reasoning {
                id: Some("rs_interleaved".to_string()),
                content: ReasoningContent::Text {
                    text: "chain-of-thought".to_string(),
                    signature: None,
                },
            });
            yield Ok(RawStreamingChoice::Message("final-text".to_string()));
            yield Ok(RawStreamingChoice::ToolCall(
                RawStreamingToolCall::new(
                    "tool_1".to_string(),
                    "mock_tool".to_string(),
                    serde_json::json!({"arg": 1}),
                ),
            ));
            yield Ok(RawStreamingChoice::FinalResponse(mock_final(3)));
        };

        CompletionStream::from_stream(stream)
    }

    fn create_text_tool_text_stream() -> CompletionStream {
        let stream = stream! {
            yield Ok(RawStreamingChoice::Message("first".to_string()));
            yield Ok(RawStreamingChoice::ToolCall(
                RawStreamingToolCall::new(
                    "tool_split".to_string(),
                    "mock_tool".to_string(),
                    serde_json::json!({"arg": "x"}),
                ),
            ));
            yield Ok(RawStreamingChoice::Message("second".to_string()));
            yield Ok(RawStreamingChoice::FinalResponse(mock_final(3)));
        };

        CompletionStream::from_stream(stream)
    }

    fn create_text_metadata_stream() -> CompletionStream {
        let stream = stream! {
            yield Ok(RawStreamingChoice::TextStart {
                additional_params: None,
            });
            yield Ok(RawStreamingChoice::Message("first".to_string()));
            yield Ok(RawStreamingChoice::TextAdditionalParams(serde_json::json!({
                "citations": [{
                    "type": "char_location",
                    "cited_text": "First citation.",
                    "document_index": 0,
                    "start_char_index": 0,
                    "end_char_index": 15
                }]
            })));
            yield Ok(RawStreamingChoice::TextAdditionalParams(serde_json::json!({
                "citations": [{
                    "type": "char_location",
                    "cited_text": "Second citation.",
                    "document_index": 0,
                    "start_char_index": 16,
                    "end_char_index": 32
                }]
            })));
            yield Ok(RawStreamingChoice::TextStart {
                additional_params: Some(serde_json::json!({
                    "block": 2
                })),
            });
            yield Ok(RawStreamingChoice::Message("second".to_string()));
            yield Ok(RawStreamingChoice::FinalResponse(mock_final(3)));
        };

        CompletionStream::from_stream(stream)
    }

    #[tokio::test]
    async fn into_completion_response_derives_usage_from_final_response() {
        let mut stream = create_mock_stream();

        // Drain the stream so the final response (and its usage) is captured.
        while stream.next().await.is_some() {}

        // usage() surfaces the final response's token usage...
        assert_eq!(stream.usage().total_tokens, 15);

        // ...and checked finalization carries it instead of a zero sentinel.
        let response = stream
            .into_response()
            .expect("drained stream should finalize");
        assert_eq!(response.usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn usage_is_zero_sentinel_before_final_response() {
        // A stream that never yields a FinalResponse reports the zero sentinel.
        let stream = CompletionStream::from_stream(stream! {
            yield Ok(RawStreamingChoice::Message("no final response".to_string()));
        });
        assert_eq!(stream.usage().total_tokens, 0);
    }

    #[tokio::test]
    async fn conversion_without_final_record_keeps_missing_metadata_and_zero_usage() {
        let mut stream = CompletionStream::from_stream(stream! {
            yield Ok(RawStreamingChoice::Message("no final response".to_string()));
        });
        while stream.next().await.is_some() {}

        let response = stream
            .into_response()
            .expect("drained stream should finalize");

        assert_eq!(response.usage, Usage::new());
        assert_eq!(response.message_id, None);
        assert_eq!(response.finish_reason, None);
        assert!(response.provider.is_empty());
        assert_eq!(response.model, None);
        assert!(matches!(
            response.choice.first_ref(),
            AssistantContent::Text(Text { text, .. }) if text == "no final response"
        ));
    }

    #[tokio::test]
    async fn accessors_and_conversion_preserve_terminal_metadata_precedence() {
        let final_record = StreamFinal::new("mock-provider", Usage::new())
            .with_message_id("terminal-id")
            .with_model("mock-model")
            .with_finish_reason(crate::completion::FinishReason::Stop);
        let source = futures::stream::iter([
            Ok(RawStreamingChoice::Message("hello".to_owned())),
            Ok(RawStreamingChoice::FinalResponse(final_record.clone())),
            Ok(RawStreamingChoice::MessageId("stream-id".to_owned())),
        ]);
        let mut stream = CompletionStream::from_stream(source);
        let mut emitted_finals = 0;

        while let Some(item) = stream.next().await {
            if matches!(item, Ok(StreamedAssistantContent::Final(_))) {
                emitted_finals += 1;
            }
        }

        assert_eq!(emitted_finals, 1);
        assert_eq!(stream.message_id(), Some("stream-id"));
        assert_eq!(stream.final_record(), Some(&final_record));
        assert!(matches!(
            stream.choice().first_ref(),
            AssistantContent::Text(Text { text, .. }) if text == "hello"
        ));

        let response = stream
            .into_response()
            .expect("drained stream should finalize");
        assert_eq!(response.usage, Usage::new());
        assert_eq!(response.message_id.as_deref(), Some("stream-id"));
        assert_eq!(response.provider, "mock-provider");
        assert_eq!(response.model.as_deref(), Some("mock-model"));
        assert_eq!(
            response.finish_reason,
            Some(crate::completion::FinishReason::Stop)
        );
    }

    #[tokio::test]
    async fn inherent_next_resumes_the_same_pending_provider_stream() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let polls = Arc::new(AtomicUsize::new(0));
        let source_polls = Arc::clone(&polls);
        let source = futures::stream::once(futures::future::poll_fn(move |cx| {
            if source_polls.fetch_add(1, Ordering::SeqCst) == 0 {
                cx.waker().wake_by_ref();
                Poll::Pending
            } else {
                Poll::Ready(Ok(RawStreamingChoice::Message("resumed".to_owned())))
            }
        }));
        let mut stream = CompletionStream::from_stream(source);

        {
            let next = stream.next();
            futures::pin_mut!(next);
            assert!(futures::poll!(next).is_pending());
        }

        let item = stream.next().await;
        assert!(matches!(
            item,
            Some(Ok(StreamedAssistantContent::Text(Text { text, .. }))) if text == "resumed"
        ));
        assert_eq!(polls.load(Ordering::SeqCst), 2);
    }

    mod inherent_next_without_stream_ext {
        use super::{CompletionError, CompletionStream, RawStreamingChoice};

        #[tokio::test]
        async fn polls_without_the_extension_trait_in_scope() {
            let source = futures::stream::iter([Ok::<_, CompletionError>(
                RawStreamingChoice::Message("hello".to_owned()),
            )]);
            let mut stream = CompletionStream::from_stream(source);

            assert!(stream.next().await.is_some());
        }
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
                Ok(StreamedAssistantContent::ToolCall {
                    tool_call,
                    internal_call_id,
                }) => {
                    println!("\nTool Call: {tool_call:?}, internal_call_id={internal_call_id:?}");
                    chunk_count += 1;
                }
                Ok(StreamedAssistantContent::ToolCallDelta {
                    id,
                    internal_call_id,
                    content,
                }) => {
                    println!(
                        "\nTool Call delta: id={id:?}, internal_call_id={internal_call_id:?}, content={content:?}"
                    );
                    chunk_count += 1;
                }
                Ok(StreamedAssistantContent::Final(res)) => {
                    println!("\nFinal response: {res:?}");
                }
                Ok(StreamedAssistantContent::Reasoning(reasoning)) => {
                    let reasoning = reasoning.display_text();
                    print!("{reasoning}");
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                }
                Ok(StreamedAssistantContent::ReasoningDelta { reasoning, .. }) => {
                    println!("Reasoning delta: {reasoning}");
                    chunk_count += 1;
                }
                Ok(StreamedAssistantContent::Unknown(value)) => {
                    println!("\nUnknown item: {value:?}");
                    chunk_count += 1;
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
        assert_eq!(
            stream.termination(),
            Some(CompletionStreamTermination::Cancelled)
        );
        assert!(matches!(
            stream.into_response(),
            Err(CompletionStreamFinalizationError::Cancelled)
        ));
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

    #[tokio::test]
    async fn paused_stream_registers_a_waker_without_polling_the_source() {
        use futures::task::{ArcWake, waker};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct WakeCounter(AtomicUsize);

        impl ArcWake for WakeCounter {
            fn wake_by_ref(arc_self: &Arc<Self>) {
                arc_self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let source_polls = Arc::new(AtomicUsize::new(0));
        let counted_polls = Arc::clone(&source_polls);
        let source = futures::stream::poll_fn(move |_| {
            counted_polls.fetch_add(1, Ordering::SeqCst);
            Poll::Ready(Some(Ok(RawStreamingChoice::Message("resumed".to_owned()))))
        });
        let mut stream = CompletionStream::from_stream(source);
        stream.pause();

        let wake_count = Arc::new(WakeCounter(AtomicUsize::new(0)));
        let task_waker = waker(Arc::clone(&wake_count));
        let mut cx = Context::from_waker(&task_waker);
        {
            let next = stream.next();
            futures::pin_mut!(next);
            for _ in 0..8 {
                assert!(next.as_mut().poll(&mut cx).is_pending());
            }
        }

        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(source_polls.load(Ordering::SeqCst), 0);
        assert_eq!(wake_count.0.load(Ordering::SeqCst), 0);

        stream.resume();
        assert_eq!(wake_count.0.load(Ordering::SeqCst), 1);
        assert!(matches!(
            stream.next().await,
            Some(Ok(StreamedAssistantContent::Text(Text { text, .. }))) if text == "resumed"
        ));
        assert_eq!(source_polls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn pause_handle_resumes_a_pending_read_after_a_real_delay() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let source_polls = Arc::new(AtomicUsize::new(0));
        let counted_polls = Arc::clone(&source_polls);
        let source = futures::stream::poll_fn(move |_| {
            counted_polls.fetch_add(1, Ordering::SeqCst);
            Poll::Ready(Some(Ok(RawStreamingChoice::Message("resumed".to_owned()))))
        });
        let mut stream = CompletionStream::from_stream(source);
        let pause = stream.pause_handle();
        pause.pause();

        let mut next = Box::pin(stream.next());
        tokio::select! {
            biased;
            item = &mut next => panic!("paused read completed early: {item:?}"),
            _ = tokio::time::sleep(Duration::from_millis(20)) => {}
        }
        assert_eq!(source_polls.load(Ordering::SeqCst), 0);
        pause.resume();

        let item = tokio::time::timeout(Duration::from_millis(100), next)
            .await
            .expect("resume must wake the pending read");
        assert!(matches!(
            item,
            Some(Ok(StreamedAssistantContent::Text(Text { text, .. }))) if text == "resumed"
        ));
        assert_eq!(source_polls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn provider_error_is_terminal_even_when_its_text_mentions_aborted() {
        let source = futures::stream::iter([
            Err(CompletionError::ProviderError(
                "upstream aborted its own operation".to_owned(),
            )),
            Ok(RawStreamingChoice::Message(
                "must not be delivered".to_owned(),
            )),
        ]);
        let mut stream = CompletionStream::from_stream(source);

        assert!(matches!(
            stream.next().await,
            Some(Err(CompletionError::ProviderError(message)))
                if message == "upstream aborted its own operation"
        ));
        assert_eq!(
            stream.termination(),
            Some(CompletionStreamTermination::Failed)
        );
        assert!(stream.next().await.is_none());
        assert!(stream.next().await.is_none());
        assert!(matches!(
            stream.into_response(),
            Err(CompletionStreamFinalizationError::Failed)
        ));
    }

    #[tokio::test]
    async fn metadata_only_sequences_are_consumed_iteratively() {
        let final_record = mock_final(1);
        let mut items = Vec::with_capacity(40_003);
        for index in 0..10_000 {
            items.push(Ok(RawStreamingChoice::MessageId(format!("msg_{index}"))));
        }
        items.push(Ok(RawStreamingChoice::FinalResponse(final_record.clone())));
        for _ in 0..10_000 {
            items.push(Ok(RawStreamingChoice::FinalResponse(final_record.clone())));
        }
        for _ in 0..10_000 {
            items.push(Ok(RawStreamingChoice::TextStart {
                additional_params: None,
            }));
        }
        for index in 0..10_000 {
            items.push(Ok(RawStreamingChoice::TextAdditionalParams(
                serde_json::json!({"sequence": index}),
            )));
        }
        items.push(Ok(RawStreamingChoice::Message("done".to_owned())));

        let mut stream = CompletionStream::from_stream(futures::stream::iter(items));
        assert!(matches!(
            stream.next().await,
            Some(Ok(StreamedAssistantContent::Final(_)))
        ));
        assert!(matches!(
            stream.next().await,
            Some(Ok(StreamedAssistantContent::Text(Text { text, .. }))) if text == "done"
        ));
        assert!(stream.next().await.is_none());
        assert_eq!(
            stream.termination(),
            Some(CompletionStreamTermination::Exhausted)
        );
    }

    #[test]
    fn running_stream_cannot_be_finalized() {
        let stream = CompletionStream::from_stream(futures::stream::pending());
        assert!(matches!(
            stream.into_response(),
            Err(CompletionStreamFinalizationError::Running)
        ));
    }

    #[tokio::test]
    async fn test_stream_aggregates_reasoning_content() {
        let mut stream = create_reasoning_stream();
        while stream.next().await.is_some() {}

        let choice_items = stream.choice();

        assert!(choice_items.iter().any(|item| matches!(
            item,
            AssistantContent::Reasoning(Reasoning {
                id: Some(id),
                content
            }) if id == "rs_1"
                && matches!(
                    content.first(),
                    Some(ReasoningContent::Text {
                        text,
                        signature: Some(signature)
                    }) if text == "step one" && signature == "sig_1"
                )
        )));
    }

    #[tokio::test]
    async fn test_stream_reasoning_only_does_not_inject_empty_text() {
        let mut stream = create_reasoning_only_stream();
        while stream.next().await.is_some() {}

        let choice_items = stream.choice();
        assert_eq!(choice_items.len(), 1);
        assert!(matches!(
            choice_items.first_ref(),
            AssistantContent::Reasoning(Reasoning { id: Some(id), .. }) if id == "rs_only"
        ));
    }

    #[tokio::test]
    async fn test_stream_aggregates_assistant_items_in_arrival_order() {
        let mut stream = create_interleaved_stream();
        while stream.next().await.is_some() {}

        let choice_items = stream.choice();
        assert_eq!(choice_items.len(), 3);
        assert!(matches!(
            choice_items.first_ref(),
            AssistantContent::Reasoning(Reasoning { id: Some(id), .. }) if id == "rs_interleaved"
        ));
        assert!(matches!(
            choice_items.iter().nth(1),
            Some(AssistantContent::Text(Text { text, .. })) if text == "final-text"
        ));
        assert!(matches!(
            choice_items.iter().nth(2),
            Some(AssistantContent::ToolCall(ToolCall { id, .. })) if id == "tool_1"
        ));
    }

    #[tokio::test]
    async fn unknown_choice_reaches_consumer_but_not_aggregated_choice() {
        let unknown = serde_json::json!({
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
        });
        let yielded = unknown.clone();
        let stream = stream! {
            yield Ok(RawStreamingChoice::Unknown(yielded));
            yield Ok(RawStreamingChoice::Message("done".to_string()));
            yield Ok(RawStreamingChoice::FinalResponse(mock_final(1)));
        };
        let mut stream = CompletionStream::from_stream(stream);

        let mut consumer_unknown = None;
        let mut consumer_text = String::new();
        while let Some(item) = stream.next().await {
            match item.expect("stream item should be Ok") {
                StreamedAssistantContent::Unknown(value) => consumer_unknown = Some(value),
                StreamedAssistantContent::Text(text) => consumer_text.push_str(&text.text),
                _ => {}
            }
        }

        // The consumer receives the unmodeled item verbatim ...
        assert_eq!(consumer_unknown.as_ref(), Some(&unknown));
        assert_eq!(consumer_text, "done");

        // ... but it is structurally absent from the aggregated assistant choice
        // (the sole source of persisted history): only the text item remains.
        let choice_items = stream.choice();
        assert_eq!(choice_items.len(), 1);
        assert!(matches!(
            choice_items.first_ref(),
            AssistantContent::Text(Text { text, .. }) if text == "done"
        ));
    }

    #[tokio::test]
    async fn test_stream_keeps_non_contiguous_text_chunks_split_by_tool_call() {
        let mut stream = create_text_tool_text_stream();
        while stream.next().await.is_some() {}

        let choice_items = stream.choice();
        assert_eq!(choice_items.len(), 3);
        assert!(matches!(
            choice_items.first_ref(),
            AssistantContent::Text(Text { text, .. }) if text == "first"
        ));
        assert!(matches!(
            choice_items.iter().nth(1),
            Some(AssistantContent::ToolCall(ToolCall { id, .. })) if id == "tool_split"
        ));
        assert!(matches!(
            choice_items.iter().nth(2),
            Some(AssistantContent::Text(Text { text, .. })) if text == "second"
        ));
    }

    #[tokio::test]
    async fn test_stream_preserves_text_additional_params() {
        let mut stream = create_text_metadata_stream();
        while stream.next().await.is_some() {}

        let choice_items = stream.choice();
        assert_eq!(choice_items.len(), 2);

        let AssistantContent::Text(Text {
            text,
            additional_params: Some(additional_params),
        }) = choice_items.first_ref()
        else {
            panic!("expected first text item with metadata");
        };
        assert_eq!(text, "first");
        assert_eq!(
            additional_params["citations"]
                .as_array()
                .expect("citations should be an array")
                .len(),
            2
        );

        let Some(AssistantContent::Text(Text {
            text,
            additional_params: Some(additional_params),
        })) = choice_items.iter().nth(1)
        else {
            panic!("expected second text item with metadata");
        };
        assert_eq!(text, "second");
        assert_eq!(additional_params["block"], 2);
    }
}

/// Describes responses from a streamed provider response which is either text, a tool call or a final usage response.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum StreamedAssistantContent {
    /// Text delta emitted by the assistant.
    Text(Text),
    /// Complete tool call emitted by the assistant.
    ToolCall {
        tool_call: ToolCall,
        /// Rig-generated unique identifier for this tool call.
        /// Use this to correlate with ToolCallDelta events.
        internal_call_id: String,
    },
    /// Partial tool call data emitted by the assistant.
    ToolCallDelta {
        /// Provider-supplied tool call ID.
        id: String,
        /// Rig-generated unique identifier for this tool call.
        internal_call_id: String,
        content: ToolCallDeltaContent,
    },
    /// Complete reasoning block emitted by the assistant.
    Reasoning(Reasoning),
    /// Partial reasoning text emitted by the assistant.
    ReasoningDelta {
        /// Provider-supplied reasoning block ID, when present.
        id: Option<String>,
        /// Partial reasoning text.
        reasoning: String,
    },
    /// The provider's normalized terminal record, if yielded by the provider
    /// stream. `StreamFinal`'s required `kind` field is the discriminant that
    /// keeps this variant distinguishable from [`Self::Unknown`] under
    /// `#[serde(untagged)]`.
    Final(StreamFinal),
    /// A provider-native output item rig does not model, preserved verbatim —
    /// e.g. an OpenAI Responses hosted-tool result (`web_search_call`,
    /// `file_search_call`, `computer_call`, `code_interpreter_call`). It is
    /// yielded to the consumer for inspection/forwarding but is not added to the
    /// accumulated assistant message or persisted history. Kept last because the
    /// enum is `#[serde(untagged)]` and a raw [`Value`](serde_json::Value)
    /// matches anything, so earlier (typed) variants must be tried first.
    Unknown(serde_json::Value),
}

impl StreamedAssistantContent {
    /// Create a text stream item.
    pub fn text(text: &str) -> Self {
        Self::Text(Text::new(text.to_string()))
    }

    /// Create a final response stream item.
    pub fn final_response(res: StreamFinal) -> Self {
        Self::Final(res)
    }
}

/// Streamed user content. This content is primarily used to represent tool results from tool calls made during a multi-turn/step agent prompt.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum StreamedUserContent {
    /// Tool result emitted during a multi-turn streaming agent loop.
    ToolResult {
        tool_result: ToolResult,
        /// Rig-generated unique identifier for the tool call this result
        /// belongs to. Use this to correlate with the originating
        /// [`StreamedAssistantContent::ToolCall::internal_call_id`].
        internal_call_id: String,
    },
}

impl StreamedUserContent {
    /// Create a streamed tool result correlated to an internal tool call ID.
    pub fn tool_result(tool_result: ToolResult, internal_call_id: String) -> Self {
        Self::ToolResult {
            tool_result,
            internal_call_id,
        }
    }
}
