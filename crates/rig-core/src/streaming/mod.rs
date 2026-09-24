//! Runtime-independent completion events, response aggregation, and stream controls.
//!
//! ```
//! use rig_core::streaming::PauseControl;
//!
//! let control = PauseControl::new();
//! control.pause();
//! assert!(control.is_paused());
//! control.resume();
//! ```

mod accumulator;
mod block_id;
mod event;

use futures::StreamExt as _;

use crate::completion::{CompletionEnd, CompletionResponse, Usage};
use crate::error::ErrorReport;
use crate::error::ProviderError;
use crate::id::MessageId;
use crate::message::{AssistantContent, ToolResult};
pub use accumulator::BlockAccumulator;
pub use block_id::{BlockId, MintKind, SyntheticIds, non_empty_id};
pub use event::{BlockClose, BlockKind, Delta, StreamEvent, ToolCallEnd};
use futures::Stream;
use futures::stream::{AbortHandle, Abortable};
use futures::task::AtomicWaker;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

/// Mutable state borrowed for one event-folding step, shared by streaming
/// responses and [`CompletionFold`](crate::operation::CompletionFold).
pub(crate) struct FoldStep<'a> {
    pub accumulator: &'a mut BlockAccumulator,
    pub response: &'a mut Option<CompletionEnd>,
    pub message_id: &'a mut Option<MessageId>,
}

/// What one fold step decided about an event.
pub(crate) enum Absorbed {
    /// Forward this event (possibly rewritten with the block it finalized).
    Yield(StreamEvent),
    /// The accumulator rejected it; the stream keeps consuming.
    Failed(ErrorReport),
    /// A duplicate terminal: the first one latched.
    Skip,
}

/// Absorb one event into the fold.
pub(crate) fn absorb(step: FoldStep<'_>, event: StreamEvent) -> Absorbed {
    match event {
        StreamEvent::BlockStart {
            id,
            kind: BlockKind::Message,
        } => {
            // The wire announced the assistant message's own id; it
            // outranks the terminal record's.
            if let Some(message_id) = id.wire_str().and_then(MessageId::non_empty) {
                *step.message_id = Some(message_id);
            }
            Absorbed::Yield(StreamEvent::BlockStart {
                id,
                kind: BlockKind::Message,
            })
        }
        StreamEvent::Final(mut response) => {
            // A second terminal is a provider defect; the first one latched.
            if step.response.is_some() {
                return Absorbed::Skip;
            }
            // Finish-reason reconciliation against the accumulator's
            // authoritative view of completed calls, so a `stop` that was
            // really a tool call reads the same on both surfaces.
            response.finish_reason = response
                .finish_reason
                .map(|reason| reason.reconcile_with_output(step.accumulator.saw_tool_call()));
            // An explicit message-id block keeps precedence over the
            // terminal record's.
            if step.message_id.is_none() {
                step.message_id.clone_from(&response.message_id);
            }
            *step.response = Some(response.clone());
            Absorbed::Yield(StreamEvent::Final(response))
        }
        // Passed straight through; never folded into the aggregated choice.
        StreamEvent::Unknown(value) => Absorbed::Yield(StreamEvent::Unknown(value)),
        event => match step.accumulator.apply(&event) {
            // A block end that finalized a block publishes it under the key
            // its deltas carried.
            Ok(Some((id, block))) => {
                let StreamEvent::BlockEnd { end, .. } = event else {
                    // Only ends finalize; the accumulator upholds it.
                    return Absorbed::Yield(event);
                };
                Absorbed::Yield(StreamEvent::BlockEnd {
                    id,
                    end,
                    block: Some(block),
                })
            }
            Ok(None) => Absorbed::Yield(event),
            // Malformed complete input surfaces in-band.
            Err(error) => Absorbed::Failed(error),
        },
    }
}

/// Record `issuer` on every reasoning part of `choice` that names none.
pub fn stamp_reasoning(choice: Vec<AssistantContent>, issuer: &str) -> Vec<AssistantContent> {
    choice
        .into_iter()
        .map(|part| match part {
            AssistantContent::Reasoning(reasoning) if reasoning.provider.is_none() => {
                AssistantContent::Reasoning(reasoning.with_provider(issuer))
            }
            part => part,
        })
        .collect()
}

/// The folded completion response: the aggregated choice, its reasoning
/// stamped with the terminal record's issuer, and the terminal record with
/// the fold's derived facts: the message id (a message block's over the
/// record's), the finish reason reconciled with the choice's tool calls,
/// and no issuer left to apply.
pub(crate) fn fold_finish(
    mut accumulator: BlockAccumulator,
    terminal: CompletionEnd,
    message_id: Option<MessageId>,
) -> CompletionResponse {
    let choice = stamp_reasoning(accumulator.finish(), terminal.issuer());
    let has_tool_call = choice.iter().any(AssistantContent::is_tool_call);
    CompletionResponse {
        end: CompletionEnd {
            message_id: message_id.or(terminal.message_id),
            finish_reason: terminal
                .finish_reason
                .map(|reason| reason.reconcile_with_output(has_tool_call)),
            // Applied: every reasoning part of `choice` names its issuer.
            reasoning_issuer: None,
            meta: terminal.meta,
        },
        choice,
    }
}

/// Pause flag and single-consumer waker. Must not be shared across streams.
struct PauseState {
    paused: AtomicBool,
    waker: AtomicWaker,
}

/// Control for pausing and resuming a streaming response
#[derive(Clone)]
pub struct PauseControl {
    state: Arc<PauseState>,
}

impl PauseControl {
    /// Create a pause controller in the running state.
    pub fn new() -> Self {
        Self {
            state: Arc::new(PauseState {
                paused: AtomicBool::new(false),
                waker: AtomicWaker::new(),
            }),
        }
    }

    /// Pause polling of the public stream until [`PauseControl::resume`] is called.
    pub fn pause(&self) {
        self.state.paused.store(true, Ordering::Release);
    }

    /// Resume polling after a pause.
    pub fn resume(&self) {
        self.state.paused.store(false, Ordering::Release);
        self.state.waker.wake();
    }

    /// Returns whether the stream is currently paused.
    pub fn is_paused(&self) -> bool {
        self.state.paused.load(Ordering::Acquire)
    }
}

impl Default for PauseControl {
    fn default() -> Self {
        Self::new()
    }
}

/// Adapter-selected policy for tool arguments that fail to parse at an end event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnparseableToolInput {
    /// Drop the call silently: the input never fully arrived (the
    /// OpenAI-compatible end-of-stream flush of pending calls).
    Drop,
    /// Deliver the call with `{}` arguments: the wire superseded the call
    /// mid-assembly (the OpenAI-compatible same-slot eviction path).
    EmptyObject,
    /// Surface an in-band error item: the wire promised a complete block
    /// (Anthropic `content_block_stop`, Bedrock `contentBlockStop`).
    Error,
    /// Leave the call open and emit nothing: the end was a completion
    /// *probe* (the OpenAI-compatible single-chunk immediate-emission path),
    /// and input that does not yet finalize may still be extended by later
    /// fragments and closed by a genuine flush.
    Keep,
}

/// Decoration a provider attaches to a streamed tool call that is still
/// assembling, matched by its established provider id (e.g. OpenRouter
/// encrypted reasoning details). Carried onto the completed call by the
/// adapter's end event.
#[derive(Debug, Clone)]
pub struct ToolCallDecoration {
    /// Established provider id of the call to decorate.
    pub tool_id: String,
    /// Provider signature to attach to the completed call.
    pub signature: Option<String>,
    /// Provider-specific metadata to attach to the completed call.
    pub additional_params: Option<serde_json::Value>,
}

/// Unmodeled JSON payload with content-redacted `Debug` output.
/// Serialization preserves the payload; [`Self::value`] explicitly exposes it.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct UnknownPayload(serde_json::Value);

impl UnknownPayload {
    /// Wrap a raw unmodeled payload.
    pub fn new(value: serde_json::Value) -> Self {
        Self(value)
    }

    /// The raw payload, for consumers who opt in to the content.
    pub fn value(&self) -> &serde_json::Value {
        &self.0
    }
}

impl std::fmt::Debug for UnknownPayload {
    /// Reports serialized size without exposing payload content.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bytes = serde_json::to_vec(&self.0).map_or(0, |json| json.len());
        write!(f, "UnknownPayload({bytes} bytes redacted)")
    }
}

impl From<serde_json::Value> for UnknownPayload {
    fn from(value: serde_json::Value) -> Self {
        Self(value)
    }
}

#[cfg(test)]
mod unknown_payload_tests;

/// Adapter events with provider errors. [`StreamingCompletionResponse::stream`]
/// converts errors to [`ErrorReport`] for consumers.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub type StreamingResult = Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>;

/// The stream a provider hands to [`StreamingCompletionResponse::stream`]
/// (browser wasm: `!Send` allowed).
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub type StreamingResult = Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>>>>;

/// The one stream item type: what [`StreamingCompletionResponse`] yields,
/// what the accumulator applies, what the bus carries.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub type StreamEvents = Pin<Box<dyn Stream<Item = Result<StreamEvent, ErrorReport>> + Send>>;

/// The one stream item type (browser wasm: `!Send` allowed).
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub type StreamEvents = Pin<Box<dyn Stream<Item = Result<StreamEvent, ErrorReport>>>>;

pub struct StreamingCompletionResponse {
    pub(crate) inner: Abortable<StreamEvents>,
    pub(crate) abort_handle: AbortHandle,
    pub(crate) pause_control: PauseControl,
    /// Accumulates the streamed parts of the final aggregated choice.
    accumulator: BlockAccumulator,
    /// Stable descriptor name of the provider producing this stream.
    ///
    /// Known when the stream is opened rather than when it terminates, so a
    /// stream that errors or is cancelled before its terminal record still
    /// names its provider.
    provider: String,
    /// Whether the terminal record names the provider: a stream that came
    /// over the bus ([`Self::from_events`]) is opened under the handler's
    /// label, and the provider behind it is only known once its terminal
    /// record arrives; a stream a provider opened ([`Self::stream`]) names
    /// its provider up front.
    provider_from_terminal: bool,
    /// The issuer of this stream's reasoning when it is known before the
    /// terminal record ([`Self::with_reasoning_issuer`]).
    reasoning_issuer: Option<String>,
    /// Prevents polling the inner stream after it ends.
    finished: bool,
    /// The provider's normalized terminal record, `None` until the stream
    /// yields it (and forever on truncation or a terminal error).
    pub response: Option<CompletionEnd>,
    /// Provider-assigned message ID (e.g. OpenAI Responses API `msg_` ID).
    pub message_id: Option<MessageId>,
}

impl StreamingCompletionResponse {
    /// Wrap a provider stream and initialize aggregation state.
    ///
    /// `provider` is the stable descriptor name of the provider producing the
    /// stream; it is recorded up front so it is available even when the stream
    /// never reaches its terminal record.
    pub fn stream(provider: impl Into<String>, inner: StreamingResult) -> Self {
        // The one place a provider's error half becomes the wire's: from
        // here on every item is `Result<StreamEvent, ErrorReport>`.
        let mapped: StreamEvents =
            Box::pin(inner.map(|item| item.map_err(|error| ErrorReport::from(&error))));
        Self {
            provider_from_terminal: false,
            ..Self::from_events(provider, mapped)
        }
    }

    /// Wraps normalized events without error conversion. Uses `provider` until
    /// a terminal record supplies a nonempty provider name.
    pub fn from_events(provider: impl Into<String>, inner: StreamEvents) -> Self {
        let (abort_handle, abort_registration) = AbortHandle::new_pair();
        let abortable_stream = Abortable::new(inner, abort_registration);
        let pause_control = PauseControl::new();
        Self {
            inner: abortable_stream,
            abort_handle,
            pause_control,
            accumulator: BlockAccumulator::new(),
            provider: provider.into(),
            provider_from_terminal: true,
            reasoning_issuer: None,
            finished: false,
            response: None,
            message_id: None,
        }
    }

    /// Stable descriptor name of the provider producing this stream.
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// Name the issuer of this stream's reasoning up front, for a transport
    /// or deployment of another provider's models, so a partial turn taken
    /// before the terminal record records it too. The terminal record must
    /// name the same issuer ([`CompletionEnd::reasoning_issuer`]): once it
    /// arrives, it is the one read.
    pub fn with_reasoning_issuer(mut self, issuer: impl Into<String>) -> Self {
        self.reasoning_issuer = Some(issuer.into());
        self
    }

    /// The issuer this stream's reasoning records: the terminal record's
    /// [`CompletionEnd::issuer`] once it has arrived; before it, the issuer
    /// named up front, else the provider that opened the stream. `None`
    /// before the terminal of a stream rebuilt from events
    /// ([`Self::from_events`]), whose opening label names a handler, not an
    /// issuer: its reasoning is then of unknown provenance.
    pub fn reasoning_issuer(&self) -> Option<&str> {
        match &self.response {
            Some(terminal) => Some(terminal.issuer().as_str()),
            None => self
                .reasoning_issuer
                .as_deref()
                .or((!self.provider_from_terminal).then_some(self.provider.as_str())),
        }
    }

    /// Returns the accumulated choice without consuming it.
    /// See [`BlockAccumulator::snapshot`] for unfinished and empty-part handling.
    pub fn snapshot(&self) -> Vec<AssistantContent> {
        self.accumulator.snapshot()
    }

    /// Consume the stream into the unary response shape: the aggregated
    /// choice and the terminal record. A stream that produced no terminal
    /// record is truncated per the emission contract and is refused.
    ///
    /// Events not yet polled are not part of the choice: drain the stream
    /// first when the whole turn is wanted.
    pub fn finish(self) -> Result<CompletionResponse, ProviderError> {
        let Some(terminal) = self.response else {
            return Err(ProviderError::Response(
                "provider stream ended without a terminal record; treating the turn as truncated"
                    .to_owned(),
            ));
        };
        Ok(fold_finish(self.accumulator, terminal, self.message_id))
    }

    /// Cancel the stream and immediately drop the provider's inner stream.
    /// Cancellation is surfaced as normal stream termination.
    ///
    /// Cancelling also resumes a paused stream: a consumer parked on the
    /// pause channel must observe the termination instead of waiting forever
    /// for a resume that will never affect a stream that no longer exists.
    pub fn cancel(&mut self) {
        self.abort_handle.abort();
        let (abort_handle, abort_registration) = AbortHandle::new_pair();
        let empty: StreamEvents = Box::pin(futures::stream::poll_fn(|_| Poll::Ready(None)));
        self.inner = Abortable::new(empty, abort_registration);
        self.abort_handle = abort_handle;
        self.pause_control.resume();
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

    /// Returns terminal usage, or [`Usage::default`] before a terminal record.
    /// Unreported counters remain `None`.
    pub fn usage(&self) -> Usage {
        self.response
            .as_ref()
            .map(|response| response.meta.usage)
            .unwrap_or_default()
    }
}

impl Stream for StreamingCompletionResponse {
    type Item = Result<StreamEvent, ErrorReport>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let stream = self.get_mut();

        // Do not poll the inner stream after termination.
        if stream.finished {
            return Poll::Ready(None);
        }

        if stream.is_paused() {
            // Register before rechecking to avoid losing a concurrent resume.
            // Parking without a self-wake prevents busy polling while paused.
            stream.pause_control.state.waker.register(cx.waker());
            if stream.is_paused() {
                return Poll::Pending;
            }
        }

        // Iterate over duplicate terminals without growing the stack.
        loop {
            return match Pin::new(&mut stream.inner).poll_next(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(None) => {
                    stream.finished = true;
                    Poll::Ready(None)
                }
                // Cancellation ends the stream without an error item; actual
                // errors remain visible even if later events can recover.
                Poll::Ready(Some(Err(err))) => Poll::Ready(Some(Err(err))),
                Poll::Ready(Some(Ok(event))) => {
                    let step = FoldStep {
                        accumulator: &mut stream.accumulator,
                        response: &mut stream.response,
                        message_id: &mut stream.message_id,
                    };
                    let absorbed = absorb(step, event);
                    // A stream that came over the bus learns its provider
                    // from the terminal record.
                    if stream.provider_from_terminal
                        && let Absorbed::Yield(StreamEvent::Final(terminal)) = &absorbed
                    {
                        stream.provider = terminal.meta.provider.to_string();
                    }
                    match absorbed {
                        Absorbed::Yield(event) => Poll::Ready(Some(Ok(event))),
                        // The stream keeps consuming, matching the
                        // malformed-frame contract.
                        Absorbed::Failed(error) => Poll::Ready(Some(Err(error))),
                        Absorbed::Skip => continue,
                    }
                }
            };
        }
    }
}

// Test module
#[cfg(test)]
mod tests;

/// Streamed user content. This content is primarily used to represent tool results from tool calls made during a multi-turn/step agent prompt.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum StreamedUserContent {
    /// Tool result emitted during a multi-turn streaming agent loop.
    ToolResult {
        tool_result: ToolResult,
        /// The block of the originating
        /// tool-call block; `tool_result.call` is
        /// the durable identifier of the answered call.
        id: BlockId,
    },
}

impl StreamedUserContent {
    /// Create a streamed tool result correlated to the block of its call.
    pub fn tool_result(tool_result: ToolResult, id: BlockId) -> Self {
        Self::ToolResult { tool_result, id }
    }
}
