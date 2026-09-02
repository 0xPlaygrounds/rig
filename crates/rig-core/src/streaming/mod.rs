//! This module provides functionality for working with streaming completion models.
//! It provides traits and types for generating streaming completion requests and
//! handling streaming completion responses.
//!
//! Provider implementations use these types to expose raw streamed completion
//! events without depending on a runtime.

mod accumulator;
mod block_id;
mod event;

use crate::completion::{CompletionError, CompletionResponse, Usage};
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

/// Shared pause flag plus the parked consumer's waker.
///
/// `AtomicWaker` holds a single waker, so this is correct only while one
/// task polls the stream — which `poll_next` taking `Pin<&mut Self>`
/// enforces. A design that shares one control across multiple streams must
/// switch to a multi-waiter primitive instead.
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

/// How the shared assembler treats an argument payload that does not parse as
/// JSON when a streamed tool call's input ends.
///
/// This is genuine wire-family policy, declared by the adapter on the end
/// event rather than hand-rolled per provider.
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

/// The provider's terminal stream record, normalized.
///
/// This replaces the provider-typed final payload that streams used to carry:
/// usage is a plain field rather than a trait method, and the finish reason is
/// normalized exactly as on the unary [`CompletionResponse`].
///
/// Providers that want their own terminal type keep it behind
/// their adapter's terminal mapping and serialize it onto [`StreamFinal::raw`].
///
/// # Emission contract
///
/// A terminal record is emitted only when the provider signaled genuine
/// completion — its own end-of-response event (an Anthropic `message_delta`
/// with a stop reason, an OpenAI `[DONE]` / `response.completed`, a Gemini
/// chunk carrying `finishReason`, and so on). Three failure shapes reach a
/// consumer, and they are distinct:
///
/// | Shape | `Err` item | Stream continues | Terminal record |
/// |---|---|---|---|
/// | Transport error (connection lost, HTTP failure) | yes | no | never |
/// | Malformed frame (recoverable parse error) | yes | yes | if a genuine terminal later arrives |
/// | Truncation (EOF without the provider's end event) | no | — | never |
///
/// On a terminal error (a transport failure or the provider's own failure
/// event), tool calls that were fully delivered before the failure are yielded
/// *before* the terminal `Err`; nothing follows the error — the stream then
/// ends without a terminal record.
///
/// Consequently an `Err` item is **not** by itself terminal: a malformed frame
/// is surfaced and the stream keeps consuming, so a later genuine terminal
/// still completes it. Consumers must drain the stream to `None` rather than
/// stop at the first `Err`, and must treat the absence of a terminal record as
/// truncation, never as a successful zero-usage completion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "StreamFinalRepr")]
pub struct StreamFinal {
    /// Token usage reported by the provider for this streamed completion.
    /// Zero-valued usage is the documented sentinel for missing metrics.
    pub usage: Usage,
    /// Why the model stopped generating, when the provider reported it.
    ///
    /// [`StreamingCompletionResponse`] applies
    /// [`FinishReason::reconcile_with_output`](crate::completion::FinishReason::reconcile_with_output)
    /// to this value using the tool calls actually seen on the stream, so a
    /// provider adapter does not need to (and cannot — it has no view of the
    /// preceding events).
    #[serde(default)]
    pub finish_reason: Option<crate::completion::FinishReason>,
    /// Provider-assigned *assistant message* ID, when available — only IDs the
    /// provider would recognize on a replayed assistant message. Response-scoped
    /// identifiers belong in [`StreamFinal::response_id`].
    #[serde(default)]
    pub message_id: Option<String>,
    /// Provider-assigned response-scoped ID, when available — e.g. an OpenAI
    /// chat `chatcmpl-` ID. Never replayed to a provider as a message ID.
    #[serde(default)]
    pub response_id: Option<String>,
    /// The provider's transport-level request identifier, taken from the SSE
    /// connection's HTTP response headers (Anthropic `request-id`, OpenAI/xAI
    /// `x-request-id`). When the source reconnected, this is the connection
    /// that delivered this terminal record. Never the body's message/response
    /// id. `None` means the provider did not report one — a documented
    /// outcome, never an error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// Stable descriptor name of the provider that produced this stream.
    pub provider: String,
    /// Provider-reported model identifier, when available.
    #[serde(default)]
    pub model: Option<String>,
    /// The provider's own terminal record for this stream, serialized by the
    /// adapter that mapped it. It is the terminal record as rig's wire type parsed it —
    /// fields that type does not model are not here — and it is the terminal
    /// record only, not the stream's frames; see the module docs for why
    /// frames are a separate mechanism. Every in-tree adapter populates it
    /// unconditionally.
    ///
    /// An escape hatch for provider-specific data rig does not normalize — it
    /// never replaces a normalized field, and every normalized field means the
    /// same thing whatever this holds. `Value::Null` means the record was
    /// built without a provider behind it — [`StreamFinal::new`] without
    /// `with_raw` (test doubles, hand-built records), or a record persisted
    /// before the field existed — never that the provider sent nothing: no
    /// stream that reached its terminal yields `Null` here.
    ///
    /// Typed access is recoverable: provider terminal types are
    /// `Deserialize`, so `provider::StreamingCompletionResponse::deserialize(&raw)`
    /// returns the provider's own type.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub raw: serde_json::Value,
}

impl StreamFinal {
    /// Create a terminal record for `provider` with `usage`; optional metadata
    /// starts unset and is filled in with the `with_*` helpers.
    pub fn new(provider: impl Into<String>, usage: Usage) -> Self {
        Self {
            usage,
            finish_reason: None,
            message_id: None,
            response_id: None,
            provider_request_id: None,
            provider: provider.into(),
            model: None,
            raw: serde_json::Value::Null,
        }
    }

    /// Attach the normalized finish reason.
    pub fn with_finish_reason(self, finish_reason: crate::completion::FinishReason) -> Self {
        self.with_optional_finish_reason(Some(finish_reason))
    }

    /// Attach the normalized finish reason when the provider reported one.
    pub fn with_optional_finish_reason(
        mut self,
        finish_reason: Option<crate::completion::FinishReason>,
    ) -> Self {
        self.finish_reason = finish_reason;
        self
    }

    /// This terminal record's identity metadata as one
    /// [`crate::completion::ResponseIdentity`] carrier.
    pub fn identity(&self) -> crate::completion::ResponseIdentity {
        crate::completion::ResponseIdentity {
            message_id: self.message_id.clone(),
            response_id: self.response_id.clone(),
            provider_request_id: self.provider_request_id.clone(),
        }
    }
}

crate::provider_response::response_metadata_setters!(StreamFinal);

/// Wire-shape mirror of [`StreamFinal`], used only for deserialization.
///
/// Serde must never construct an invariant-bearing value structurally: a plain
/// derive would let `"message_id":""` skip the empty-string filtering the
/// `with_*` setters apply. This mirror deserializes the exact wire shape and
/// [`From`] funnels it through
/// [`StreamFinal::new`] and the setters, so every deserialized value satisfies
/// the same invariants as a constructed one. Serialization stays derived on
/// [`StreamFinal`] itself, so the wire format is unchanged.
#[derive(Deserialize)]
struct StreamFinalRepr {
    usage: Usage,
    #[serde(default)]
    finish_reason: Option<crate::completion::FinishReason>,
    #[serde(default)]
    message_id: Option<String>,
    #[serde(default)]
    response_id: Option<String>,
    #[serde(default)]
    provider_request_id: Option<String>,
    provider: String,
    #[serde(default)]
    model: Option<String>,
    // `default` because persisted terminal records predate the field; a
    // missing key loads as `Null`, which is exactly what "no provider record
    // behind this value" means.
    #[serde(default)]
    raw: serde_json::Value,
}

impl From<StreamFinalRepr> for StreamFinal {
    fn from(repr: StreamFinalRepr) -> Self {
        let StreamFinalRepr {
            usage,
            finish_reason,
            message_id,
            response_id,
            provider_request_id,
            provider,
            model,
            raw,
        } = repr;
        Self::new(provider, usage)
            .with_optional_finish_reason(finish_reason)
            .with_optional_message_id(message_id)
            .with_optional_response_id(response_id)
            .with_optional_provider_request_id(provider_request_id)
            .with_optional_model(model)
            .with_raw(raw)
    }
}

/// An unmodeled wire payload on the raw passthrough channel.
///
/// Wraps the raw JSON with a **redacted** `Debug` (structural metadata only):
/// unmodeled frames can carry model output or other sensitive provider data,
/// and `warn!(?value)`-style Debug captures in streaming modules were a
/// recurring leak class a text scanner existed to police. With the payload
/// unable to Debug-print its content, that class is structurally closed for
/// the JSON channel — the redaction is a property of the type, not a
/// convention. Consumers who want the content opt in explicitly via
/// [`UnknownPayload::value`]; serialization
/// is `#[serde(transparent)]`, so wire round-trips are unchanged.
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
    /// Structural metadata only — never the payload.
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

/// A provider stream: the events an adapter emits, with in-band errors, as
/// consumed by [`StreamingCompletionResponse`].
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub type StreamingResult = Pin<Box<dyn Stream<Item = Result<StreamEvent, CompletionError>> + Send>>;

/// A provider stream, on browser wasm (no `Send`).
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub type StreamingResult = Pin<Box<dyn Stream<Item = Result<StreamEvent, CompletionError>>>>;

/// The response from a streaming completion request: the provider's events,
/// yielded as they arrive, and the aggregated choice they fold into.
///
/// Every yielded [`StreamEvent`] has already been applied to the
/// accumulator, so [`StreamingCompletionResponse::snapshot`] is always
/// consistent with the events seen so far, and a
/// [`StreamEvent::BlockEnd`] carries the block it finalized in `block`.
pub struct StreamingCompletionResponse {
    pub(crate) inner: Abortable<StreamingResult>,
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
    /// Whether the inner stream already ended: re-polling a drained stream
    /// — which `Stream` permits and combinators do — stays drained.
    finished: bool,
    /// The provider's normalized terminal record, `None` until the stream
    /// yields it (and forever on truncation or a terminal error).
    pub response: Option<StreamFinal>,
    /// Provider-assigned message ID (e.g. OpenAI Responses API `msg_` ID).
    pub message_id: Option<String>,
}

impl StreamingCompletionResponse {
    /// Wrap a provider stream and initialize aggregation state.
    ///
    /// `provider` is the stable descriptor name of the provider producing the
    /// stream; it is recorded up front so it is available even when the stream
    /// never reaches its terminal record.
    pub fn stream(provider: impl Into<String>, inner: StreamingResult) -> Self {
        let (abort_handle, abort_registration) = AbortHandle::new_pair();
        let abortable_stream = Abortable::new(inner, abort_registration);
        let pause_control = PauseControl::new();
        Self {
            inner: abortable_stream,
            abort_handle,
            pause_control,
            accumulator: BlockAccumulator::new(),
            provider: provider.into(),
            finished: false,
            response: None,
            message_id: None,
        }
    }

    /// Stable descriptor name of the provider producing this stream.
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// The aggregated choice so far: every block the events yielded to this
    /// point have opened, in arrival order. Non-destructive — two snapshots
    /// are equal and neither changes what [`Self::finish`] returns.
    pub fn snapshot(&self) -> Vec<AssistantContent> {
        self.accumulator.snapshot()
    }

    /// Consume the stream into the unary response shape: the aggregated
    /// choice, the terminal record's usage and metadata. Usage is the zero
    /// sentinel (`Usage::new`) when the stream produced no terminal record;
    /// `provider` comes from the stream itself, so it is populated even then.
    ///
    /// Events not yet polled are not part of the choice: drain the stream
    /// first when the whole turn is wanted.
    pub fn finish(mut self) -> CompletionResponse {
        let choice = self.accumulator.finish();
        let terminal = self.response.as_ref();
        CompletionResponse::new(
            choice,
            terminal.map(|response| response.usage).unwrap_or_default(),
            self.provider.clone(),
        )
        // An explicit message-id block outranks the terminal record's ID.
        .with_optional_message_id(
            self.message_id
                .clone()
                .or_else(|| terminal.and_then(|response| response.message_id.clone())),
        )
        .with_optional_response_id(terminal.and_then(|response| response.response_id.clone()))
        .with_optional_provider_request_id(
            terminal.and_then(|response| response.provider_request_id.clone()),
        )
        .with_optional_finish_reason(terminal.and_then(|response| response.finish_reason.clone()))
        .with_optional_model(terminal.and_then(|response| response.model.clone()))
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
        let empty: StreamingResult = Box::pin(futures::stream::poll_fn(|_| Poll::Ready(None)));
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

    /// Token usage reported by the provider for this response.
    ///
    /// Returns the usage carried by the final response once the stream has
    /// produced it. Until then — or when the provider does not report streamed
    /// usage — this returns [`Usage::new`], the zero-valued sentinel for missing
    /// usage metrics.
    pub fn usage(&self) -> Usage {
        self.response
            .as_ref()
            .map(|response| response.usage)
            .unwrap_or_default()
    }

    /// This stream's identity metadata as one
    /// [`crate::completion::ResponseIdentity`] carrier.
    ///
    /// The message id is read from the stream rather than the terminal record:
    /// an explicit `MessageId` event outranks the terminal's id, and the
    /// terminal record backfills the field when the stream never saw one. The
    /// response-scoped and transport ids exist only on the terminal record, so
    /// they stay `None` for a stream that ended without one.
    pub fn identity(&self) -> crate::completion::ResponseIdentity {
        crate::completion::ResponseIdentity {
            message_id: self.message_id.clone(),
            ..self
                .response
                .as_ref()
                .map(StreamFinal::identity)
                .unwrap_or_default()
        }
    }
}

impl Stream for StreamingCompletionResponse {
    type Item = Result<StreamEvent, CompletionError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let stream = self.get_mut();

        // A drained stream stays drained (#2258 H6).
        if stream.finished {
            return Poll::Ready(None);
        }

        if stream.is_paused() {
            // Park rather than re-waking immediately: a self-wake turns a
            // pause into a busy poll loop that burns the executor for as long
            // as the consumer stays paused (#2258 H7). Register-then-recheck
            // is the `AtomicWaker` protocol that also closes the resume race:
            // `resume` clears the flag before waking, and this poll registers
            // its waker before re-reading the flag, so a resume racing this
            // branch either sees the registered waker (and wakes the task) or
            // is observed by the re-check below.
            stream.pause_control.state.waker.register(cx.waker());
            if stream.is_paused() {
                return Poll::Pending;
            }
        }

        // Non-yielding events (duplicate terminals) loop rather than recurse
        // — a long run of them must not grow the stack (#2258 review P3).
        loop {
            return match Pin::new(&mut stream.inner).poll_next(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(None) => {
                    stream.finished = true;
                    Poll::Ready(None)
                }
                // Every error reaches the consumer. Cancellation is *not* an
                // error here: `cancel()` aborts through `Abortable`, which
                // terminates the inner stream with `Ready(None)` above, so
                // the aggregated choice is finished normally.
                Poll::Ready(Some(Err(err))) => Poll::Ready(Some(Err(err))),
                Poll::Ready(Some(Ok(event))) => match event {
                    StreamEvent::BlockStart {
                        id,
                        kind: BlockKind::Message,
                    } => {
                        // The wire announced the assistant message's own id;
                        // it outranks the terminal record's.
                        if let Some(message_id) = id.wire_str() {
                            stream.message_id = Some(message_id.to_owned());
                        }
                        Poll::Ready(Some(Ok(StreamEvent::BlockStart {
                            id,
                            kind: BlockKind::Message,
                        })))
                    }
                    StreamEvent::Final(mut response) => {
                        // A second terminal is a provider defect; the first
                        // one latched.
                        if stream.response.is_some() {
                            continue;
                        }
                        // Finish-reason reconciliation against the
                        // accumulator's authoritative view of completed calls:
                        // the streaming counterpart of the unary path's, so
                        // both agree about a `stop` that was really a tool
                        // call.
                        response.finish_reason = response.finish_reason.map(|reason| {
                            reason.reconcile_with_output(stream.accumulator.saw_tool_call())
                        });
                        // An explicit message-id block keeps precedence; the
                        // terminal record only fills a gap.
                        if stream.message_id.is_none() {
                            stream.message_id.clone_from(&response.message_id);
                        }
                        stream.response = Some(response.clone());
                        Poll::Ready(Some(Ok(StreamEvent::Final(response))))
                    }
                    StreamEvent::Unknown(value) => {
                        // Passed straight through; never folded into the
                        // aggregated choice (no `AssistantContent::Unknown`).
                        Poll::Ready(Some(Ok(StreamEvent::Unknown(value))))
                    }
                    event => match stream.accumulator.apply(&event) {
                        // A block end that finalized a block publishes it
                        // under the key its deltas carried.
                        Ok(Some((id, block))) => {
                            let StreamEvent::BlockEnd { end, .. } = event else {
                                // Only ends finalize; the accumulator upholds it.
                                return Poll::Ready(Some(Ok(event)));
                            };
                            Poll::Ready(Some(Ok(StreamEvent::BlockEnd {
                                id,
                                end,
                                block: Some(block),
                            })))
                        }
                        Ok(None) => Poll::Ready(Some(Ok(event))),
                        // Malformed complete input surfaces in-band; the stream
                        // keeps consuming, matching the malformed-frame contract.
                        Err(err) => Poll::Ready(Some(Err(err))),
                    },
                },
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
