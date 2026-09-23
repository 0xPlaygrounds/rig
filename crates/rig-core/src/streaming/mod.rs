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

use crate::completion::{CompletionResponse, Usage};
use crate::error::ErrorReport;
use crate::error::ProviderError;
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
    pub response: &'a mut Option<StreamFinal>,
    pub message_id: &'a mut Option<String>,
    pub provider: &'a mut String,
    /// Whether the terminal record names the provider (a stream that came
    /// over the bus) rather than the opener.
    pub provider_from_terminal: bool,
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
            if let Some(message_id) = id.wire_str() {
                *step.message_id = Some(message_id.to_owned());
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
            // An explicit message-id block keeps precedence; the terminal
            // record only fills a gap.
            if step.message_id.is_none() {
                step.message_id.clone_from(&response.message_id);
            }
            if step.provider_from_terminal && !response.provider.is_empty() {
                step.provider.clone_from(&response.provider);
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
/// stamped with `issuer`, plus the terminal record's usage and metadata,
/// carrying `raw` as the provider's document for the turn. Usage reports no
/// counter when the reply produced no terminal record.
pub(crate) fn fold_finish(
    mut accumulator: BlockAccumulator,
    terminal: Option<&StreamFinal>,
    message_id: Option<String>,
    provider: String,
    issuer: &str,
    raw: serde_json::Value,
) -> CompletionResponse {
    let choice = stamp_reasoning(accumulator.finish(), issuer);
    CompletionResponse::new(
        choice,
        terminal.map(|response| response.usage).unwrap_or_default(),
        provider,
        raw,
    )
    // An explicit message-id block outranks the terminal record's ID.
    .with_optional_message_id(
        message_id.or_else(|| terminal.and_then(|response| response.message_id.clone())),
    )
    .with_optional_response_id(terminal.and_then(|response| response.response_id.clone()))
    .with_optional_provider_request_id(
        terminal.and_then(|response| response.provider_request_id.clone()),
    )
    .with_optional_finish_reason(terminal.and_then(|response| response.finish_reason.clone()))
    .with_optional_model(terminal.and_then(|response| response.model.clone()))
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

/// Normalized terminal record, emitted only after provider-signaled completion.
/// EOF without a terminal record is truncation, not a successful completion.
///
/// Recoverable malformed frames yield errors and allow subsequent events.
/// Transport or provider terminal failures yield already-completed tool calls
/// before the final error, then end without a terminal record. Consumers must
/// drain to `None` rather than treating every error item as terminal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "StreamFinalRepr")]
pub struct StreamFinal {
    /// Token usage reported by the provider for this streamed completion.
    /// A counter the provider did not report is `None`.
    pub usage: Usage,
    /// Provider-reported finish reason. [`StreamingCompletionResponse`] reconciles
    /// it with the completed tool calls before yielding the terminal event.
    #[serde(default)]
    pub finish_reason: Option<crate::completion::FinishReason>,
    /// Provider-assigned assistant message ID suitable for replay.
    /// Response-scoped identifiers belong in [`Self::response_id`].
    #[serde(default)]
    pub message_id: Option<String>,
    /// Provider-assigned response ID. Must not be replayed as a message ID.
    #[serde(default)]
    pub response_id: Option<String>,
    /// Request identifier from the HTTP headers of the connection delivering
    /// this terminal record, including after reconnects. `None` if unreported;
    /// never the body's message or response ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// Stable descriptor name of the provider that produced this stream.
    pub provider: String,
    /// The service whose reasoning this stream carries, when it is not
    /// [`Self::provider`]: a transport or deployment of another provider's
    /// models. The stream's reasoning records it as its issuer
    /// ([`crate::message::Reasoning::provider`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_issuer: Option<String>,
    /// Provider-reported model identifier, when available.
    #[serde(default)]
    pub model: Option<String>,
    /// Required provider terminal document serialized from the adapter's parsed
    /// wire type, not a transcript of frames. Unmodeled fields may be absent.
    /// This metadata does not override normalized fields and can be deserialized
    /// into the corresponding provider terminal type.
    pub raw: serde_json::Value,
}

impl StreamFinal {
    /// Create a terminal record for `provider` with `usage` and the
    /// provider's own terminal document `raw` (see [`Self::raw`]); optional
    /// metadata starts unset and is filled in with the `with_*` helpers.
    pub fn new(provider: impl Into<String>, usage: Usage, raw: serde_json::Value) -> Self {
        Self {
            usage,
            finish_reason: None,
            message_id: None,
            response_id: None,
            provider_request_id: None,
            provider: provider.into(),
            reasoning_issuer: None,
            model: None,
            raw,
        }
    }

    /// Name the service whose reasoning this stream carries; see
    /// [`Self::reasoning_issuer`].
    pub fn with_reasoning_issuer(mut self, issuer: impl Into<String>) -> Self {
        self.reasoning_issuer = Some(issuer.into());
        self
    }

    /// The issuer this stream's reasoning records: [`Self::reasoning_issuer`]
    /// when set, otherwise [`Self::provider`].
    pub fn issuer(&self) -> &str {
        self.reasoning_issuer.as_deref().unwrap_or(&self.provider)
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

/// Deserialization shape routed through setters to normalize empty identifiers.
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
    reasoning_issuer: Option<String>,
    #[serde(default)]
    model: Option<String>,
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
            reasoning_issuer,
            model,
            raw,
        } = repr;
        let mut terminal = Self::new(provider, usage, raw)
            .with_optional_finish_reason(finish_reason)
            .with_optional_message_id(message_id)
            .with_optional_response_id(response_id)
            .with_optional_provider_request_id(provider_request_id)
            .with_optional_model(model);
        terminal.reasoning_issuer = reasoning_issuer;
        terminal
    }
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
    /// name the same issuer ([`StreamFinal::with_reasoning_issuer`]): once
    /// it arrives, it is the one read.
    pub fn with_reasoning_issuer(mut self, issuer: impl Into<String>) -> Self {
        self.reasoning_issuer = Some(issuer.into());
        self
    }

    /// The issuer this stream's reasoning records: the terminal record's
    /// [`StreamFinal::issuer`] once it has arrived; before it, the issuer
    /// named up front, else the provider that opened the stream. `None`
    /// before the terminal of a stream rebuilt from events
    /// ([`Self::from_events`]), whose opening label names a handler, not an
    /// issuer: its reasoning is then of unknown provenance.
    pub fn reasoning_issuer(&self) -> Option<&str> {
        match &self.response {
            Some(terminal) => Some(terminal.issuer()),
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
    /// choice, the terminal record's usage and metadata, and the terminal
    /// record's document as `raw`. A stream that produced no terminal
    /// record is truncated per the emission contract and is refused: there
    /// is no document to build a response from.
    ///
    /// Events not yet polled are not part of the choice: drain the stream
    /// first when the whole turn is wanted.
    pub fn finish(self) -> Result<CompletionResponse, ProviderError> {
        let Some(terminal) = self.response.as_ref() else {
            return Err(ProviderError::Response(
                "provider stream ended without a terminal record; treating the turn as truncated"
                    .to_owned(),
            ));
        };
        let issuer = terminal.issuer().to_owned();
        Ok(fold_finish(
            self.accumulator,
            Some(terminal),
            self.message_id.clone(),
            self.provider.clone(),
            &issuer,
            terminal.raw.clone(),
        ))
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
            .map(|response| response.usage)
            .unwrap_or_default()
    }

    /// Returns response identity. A message-start ID takes precedence over the
    /// terminal message ID. Response and transport IDs require a terminal record.
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
                        provider: &mut stream.provider,
                        provider_from_terminal: stream.provider_from_terminal,
                    };
                    match absorb(step, event) {
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
