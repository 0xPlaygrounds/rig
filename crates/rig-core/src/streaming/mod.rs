//! Runtime-independent completion events, and [`Streamed`], the one stream
//! every reply arrives as.
//!
//! ```
//! use rig_core::streaming::{StreamEvent, StreamFinal};
//! use rig_core::completion::Usage;
//!
//! let terminal = StreamEvent::Final(StreamFinal::new("mock", Usage::default(), serde_json::Value::Null));
//! assert!(matches!(terminal, StreamEvent::Final(_)));
//! ```

mod block_id;
mod event;
mod update;

use crate::completion::{CompletionResponse, Usage};
use crate::driver::{Progress, Source, record_request_id};
use crate::error::ErrorReport;
use crate::error::ProviderError;
use crate::message::{AssistantContent, ToolResult};
use crate::operation::{Completion, CompletionReply};
use crate::wire::{End, Fold, Operation, Ready, Reply};
pub use block_id::{BlockId, MintKind, SyntheticIds, non_empty_id};
pub use event::{BlockClose, BlockKind, Delta, StreamEvent, ToolCallEnd};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::task::{Context, Poll};
pub use update::{PartKind, Update};

/// Seal every reasoning part of `choice` to `issuer`, the service that
/// issued this reply's reasoning.
pub fn stamp_reasoning(choice: Vec<AssistantContent>, issuer: &str) -> Vec<AssistantContent> {
    choice
        .into_iter()
        .map(|part| match part {
            AssistantContent::Reasoning(reasoning) => {
                AssistantContent::Reasoning(reasoning.reseal(issuer.to_owned()))
            }
            part => part,
        })
        .collect()
}

/// The folded completion response: the collected choice, its reasoning
/// stamped with `issuer`, plus the terminal record's usage and metadata,
/// carrying `raw` as the provider's document for the turn. Usage reports no
/// counter when the reply produced no terminal record.
pub(crate) fn fold_finish(
    choice: Vec<AssistantContent>,
    terminal: Option<&StreamFinal>,
    message_id: Option<String>,
    provider: String,
    issuer: &str,
    raw: serde_json::Value,
) -> CompletionResponse {
    let choice = stamp_reasoning(choice, issuer);
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
    /// Provider-reported finish reason. The completion sink reconciles it
    /// with the completed tool calls before the terminal event leaves it.
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
    /// models. The stream's reasoning is sealed to it
    /// ([`crate::message::Sealed::issuer`]).
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

/// The one completion stream item type: what a [`CompletionStream`]
/// yields, what a [`CompletionFold`](crate::operation::CompletionFold) collects, what the bus carries.
pub type StreamEvents =
    crate::wasm_compat::WasmBoxedStream<'static, Result<StreamEvent, ErrorReport>>;

/// One reply as it arrives: its canonical events, and the fold that has
/// seen each of them. It is what [`Model::stream`](crate::Model::stream)
/// returns for every operation, and what a call drains.
///
/// Every item passes the operation's fold on its way out: a decoder's
/// items are made canonical as they are pushed, and each event is absorbed
/// by the fold before it is yielded. A failure is an in-band error item.
/// Stop polling to pause; drop the stream to cancel.
pub struct Streamed<Op: Operation> {
    source: Box<dyn Source<Op>>,
    fold: Op::Fold,
    /// Items the source pushed that the consumer has not taken.
    ready: Ready<Op>,
    span: tracing::Span,
    /// What the driver learned about the reply once it closed.
    reply: Reply,
    finished: bool,
    /// Where [`Streamed::updates`] stopped, so a later call continues it.
    projection: update::Projection,
}

/// A streamed completion.
pub type CompletionStream = Streamed<Completion>;

impl<Op: Operation> Streamed<Op> {
    /// The reply `source` delivers under `span`, through `fold`.
    pub(crate) fn new(
        source: Box<dyn Source<Op>>,
        fold: Op::Fold,
        span: tracing::Span,
        provider: impl Into<String>,
    ) -> Self {
        Self {
            source,
            fold,
            ready: Ready::default(),
            span,
            reply: Reply {
                provider: provider.into(),
                raw: serde_json::Value::Null,
                provider_request_id: None,
            },
            finished: false,
            projection: update::Projection::default(),
        }
    }

    /// What the fold has seen so far.
    pub fn folded(&self) -> &Op::Fold {
        &self.fold
    }

    /// The response the events seen so far fold into. Events not yet polled
    /// are not part of it.
    pub fn finish(self) -> Result<Op::Response, ProviderError> {
        self.fold.finish(self.reply)
    }

    /// Poll to the end, then finish: a call's response. The first error
    /// fails it.
    pub(crate) async fn drain(mut self) -> Result<Op::Response, ProviderError> {
        while let Some(item) = futures::future::poll_fn(|cx| self.poll_step(cx)).await {
            item?;
        }
        self.finish()
    }

    /// The request path of the reply being read, when the transport named
    /// one.
    pub(crate) fn route(&self) -> &str {
        self.source.route()
    }

    /// The one place an item is consumed: an event is absorbed by the fold
    /// before it leaves, and an error is stamped with the request id.
    pub(crate) fn poll_step(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Op::Event, ProviderError>>> {
        loop {
            if let Some(item) = self.ready.pop() {
                return Poll::Ready(Some(match item {
                    Ok(event) => self.fold.absorb(&event).map(|()| event),
                    Err(error) => {
                        // An id an upstream constructor already attached
                        // wins: it saw the reply.
                        let error = error
                            .with_provider_request_id(self.ready.request_id().map(str::to_owned));
                        record_request_id(&self.span, error.provider_request_id());
                        Err(error)
                    }
                }));
            }
            if self.finished {
                return Poll::Ready(None);
            }
            let progress = self.source.poll_into(cx, &mut self.fold, &mut self.ready);
            // What the transport reported so far: the latest reply's request
            // id, before the reply closes.
            if let Some(request_id) = self.ready.request_id() {
                self.reply.provider_request_id = Some(request_id.to_owned());
            }
            match progress {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Progress::Pushed) => {}
                Poll::Ready(Progress::Closed(reply)) => {
                    self.reply = reply;
                    self.finished = true;
                }
            }
        }
    }
}

/// A relayed stream's items, pushed through the fold as they arrive.
struct Relayed {
    events: StreamEvents,
    ended: bool,
    label: String,
}

impl Source<Completion> for Relayed {
    fn poll_into(
        &mut self,
        cx: &mut Context<'_>,
        fold: &mut CompletionReply,
        ready: &mut Ready<Completion>,
    ) -> Poll<Progress> {
        if self.ended {
            return Poll::Ready(Progress::Closed(Reply {
                provider: self.label.clone(),
                raw: serde_json::Value::Null,
                provider_request_id: None,
            }));
        }
        match self.events.as_mut().poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(item)) => {
                fold.push(
                    item.map_err(|report| ProviderError::Relayed(Box::new(report))),
                    ready,
                );
                fold.settle(ready, None);
                Poll::Ready(Progress::Pushed)
            }
            Poll::Ready(None) => {
                self.ended = true;
                fold.settle(ready, Some(End::Eof));
                Poll::Ready(Progress::Pushed)
            }
        }
    }
}

impl Streamed<Completion> {
    /// A stream relayed over the bus under `label`, whose terminal record
    /// names the provider behind it. The events pass the completion fold's
    /// canonicalization, which leaves a stream the origin made canonical as
    /// it is. Any other stream gains its closes and blocks, and loses a
    /// second terminal. An end that carries its block keeps it.
    ///
    /// Items are yielded as they arrive, so closes the stream's end adds
    /// follow an error already yielded. A relay cannot tell a terminal
    /// failure from an in-band one without waiting for the next item, and
    /// holding an error back would stall it behind an origin that never
    /// closes its stream.
    pub fn relay(label: impl Into<String>, events: StreamEvents) -> Self {
        let label = label.into();
        Self::new(
            Box::new(Relayed {
                events,
                ended: false,
                label: label.clone(),
            }),
            CompletionReply::relayed(label.clone()),
            tracing::Span::none(),
            label,
        )
    }
}

impl Streamed<Completion> {
    /// The parts of the response as they start, grow and finish, then the
    /// response. See [`Update`] for the order and the index contract. An
    /// error is an item, as on the event stream; a stream that ends without
    /// its terminal record ends with that error instead of [`Update::Done`].
    ///
    /// A later call, or [`Self::text`], continues where an earlier one
    /// stopped. Events read from the stream itself in between are not
    /// projected as they pass: a part they touched is sent whole, or the
    /// rest of it at its end, so the contract still holds.
    ///
    /// ```no_run
    /// use futures::StreamExt;
    /// use rig_core::streaming::Update;
    ///
    /// # async fn run(model: rig_core::DynModel<rig_core::operation::Completion>) -> Result<(), Box<dyn std::error::Error>> {
    /// let mut stream = model.stream("Tell me a story.")?;
    /// let mut updates = stream.updates();
    /// while let Some(update) = updates.next().await {
    ///     if let Update::Delta { text, .. } = update? {
    ///         print!("{text}");
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn updates(&mut self) -> impl Stream<Item = Result<Update, ErrorReport>> + '_ {
        futures::stream::poll_fn(move |cx| {
            if std::mem::take(&mut self.projection.stale) {
                self.projection
                    .projector
                    .catch_up(&self.fold, &mut self.projection.queue);
            }
            loop {
                if let Some(item) = self.projection.queue.pop_front() {
                    return Poll::Ready(Some(item));
                }
                if self.projection.ended {
                    return Poll::Ready(None);
                }
                match self.poll_step(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Some(Ok(event))) => self.projection.projector.push(
                        &event,
                        &self.fold,
                        &mut self.projection.queue,
                    ),
                    Poll::Ready(Some(Err(error))) => {
                        self.projection.failed = true;
                        self.projection
                            .queue
                            .push_back(Err(ErrorReport::from(&error)));
                    }
                    Poll::Ready(None) => {
                        self.projection.ended = true;
                        self.projection
                            .projector
                            .finish(&self.fold, &mut self.projection.queue);
                        match self.fold.streamed_response(&self.reply) {
                            Ok(response) => {
                                self.projection.queue.push_back(Ok(Update::Done(response)));
                            }
                            Err(error) if !self.projection.failed => {
                                self.projection
                                    .queue
                                    .push_back(Err(ErrorReport::from(&error)));
                            }
                            Err(_) => {}
                        }
                    }
                }
            }
        })
    }

    /// The text deltas of the response, in order; [`Self::updates`] without
    /// reasoning, tool calls or the parts' boundaries.
    pub fn text(&mut self) -> impl Stream<Item = Result<String, ErrorReport>> + '_ {
        let mut text_parts = std::collections::HashSet::new();
        self.updates().filter_map(move |update| {
            std::future::ready(match update {
                Ok(Update::Start {
                    index,
                    part: PartKind::Text,
                }) => {
                    text_parts.insert(index);
                    None
                }
                Ok(Update::Delta { index, text }) if text_parts.contains(&index) => Some(Ok(text)),
                Ok(_) => None,
                Err(error) => Some(Err(error)),
            })
        })
    }

    /// What arrived so far: every part that ended, and the usage reported so
    /// far. Valid after an error, and after the caller stopped polling.
    pub fn partial(&self) -> CompletionResponse {
        self.fold.partial(&self.reply)
    }
}

// Nothing is pinned in place: the steps are boxed and the fold is only
// ever reached through `&mut`.
impl<Op: Operation> Unpin for Streamed<Op> {}

impl<Op: Operation> Stream for Streamed<Op> {
    type Item = Result<Op::Event, ErrorReport>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let item = this.poll_step(cx);
        if matches!(item, Poll::Ready(Some(_))) {
            this.projection.stale = true;
        }
        item.map(|item| item.map(|item| item.map_err(|error| ErrorReport::from(&error))))
    }
}

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
