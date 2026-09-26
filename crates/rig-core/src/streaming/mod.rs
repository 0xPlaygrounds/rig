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

use crate::completion::{CompletionResponse, Usage};
use crate::driver::{Step, record_request_id};
use crate::error::ErrorReport;
use crate::error::ProviderError;
use crate::message::{AssistantContent, ToolResult};
use crate::operation::{Canonical, Completion, CompletionFold};
use crate::wasm_compat::WasmBoxedStream;
use crate::wire::{Fold, Mode, Operation, Reply};
pub use block_id::{BlockId, MintKind, SyntheticIds, non_empty_id};
pub use event::{BlockClose, BlockKind, Delta, StreamEvent, ToolCallEnd};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::task::{Context, Poll};

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

/// The one completion stream item type: what a [`CompletionStream`]
/// yields, what a [`CompletionFold`] collects, what the bus carries.
pub type StreamEvents =
    crate::wasm_compat::WasmBoxedStream<'static, Result<StreamEvent, ErrorReport>>;

/// One reply as it arrives: its canonical events, and the fold that has
/// seen each of them. It is what [`Model::stream`](crate::Model::stream)
/// returns for every operation, and what a call drains.
///
/// Each event is stamped with what the transport reported about the reply
/// (a completion's terminal record gets the request id), recorded on the
/// call's span when streamed, and absorbed by the fold before it is
/// yielded. A failure is an in-band error item. Stop polling to pause; drop
/// the stream to cancel.
pub struct Streamed<Op: Operation> {
    steps: WasmBoxedStream<'static, Result<Step<Op>, ProviderError>>,
    fold: Op::Fold,
    mode: Mode,
    span: tracing::Span,
    /// What the transport reported so far: the provider, the latest page's
    /// request id, and once the reply closed, its document.
    reply: Reply,
    finished: bool,
}

/// A streamed completion.
pub type CompletionStream = Streamed<Completion>;

impl<Op: Operation> Streamed<Op> {
    /// The reply `steps` deliver under `span`, folded by `fold`.
    pub(crate) fn new(
        steps: WasmBoxedStream<'static, Result<Step<Op>, ProviderError>>,
        fold: Op::Fold,
        mode: Mode,
        span: tracing::Span,
        provider: impl Into<String>,
    ) -> Self {
        Self {
            steps,
            fold,
            mode,
            span,
            reply: Reply {
                provider: provider.into(),
                raw: serde_json::Value::Null,
                provider_request_id: None,
            },
            finished: false,
        }
    }

    /// What the fold has seen so far.
    pub fn folded(&self) -> &Op::Fold {
        &self.fold
    }

    /// The response the events seen so far fold into. Events not yet polled
    /// are not part of it. A unary call's response is recorded on its span.
    pub fn finish(self) -> Result<Op::Response, ProviderError> {
        let response = self.fold.finish(self.reply)?;
        if self.mode == Mode::Unary {
            Op::record(&self.span, &response);
        }
        Ok(response)
    }

    /// Poll to the end, then finish: a call's response. The first error
    /// fails it.
    pub(crate) async fn drain(mut self) -> Result<Op::Response, ProviderError> {
        while let Some(item) = futures::future::poll_fn(|cx| self.poll_step(cx)).await {
            item?;
        }
        self.finish()
    }

    /// The one place a step is consumed: an opened page's request id goes
    /// on the span, an event is stamped, recorded and absorbed, an error is
    /// stamped with the request id, and the closed reply is kept.
    fn poll_step(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Op::Event, ProviderError>>> {
        while !self.finished {
            let step = match self.steps.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    self.finished = true;
                    break;
                }
                Poll::Ready(Some(step)) => step,
            };
            match step {
                Ok(Step::Opened(page)) => {
                    record_request_id(&self.span, page.request_id.as_deref());
                    self.reply.provider_request_id = page.request_id;
                }
                Ok(Step::Event(mut event)) => {
                    Op::stamp_event(&mut event, &self.reply);
                    if self.mode == Mode::Streaming {
                        Op::record_event(&self.span, &event);
                    }
                    return Poll::Ready(Some(self.fold.absorb(&event).map(|()| event)));
                }
                Ok(Step::Closed(reply)) => self.reply = reply,
                Err(error) => {
                    // An id an upstream constructor already attached wins:
                    // it saw the reply.
                    let error =
                        error.with_provider_request_id(self.reply.provider_request_id.clone());
                    record_request_id(&self.span, error.provider_request_id());
                    return Poll::Ready(Some(Err(error)));
                }
            }
        }
        Poll::Ready(None)
    }
}

impl Streamed<Completion> {
    /// A stream relayed over the bus under `label`, whose terminal record
    /// names the provider behind it. The events pass the completion sink,
    /// which leaves a stream the origin's sink made canonical as it is and
    /// makes any other stream canonical.
    pub fn relay(label: impl Into<String>, events: StreamEvents) -> Self {
        let label = label.into();
        let steps = async_stream::stream! {
            let mut events = events;
            let mut canonical = Canonical::default();
            while let Some(item) = events.next().await {
                let mut emitted = Vec::new();
                canonical.push(
                    item.map_err(|report| ProviderError::Relayed(Box::new(report))),
                    &mut |item, _| emitted.push(item),
                );
                for item in emitted {
                    yield item.map(Step::Event);
                }
            }
            let mut emitted = Vec::new();
            canonical.finish(&mut |item, _| emitted.push(item));
            for item in emitted {
                yield item.map(Step::Event);
            }
        };
        Self::new(
            Box::pin(steps),
            CompletionFold::relayed(label.clone()),
            Mode::Streaming,
            tracing::Span::none(),
            label,
        )
    }
}

// Nothing is pinned in place: the steps are boxed and the fold is only
// ever reached through `&mut`.
impl<Op: Operation> Unpin for Streamed<Op> {}

impl<Op: Operation> Stream for Streamed<Op> {
    type Item = Result<Op::Event, ErrorReport>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut()
            .poll_step(cx)
            .map(|item| item.map(|item| item.map_err(|error| ErrorReport::from(&error))))
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
