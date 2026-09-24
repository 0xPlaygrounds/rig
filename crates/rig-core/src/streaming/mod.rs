//! Runtime-independent completion events and the stream that folds them.
//!
//! ```
//! use rig_core::streaming::{StreamEvent, StreamFinal};
//! use rig_core::completion::Usage;
//!
//! let terminal = StreamEvent::Final(StreamFinal::new("mock", Usage::default(), serde_json::Value::Null));
//! assert!(matches!(terminal, StreamEvent::Final(_)));
//! ```

mod accumulator;
mod block_id;
mod event;

use crate::completion::{CompletionResponse, Usage};
use crate::error::ErrorReport;
use crate::error::ProviderError;
use crate::message::AssistantContent;
use crate::operation::CompletionFold;
pub use accumulator::BlockAccumulator;
pub use block_id::{BlockId, MintKind, SyntheticIds, non_empty_id};
pub use event::{BlockClose, BlockKind, Delta, StreamEvent, ToolCallEnd};
use futures::Stream;
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
    /// Provider-reported finish reason. [`CompletionStream`] reconciles
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

/// The one stream item type: what [`CompletionStream`] yields, what the
/// accumulator applies, what the bus carries.
pub type StreamEvents =
    crate::wasm_compat::WasmBoxedStream<'static, Result<StreamEvent, ErrorReport>>;

/// A completion reply's events after the fold step, and the fold itself.
///
/// It yields each event as [`CompletionFold`] rewrote it: a closing
/// [`StreamEvent::BlockEnd`] carries its finalized block, the terminal's
/// finish reason is reconciled with the completed tool calls, duplicate
/// terminals are dropped, and a malformed block surfaces as an in-band
/// [`ErrorReport`]. Stop polling to pause; drop the stream to cancel.
pub struct CompletionStream {
    events: StreamEvents,
    fold: CompletionFold,
    finished: bool,
}

impl CompletionStream {
    /// The stream of a reply `provider` opened. Provider errors become
    /// [`ErrorReport`]s here, the one place a stream's error half changes.
    pub fn new(
        provider: impl Into<String>,
        events: impl Stream<Item = Result<StreamEvent, ProviderError>>
        + crate::wasm_compat::WasmCompatSend
        + 'static,
    ) -> Self {
        use futures::StreamExt as _;

        let events = events.map(|item| item.map_err(|error| ErrorReport::from(&error)));
        Self::opened(CompletionFold::opened(provider), Box::pin(events))
    }

    /// A stream relayed under `label`, whose terminal record names the
    /// provider behind it. Until that record arrives, its reasoning has no
    /// known issuer.
    pub fn relay(label: impl Into<String>, events: StreamEvents) -> Self {
        Self::opened(CompletionFold::relayed(label), events)
    }

    pub(crate) fn opened(fold: CompletionFold, events: StreamEvents) -> Self {
        Self {
            events,
            fold,
            finished: false,
        }
    }

    /// Name the issuer of this stream's reasoning up front, for a transport
    /// or deployment of another provider's models, so a partial turn taken
    /// before the terminal record records it too. The terminal record must
    /// name the same issuer ([`StreamFinal::with_reasoning_issuer`]): once
    /// it arrives, it is the one read.
    pub fn with_reasoning_issuer(mut self, issuer: impl Into<String>) -> Self {
        self.fold.name_reasoning_issuer(issuer.into());
        self
    }

    /// Stable descriptor name of the provider producing this stream: the
    /// opener's, or the terminal record's for a relayed stream.
    pub fn provider(&self) -> &str {
        self.fold.provider()
    }

    /// The issuer this stream's reasoning records: the terminal record's
    /// once it has arrived; before it, the issuer named up front, else the
    /// provider that opened the stream. `None` before the terminal of a
    /// relayed stream, whose label names a handler, not an issuer.
    pub fn reasoning_issuer(&self) -> Option<&str> {
        self.fold.reasoning_issuer()
    }

    /// The provider's normalized terminal record, `None` until the stream
    /// yields it (and forever on truncation or a terminal error).
    pub fn terminal(&self) -> Option<&StreamFinal> {
        self.fold.terminal()
    }

    /// The provider-assigned message id, from a message block or the
    /// terminal record.
    pub fn message_id(&self) -> Option<&str> {
        self.fold.message_id()
    }

    /// The accumulated choice so far. See [`BlockAccumulator::snapshot`] for
    /// unfinished and empty-part handling.
    pub fn snapshot(&self) -> Vec<AssistantContent> {
        self.fold.snapshot()
    }

    /// Terminal usage, or [`Usage::default`] before a terminal record.
    /// Unreported counters remain `None`.
    pub fn usage(&self) -> Usage {
        self.fold.usage()
    }

    /// Response identity. A message-start id takes precedence over the
    /// terminal's; response and transport ids require a terminal record.
    pub fn identity(&self) -> crate::completion::ResponseIdentity {
        self.fold.identity()
    }

    /// The assembled turn: the aggregated choice, the terminal record's
    /// usage and metadata, and its document as `raw`. A stream that yielded
    /// no terminal record is truncated and is refused. Events not yet
    /// polled are not part of it.
    pub fn finish(self) -> Result<CompletionResponse, ProviderError> {
        self.fold.finish_stream()
    }
}

impl Stream for CompletionStream {
    type Item = Result<StreamEvent, ErrorReport>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let stream = self.get_mut();
        if stream.finished {
            return Poll::Ready(None);
        }
        // Loop over dropped duplicate terminals without growing the stack.
        loop {
            return match stream.events.as_mut().poll_next(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(None) => {
                    stream.finished = true;
                    Poll::Ready(None)
                }
                Poll::Ready(Some(Err(error))) => Poll::Ready(Some(Err(error))),
                Poll::Ready(Some(Ok(event))) => match stream.fold.step(event) {
                    Some(item) => Poll::Ready(Some(item)),
                    None => continue,
                },
            };
        }
    }
}

#[cfg(test)]
mod tests;
