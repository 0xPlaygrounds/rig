//! This module provides functionality for working with streaming completion models.
//! It provides traits and types for generating streaming completion requests and
//! handling streaming completion responses.
//!
//! Provider implementations use these types to expose raw streamed completion
//! events without depending on a runtime.

mod identity;
mod parts;

use crate::completion::{CompletionError, CompletionResponse, Usage};
use crate::id::InternalCallId;
use crate::message::{
    AssistantContent, Reasoning, ReasoningContent, Text, ToolCall, ToolFunction, ToolResult,
};
use crate::wasm_compat::WasmCompatSend;
use futures::stream::{AbortHandle, Abortable};
use futures::task::AtomicWaker;
use futures::{Stream, StreamExt};
pub use identity::{MintKind, StreamPartId, SyntheticIds, WireId};
use parts::PartsAccumulator;
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

/// The content of a tool call delta - either the tool name or argument data
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum ToolCallDeltaContent {
    /// Tool/function name emitted by the provider.
    Name(String),
    /// Partial JSON argument data emitted by the provider.
    Delta(String),
}

/// How the shared assembler treats an argument payload that does not parse as
/// JSON when a streamed tool call's input ends.
///
/// This is genuine wire-family policy, declared by the adapter on the end
/// event rather than hand-rolled per provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// End of a streamed tool call's input: the signal for the shared assembler
/// ([`RawStreamingChoice::ToolInputEnd`]) to finalize the call.
///
/// Optional fields are authoritative wire values that supersede the assembled
/// state — a wire whose completed item restates the call (OpenAI Responses
/// `output_item.done`) carries them; delta-only wires leave them `None` and
/// the assembled fragments are parsed instead.
#[derive(Debug, Clone)]
pub struct ToolInputEnd {
    /// Assembly identity: the id the call's fragments were emitted under.
    pub id: StreamPartId,
    /// Authoritative provider-issued tool id, when one exists (e.g. an id
    /// that arrived after the call opened id-less). The durable handle;
    /// absence is `None`, never an empty string.
    pub tool_id: Option<WireId>,
    /// Authoritative tool name from the wire's completed item.
    pub name: Option<String>,
    /// Authoritative parsed arguments from the wire's completed item.
    pub arguments: Option<serde_json::Value>,
    /// Provider call-correlation id (e.g. OpenAI Responses `call_id`).
    pub call_id: Option<String>,
    /// Provider signature attached to the completed call.
    pub signature: Option<String>,
    /// Provider-specific metadata attached to the completed call.
    pub additional_params: Option<serde_json::Value>,
    /// Wire-family policy for assembled arguments that fail to parse.
    pub on_unparseable: UnparseableToolInput,
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

impl ToolInputEnd {
    /// End the call identified by `id`, finalizing from assembled fragments
    /// with the given unparseable-input policy.
    pub fn new(id: impl Into<StreamPartId>, on_unparseable: UnparseableToolInput) -> Self {
        Self {
            id: id.into(),
            tool_id: None,
            name: None,
            arguments: None,
            call_id: None,
            signature: None,
            additional_params: None,
            on_unparseable,
        }
    }
}

/// Discriminant for [`StreamFinal`].
///
/// [`StreamedAssistantContent`] is `#[serde(untagged)]` and its
/// [`StreamedAssistantContent::Unknown`] variant matches any JSON value, so the
/// terminal record needs a field that identifies it structurally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamFinalKind {
    /// The provider's terminal stream event.
    Final,
}

/// The provider's terminal stream record, normalized.
///
/// This replaces the provider-typed final payload that streams used to carry:
/// usage is a plain field rather than a trait method, and the finish reason is
/// normalized exactly as on the unary [`CompletionResponse`].
///
/// Providers that want their own terminal type keep it behind
/// [`RawStreamingResult`] and map it once with [`normalize_stream`].
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
    /// Discriminating field; always [`StreamFinalKind::Final`].
    pub kind: StreamFinalKind,
    /// Token usage reported by the provider for this streamed completion.
    /// Zero-valued usage is the documented sentinel for missing metrics.
    pub usage: Usage,
    /// Why the model stopped generating, when the provider reported it.
    ///
    /// [`normalize_stream`] applies
    /// [`FinishReason::reconcile_with_output`](crate::completion::FinishReason::reconcile_with_output)
    /// to this value using the tool calls actually seen on the stream, so a
    /// provider mapper does not need to (and cannot — it has no view of the
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
    /// The provider's own terminal record for this stream: the value the
    /// model's inherent `raw_stream` would have yielded as its `FinalResponse`,
    /// serialized. It is the terminal record as rig's wire type parsed it —
    /// fields that type does not model are not here — and it is the terminal
    /// record only, not the stream's frames; see the module docs for why
    /// frames are a separate mechanism. [`normalize_stream`] populates it
    /// unconditionally — the same parity the pre-normalization `Final(R)` had.
    ///
    /// An escape hatch for provider-specific data rig does not normalize — it
    /// never replaces a normalized field, and every normalized field means the
    /// same thing whatever this holds. `Value::Null` means the record was
    /// built without a provider behind it — [`StreamFinal::new`] without
    /// `with_raw` (a provider's mapper before [`normalize_stream`] attaches
    /// the terminal, test doubles, hand-built records), or a record persisted
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
            kind: StreamFinalKind::Final,
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
/// `with_*` setters apply. This mirror deserializes the exact wire shape —
/// including the discriminating `kind` field — and [`From`] funnels it through
/// [`StreamFinal::new`] and the setters, so every deserialized value satisfies
/// the same invariants as a constructed one. Serialization stays derived on
/// [`StreamFinal`] itself, so the wire format is unchanged.
#[derive(Deserialize)]
struct StreamFinalRepr {
    kind: StreamFinalKind,
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
            kind,
            usage,
            finish_reason,
            message_id,
            response_id,
            provider_request_id,
            provider,
            model,
            raw,
        } = repr;
        // `StreamFinal::new` sets the only possible discriminant; the
        // irrefutable pattern consumes the mirrored field.
        let StreamFinalKind::Final = kind;
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

/// Enum representing a streaming chunk from the model.
///
/// `R` is the terminal record type. Ordinary streams use the normalized
/// [`StreamFinal`] default; a provider's inherent `raw_stream` method
/// substitutes its own native terminal type over the same event vocabulary,
/// which is what keeps [`crate::completion::CompletionModel`] free of response
/// associated types.
#[derive(Debug, Clone)]
pub enum RawStreamingChoice<R = StreamFinal> {
    /// A text chunk from a message response
    Message(String),

    /// Start a new text content block in the accumulated final choice.
    ///
    /// This is an internal provider-normalization event. It is not yielded to
    /// public stream consumers, but lets providers preserve block boundaries
    /// and metadata for final aggregated assistant text blocks.
    TextStart {
        /// Identity of the text block being opened.
        ///
        /// The same mandatory-identity contract as
        /// [`RawStreamingChoice::Reasoning::id`]: distinct wire output items
        /// must aggregate as distinct text parts (two OpenAI Responses
        /// `message` items must not concatenate), so the accumulator keys
        /// text blocks by identity. Providers propagate the wire's item
        /// identity (`StreamPartId::Wire`: the Responses `item_id`, Anthropic's
        /// block index) when it exists, or mint one at the boundary
        /// (`StreamPartId::Minted`, via [`SyntheticIds`]). A wire that never
        /// announces text boundaries may skip `TextStart` entirely: a bare
        /// [`RawStreamingChoice::Message`] with no open block opens a
        /// boundary-minted block.
        id: StreamPartId,
        /// Provider-specific metadata attached to this text block.
        additional_params: Option<crate::message::AdditionalParams>,
    },

    /// Provider-specific metadata for the current text content block.
    ///
    /// This is not yielded to public stream consumers. The metadata is merged
    /// into the current aggregated [`Text`] block.
    /// [`crate::message::AdditionalParams`] is non-empty by construction, so
    /// a provider with nothing to attach skips the variant instead of
    /// emitting an empty carrier.
    TextAdditionalParams(crate::message::AdditionalParams),

    /// A tool call response (in its entirety) — wires that never fragment
    /// tool input emit this directly; fragmenting wires emit
    /// [`RawStreamingChoice::ToolCallDelta`] fragments closed by
    /// [`RawStreamingChoice::ToolInputEnd`], and the shared accumulator
    /// assembles the completed call.
    ToolCall(RawStreamingToolCall),
    /// A tool call partial/delta.
    ///
    /// All fragments of one call carry one `id`; the shared accumulator keys
    /// assembly by it and mints the internal correlation id when the call
    /// opens, so adapters never track per-call state.
    ToolCallDelta {
        /// Identity of the tool call this fragment extends, stable across the
        /// call's fragments.
        ///
        /// The same mandatory-identity contract as
        /// [`RawStreamingChoice::Reasoning::id`]: parallel calls interleave
        /// their fragments on real wires, so the accumulator must key
        /// assembly by identity. Providers propagate the wire's tool-call id
        /// (`StreamPartId::Wire`), or mint one at the boundary from the wire's
        /// own index (`StreamPartId::Minted`, via [`SyntheticIds`]) when the wire
        /// omits it — a shared identity would collapse parallel calls into
        /// one corrupted assembly. A minted identity keys assembly only; it
        /// never becomes the completed call's durable
        /// [`ToolCall::id`](crate::message::ToolCall::id).
        id: StreamPartId,
        content: ToolCallDeltaContent,
    },
    /// End of a streamed tool call's input: the shared accumulator finalizes
    /// the assembled fragments (or the event's authoritative payload) into a
    /// completed tool call.
    ToolInputEnd(ToolInputEnd),
    /// A reasoning (in its entirety)
    Reasoning {
        /// Identity of the reasoning item this block belongs to.
        ///
        /// Required: reasoning interleaves with other output on real wires
        /// (OpenAI Responses emits the completed item after tool calls), so
        /// the accumulator must key by identity rather than guess by
        /// adjacency. Providers propagate the wire's item id
        /// (`StreamPartId::Wire`: `item_id` on Responses events) or mint a
        /// stream-scoped id at the boundary (`StreamPartId::Minted`, via
        /// [`SyntheticIds`]) when the wire has none. Deltas and the full
        /// block for the same item MUST carry the same key.
        id: StreamPartId,
        /// The provider-issued reasoning item id, when one exists — the
        /// durable handle that becomes
        /// [`Reasoning::id`](crate::message::Reasoning::id) and round-trips
        /// upstream. Carried separately from the accumulation key: the key
        /// is opaque and can never leak; the handle is data.
        provider_id: Option<WireId>,
        /// Complete reasoning content block.
        content: ReasoningContent,
    },
    /// Open the reasoning part identified by `id`.
    ///
    /// Optional — a bare [`RawStreamingChoice::ReasoningDelta`] opens its
    /// part leniently — but a wire that announces block starts should emit
    /// it so arrival order is fixed at the wire's own boundary. A start for
    /// an already-open key is a no-op; a start for a finished key opens a
    /// new part (key reuse). Not yielded to public stream consumers.
    ReasoningStart {
        /// Accumulation key of the reasoning part being opened.
        id: StreamPartId,
        /// The provider-issued reasoning item id, when one exists.
        provider_id: Option<WireId>,
    },

    /// Close the reasoning part identified by `id` — the lifecycle
    /// primitive every wire has (or has synthesized by its adapter at the
    /// boundaries it already detects), so "is this part still open?" is
    /// never re-derived per wire.
    ///
    /// `reasoning` is the wire's authoritative whole-block restatement; it
    /// supersedes the delta accumulation. `signature` is a provider
    /// signature closing the block; it attaches to the part's text — and
    /// because an end for an already-finished key with only a signature
    /// attaches to THAT part, a trailing signature signs the block that
    /// holds the chain-of-thought instead of fabricating an empty sibling.
    /// A repeated end with no payload is a no-op: idempotence belongs to
    /// the entity, not to a guard each route must remember.
    ///
    /// The completed part is yielded to consumers as
    /// [`StreamedAssistantContent::Reasoning`] — the uniform
    /// block-completed signal across every wire — when the wire itself
    /// said something at the boundary: an end carrying a restatement or
    /// signature, or a bare end frame the wire actually sent
    /// (`wire_sent`). A bare end an adapter *synthesized* at an
    /// interleaving boundary stays silent: the consumer already received
    /// every delta, and fabricating a completion event the wire never
    /// sent would change what downstream history builders observe.
    ReasoningEnd {
        /// Accumulation key of the reasoning part being closed.
        id: StreamPartId,
        /// The wire's authoritative completed block, when it restates one.
        reasoning: Option<Reasoning>,
        /// A provider signature closing the block.
        signature: Option<String>,
        /// Whether the wire itself sent this end frame (anthropic's
        /// `content_block_stop`), as opposed to the adapter synthesizing
        /// it at a boundary the wire never announces. Wire-sent ends
        /// yield the completed block even when bare.
        wire_sent: bool,
    },

    /// Close the text block identified by `id`: later bare text deltas open
    /// a fresh block instead of extending it. (A later
    /// [`RawStreamingChoice::TextStart`] with the same key still
    /// reactivates the block — the keyed collapse is explicit.) Not yielded
    /// to public stream consumers.
    TextEnd {
        /// Accumulation key of the text block being closed.
        id: StreamPartId,
    },

    /// A reasoning partial/delta
    ReasoningDelta {
        /// Accumulation key of the reasoning item this delta extends. Same
        /// contract as [`RawStreamingChoice::Reasoning::id`]; all deltas of
        /// one block share one key.
        id: StreamPartId,
        /// The provider-issued reasoning item id, when one exists (see
        /// [`RawStreamingChoice::Reasoning::provider_id`]) — what a
        /// delta-built part records as its durable id.
        provider_id: Option<WireId>,
        /// Partial reasoning text.
        reasoning: String,
    },

    /// The final response object, must be yielded if you want the
    /// `response` field to be populated on the `StreamingCompletionResponse`
    FinalResponse(R),

    /// Provider-assigned message ID (e.g. OpenAI Responses API `msg_` ID).
    /// Captured silently into `StreamingCompletionResponse::message_id`.
    MessageId(String),

    /// A provider-native output item this version does not model — e.g. an
    /// OpenAI Responses hosted-tool result (`web_search_call`, `file_search_call`,
    /// `computer_call`, `code_interpreter_call`). Carries the raw item object
    /// verbatim. Forwarded to the stream consumer as
    /// [`StreamedAssistantContent::Unknown`] but not folded into the accumulated
    /// assistant message (there is no `AssistantContent::Unknown` history slot).
    Unknown(UnknownPayload),
}

impl<R> RawStreamingChoice<R> {
    /// Convert only the terminal record, preserving every incremental content
    /// event unchanged.
    pub fn try_map_final<S>(
        self,
        map: impl FnOnce(R) -> Result<S, CompletionError>,
    ) -> Result<RawStreamingChoice<S>, CompletionError> {
        Ok(match self {
            Self::Message(text) => RawStreamingChoice::Message(text),
            Self::TextStart {
                id,
                additional_params,
            } => RawStreamingChoice::TextStart {
                id,
                additional_params,
            },
            Self::TextAdditionalParams(params) => RawStreamingChoice::TextAdditionalParams(params),
            Self::ToolCall(call) => RawStreamingChoice::ToolCall(call),
            Self::ToolCallDelta { id, content } => {
                RawStreamingChoice::ToolCallDelta { id, content }
            }
            Self::ToolInputEnd(end) => RawStreamingChoice::ToolInputEnd(end),
            Self::Reasoning {
                id,
                provider_id,
                content,
            } => RawStreamingChoice::Reasoning {
                id,
                provider_id,
                content,
            },
            Self::ReasoningDelta {
                id,
                provider_id,
                reasoning,
            } => RawStreamingChoice::ReasoningDelta {
                id,
                provider_id,
                reasoning,
            },
            Self::ReasoningStart { id, provider_id } => {
                RawStreamingChoice::ReasoningStart { id, provider_id }
            }
            Self::ReasoningEnd {
                id,
                reasoning,
                signature,
                wire_sent,
            } => RawStreamingChoice::ReasoningEnd {
                id,
                reasoning,
                signature,
                wire_sent,
            },
            Self::TextEnd { id } => RawStreamingChoice::TextEnd { id },
            Self::FinalResponse(response) => RawStreamingChoice::FinalResponse(map(response)?),
            Self::MessageId(id) => RawStreamingChoice::MessageId(id),
            Self::Unknown(value) => RawStreamingChoice::Unknown(value),
        })
    }
}

/// Describes a streaming tool call response (in its entirety)
#[derive(Debug, Clone)]
pub struct RawStreamingToolCall {
    /// Accumulation/reconciliation key of the tool call —
    /// `StreamPartId::Wire`-derived when the provider supplied an id,
    /// minted when the wire omitted one. A key only; the durable id is
    /// [`RawStreamingToolCall::tool_id`].
    pub id: StreamPartId,
    /// The provider-issued tool id, when one exists — the durable handle
    /// that becomes [`ToolCall::id`](crate::message::ToolCall::id). Absent
    /// means absent: serializers omit the field, and nothing fabricated can
    /// take its place.
    pub tool_id: Option<WireId>,
    /// Rig-generated unique identifier for this tool call.
    pub internal_call_id: InternalCallId,
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
            // A parser-accumulator placeholder key; providers overwrite it
            // with the wire's key before emitting. Deliberately minted: an
            // unset key must never read as wire-derived.
            id: StreamPartId::minted(MintKind::Tool, u64::MAX),
            tool_id: None,
            internal_call_id: InternalCallId::new(),
            call_id: None,
            name: String::new(),
            arguments: serde_json::Value::Null,
            signature: None,
            additional_params: None,
        }
    }

    /// Create a complete tool call with a generated internal call ID.
    pub fn new(id: impl Into<StreamPartId>, name: String, arguments: serde_json::Value) -> Self {
        let id = id.into();
        // A wire-derived key doubles as the durable id (the common case:
        // providers key by the id the wire issued); minted keys carry none.
        let tool_id = id.wire_str().and_then(WireId::new);
        Self {
            id,
            tool_id,
            internal_call_id: InternalCallId::new(),
            call_id: None,
            name,
            arguments,
            signature: None,
            additional_params: None,
        }
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
        // Only provider-issued handles populate `provider`: a dual wire
        // carries (call_id, item id), a single wire carries its id in
        // `call_id`. With none, the correlation handle is minted and
        // `provider` records the absence — never an empty sentinel.
        let provider = crate::message::ProviderCallId::from_optional_wire(
            tool_call.call_id,
            tool_call.tool_id.map(WireId::into_string),
        );
        let id = crate::message::ToolCallId::for_provider(provider.as_ref());
        ToolCall {
            id,
            provider,
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
/// Provider stream whose terminal record is the provider-native `R`, on native
/// targets.
///
/// This is the raw channel of rig's two-channel contract: every provider's
/// inherent `raw_stream`/`raw_completion` returns provider-native types
/// directly from the wire decode, never routed through the normalized
/// accumulation ([`normalize_stream`] / the parts accumulator) — the
/// semantic channel maps this stream's terminal record exactly once. There
/// is deliberately no provider-*typed* payload on the normalized types; the
/// typed channels are the contract. What the normalized types also carry is
/// that same terminal record *serialized* ([`StreamFinal::raw`]) — for
/// callers who no longer hold the concrete model, an agent having erased it,
/// and so cannot reach the typed channel at all. The frames of the stream are
/// a different axis: they were never exposed on any rig surface, and exposing
/// them is a per-frame mechanism (a raw stream part), not a field on the
/// terminal record — so [`StreamFinal::raw`] captures the terminal only.
///
/// Precedent, read carefully: openai-agents also splits raw from semantic,
/// but the load-bearing part of its design is elsewhere — its semantic
/// layer never reads a delta at all (it acts only on whole done items and
/// the completed response), and delta aggregation is confined to the
/// per-provider adapter, which *synthesizes* a canonical terminal event so
/// the shared layer sees one grammar. rig cannot fully adopt that shape
/// (openai-agents' canonical grammar is one vendor's schema; rig normalizes
/// 14 wire families through one accumulator), and centralizing the
/// accumulator is what forces cross-provider identity — hence the
/// provenance-typed [`StreamPartId`]. What rig does copy from that precedent is
/// the raw channel itself and provenance-as-data rather than naming
/// convention.
pub type RawStreamingResult<R> =
    Pin<Box<dyn Stream<Item = Result<RawStreamingChoice<R>, CompletionError>> + Send>>;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
/// Provider stream whose terminal record is the provider-native `R`, on wasm
/// targets.
pub type RawStreamingResult<R> =
    Pin<Box<dyn Stream<Item = Result<RawStreamingChoice<R>, CompletionError>>>>;

/// Normalized provider stream, as consumed by [`StreamingCompletionResponse`].
pub type StreamingResult = RawStreamingResult<StreamFinal>;

/// Normalize the terminal record of a provider-native stream.
///
/// Every incremental event passes through untouched; only
/// [`RawStreamingChoice::FinalResponse`] is converted, by `map`. On the way
/// through, the stream remembers whether it emitted any tool call and applies
/// [`FinishReason::reconcile_with_output`](crate::completion::FinishReason::reconcile_with_output)
/// to the mapped record — the streaming counterpart of what
/// [`CompletionResponse::with_finish_reason`] does on the unary path, so both
/// paths agree about a `stop` that was really a tool call.
///
/// The provider-native terminal `R` is also serialized onto
/// [`StreamFinal::raw`] *before* `map` consumes it — this is the one
/// streaming seam every provider routes through, so it is the streaming
/// counterpart of the capture each provider's unary `completion` performs
/// before `normalize`. That is why `R` is bounded `Serialize`: every in-tree
/// terminal type already is, and a terminal that could not be serialized
/// could not be surfaced to callers who no longer hold the typed model.
pub fn normalize_stream<R, F>(stream: RawStreamingResult<R>, mut map: F) -> StreamingResult
where
    R: Serialize + 'static,
    F: FnMut(R) -> Result<StreamFinal, CompletionError> + WasmCompatSend + 'static,
{
    let mut emitted_tool_call = false;
    Box::pin(stream.map(move |item| {
        item.and_then(|choice| {
            // Only a completed `ToolCall` counts, because only that becomes an
            // `AssistantContent::ToolCall` in the aggregated choice — which is
            // exactly what the unary path reconciles against. Counting deltas
            // here would make a stream whose tool call never assembled report
            // `ToolCalls` while the same data converted to a unary response
            // reported `Stop`.
            if matches!(&choice, RawStreamingChoice::ToolCall(_)) {
                emitted_tool_call = true;
            }
            choice.try_map_final(|response| {
                // Capture before `map` consumes the terminal. A serialization
                // failure propagates: a silent `None` would contradict the
                // field's meaning (a provider record stands behind every
                // normalized terminal). In practice `to_value` on a value
                // that just deserialized cannot fail.
                let raw = serde_json::to_value(&response)?;
                let mut response = map(response)?.with_raw(raw);
                response.finish_reason = response
                    .finish_reason
                    .map(|reason| reason.reconcile_with_output(emitted_tool_call));
                Ok(response)
            })
        })
    }))
}

/// The response from a streaming completion request;
/// message and response are populated at the end of the
/// `inner` stream.
pub struct StreamingCompletionResponse {
    pub(crate) inner: Abortable<StreamingResult>,
    pub(crate) abort_handle: AbortHandle,
    pub(crate) pause_control: PauseControl,
    /// Accumulates the streamed parts of the final aggregated choice.
    parts: PartsAccumulator,
    /// Stable descriptor name of the provider producing this stream.
    ///
    /// Known when the stream is opened rather than when it terminates, so a
    /// stream that errors or is cancelled before its terminal record still
    /// names its provider.
    provider: String,
    /// The final aggregated message from the stream
    /// contains all text and tool calls generated
    pub choice: Vec<AssistantContent>,
    /// Whether the stream already reached its end and aggregated `choice`.
    ///
    /// [`PartsAccumulator::finish`] is destructive (it takes the accumulated
    /// parts and falls back to one empty text part), so re-polling a drained
    /// stream — which `Stream` permits and combinators do — would otherwise
    /// replace a fully aggregated `choice` with empty text (#2258 H6).
    finished: bool,
    /// Rig-generated public correlators for reasoning parts, one per
    /// accumulation key: stable across a part's deltas, unique per run, and
    /// carrying nothing an accumulation key could leak.
    reasoning_correlators: std::collections::HashMap<StreamPartId, String>,
    /// Correlators of finished reasoning parts, kept for the stream's
    /// lifetime (mirroring the accumulator's `finished_reasoning`): a
    /// trailing signature-only end — Gemini's `thoughtSignature` after a
    /// synthesized silent boundary — must restate the identity its part's
    /// deltas carried, not mint a fresh one the assembler cannot match.
    finished_reasoning_correlators: std::collections::HashMap<StreamPartId, String>,
    /// The provider's normalized terminal record, may be `None`
    /// if the provider didn't yield it during the stream
    pub response: Option<StreamFinal>,
    pub final_response_yielded: AtomicBool,
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
            parts: PartsAccumulator::new(),
            provider: provider.into(),
            // A stream that has not produced anything yet has produced nothing.
            // This used to hold a fabricated empty-text part because the field
            // could not be empty; that part was indistinguishable from a real
            // empty text block the model had emitted.
            choice: Vec::new(),
            finished: false,
            reasoning_correlators: std::collections::HashMap::new(),
            finished_reasoning_correlators: std::collections::HashMap::new(),
            response: None,
            final_response_yielded: AtomicBool::new(false),
            message_id: None,
        }
    }

    /// Stable descriptor name of the provider producing this stream.
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// Resolve the public correlator for a reasoning part that just ended,
    /// keeping the identity available for the part's afterlife.
    ///
    /// An end always clears the live delta map — a reused accumulation key
    /// opens a NEW part whose deltas must mint fresh — but the taken
    /// correlator moves to the finished map rather than dying, so trailing
    /// metadata (a late signature after a synthesized silent end) restates
    /// the identity the part's deltas carried. A restatement under a spent
    /// key is a new sibling part: it mints fresh and overwrites the entry,
    /// exactly as the accumulator overwrites its finished index. Entries
    /// live until the stream is dropped, matching `finished_reasoning`.
    fn reasoning_end_correlator(&mut self, id: StreamPartId, restated: bool) -> String {
        match self.reasoning_correlators.remove(&id) {
            Some(taken) => {
                self.finished_reasoning_correlators
                    .insert(id, taken.clone());
                taken
            }
            None if restated => {
                let minted = crate::id::generate();
                self.finished_reasoning_correlators
                    .insert(id, minted.clone());
                minted
            }
            None => self
                .finished_reasoning_correlators
                .entry(id)
                .or_insert_with(crate::id::generate)
                .clone(),
        }
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

impl From<StreamingCompletionResponse> for CompletionResponse {
    fn from(value: StreamingCompletionResponse) -> CompletionResponse {
        // Usage is the zero sentinel (`Usage::new`) when the stream produced no
        // terminal record. `provider` comes from the stream itself rather than
        // the terminal record, so it is populated even then.
        let terminal = value.response.as_ref();
        CompletionResponse::new(
            value.choice,
            terminal.map(|response| response.usage).unwrap_or_default(),
            value.provider,
        )
        // An explicit `MessageId` event outranks the terminal record's ID.
        .with_optional_message_id(
            value
                .message_id
                .or_else(|| terminal.and_then(|response| response.message_id.clone())),
        )
        .with_optional_response_id(terminal.and_then(|response| response.response_id.clone()))
        .with_optional_provider_request_id(
            terminal.and_then(|response| response.provider_request_id.clone()),
        )
        .with_optional_finish_reason(terminal.and_then(|response| response.finish_reason.clone()))
        .with_optional_model(terminal.and_then(|response| response.model.clone()))
    }
}

impl Stream for StreamingCompletionResponse {
    type Item = Result<StreamedAssistantContent, CompletionError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let stream = self.get_mut();

        // A drained stream stays drained: `finish()` consumes the accumulated
        // parts, so re-polling must not run it again and clobber `choice`
        // with the empty-text fallback (#2258 H6).
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

        // Non-yielding events (`continue` arms: block bookkeeping, dropped
        // ends, duplicate terminals) loop rather than recurse — a long run of
        // them must not grow the stack (#2258 review P3).
        loop {
            return match Pin::new(&mut stream.inner).poll_next(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(None) => {
                    // Run at the end of the inner stream to collect all tokens
                    // into a single unified `Message`. `finish` can now be
                    // empty — a turn that streamed nothing is no longer padded
                    // with a fabricated empty-text part — and an empty result
                    // leaves the already-empty `choice` alone.
                    let finished = stream.parts.finish();
                    if !finished.is_empty() {
                        stream.choice = finished;
                    }
                    stream.finished = true;

                    Poll::Ready(None)
                }
                // Every error reaches the consumer. Cancellation is *not* an
                // error here: `cancel()` aborts through `Abortable`, which
                // terminates the inner stream with `Ready(None)` above, so
                // the aggregated choice is finished normally. (Until #2258 H8
                // this arm swallowed any `ProviderError` whose text merely
                // contained "aborted", reporting clean EOF while silently
                // discarding both the error and the streamed content.)
                Poll::Ready(Some(Err(err))) => Poll::Ready(Some(Err(err))),
                Poll::Ready(Some(Ok(choice))) => match choice {
                    RawStreamingChoice::Message(text) => {
                        stream.parts.text_delta(&text);
                        Poll::Ready(Some(Ok(StreamedAssistantContent::text(&text))))
                    }
                    RawStreamingChoice::TextStart {
                        id,
                        additional_params,
                    } => {
                        stream.parts.text_start(&id, additional_params);
                        continue;
                    }
                    RawStreamingChoice::TextAdditionalParams(additional_params) => {
                        stream.parts.text_additional_params(additional_params);
                        continue;
                    }
                    RawStreamingChoice::ToolCallDelta { id, content } => {
                        // The accumulator owns assembly; it mints the internal
                        // correlation id when the call opens and returns it for
                        // every fragment, so the public delta stays correlated
                        // with the eventual completed call.
                        let internal_call_id = match &content {
                            ToolCallDeltaContent::Name(name) => {
                                stream.parts.tool_name_delta(&id, name)
                            }
                            ToolCallDeltaContent::Delta(fragment) => {
                                stream.parts.tool_args_delta(&id, fragment)
                            }
                        };
                        Poll::Ready(Some(Ok(StreamedAssistantContent::ToolCallDelta {
                            internal_call_id,
                            content,
                        })))
                    }
                    RawStreamingChoice::ToolInputEnd(end) => match stream.parts.tool_input_end(end)
                    {
                        Ok(Some((tool_call, internal_call_id))) => {
                            Poll::Ready(Some(Ok(StreamedAssistantContent::ToolCall {
                                tool_call,
                                internal_call_id,
                            })))
                        }
                        // Dropped (nameless or partial input): not content.
                        Ok(None) => continue,
                        // Malformed complete input surfaces in-band; the stream
                        // keeps consuming, matching the malformed-frame contract.
                        Err(err) => Poll::Ready(Some(Err(err))),
                    },
                    RawStreamingChoice::Reasoning {
                        id,
                        provider_id,
                        content,
                    } => {
                        // A whole block is open + authoritative restatement
                        // + close in one event. The durable `Reasoning::id`
                        // comes only from the provider-issued handle; the
                        // accumulation key is opaque and cannot reach the
                        // replayable message.
                        let restatement = Reasoning {
                            id: provider_id.map(WireId::into_string),
                            content: vec![content],
                        };
                        let completed = stream.parts.reasoning_end(&id, Some(restatement), None);
                        // The part is finished: its delta correlator (fresh-
                        // minted for a block with no prior deltas) is restated
                        // on the completed event and retained for trailing
                        // metadata under the same key.
                        let correlator = stream.reasoning_end_correlator(id, true);
                        match completed {
                            Some(completed) => {
                                Poll::Ready(Some(Ok(StreamedAssistantContent::Reasoning {
                                    reasoning: completed,
                                    id: correlator,
                                })))
                            }
                            None => continue,
                        }
                    }
                    RawStreamingChoice::ReasoningStart { id, provider_id } => {
                        // A start that genuinely opened a part installs a
                        // fresh live correlator: without it, a part that
                        // closes with no deltas (a signature-only end under
                        // a reused key) would fall back to the finished map
                        // and inherit the PREVIOUS part's public identity.
                        if stream.parts.reasoning_start(&id, provider_id.as_ref()) {
                            stream
                                .reasoning_correlators
                                .insert(id, crate::id::generate());
                        }
                        continue;
                    }
                    RawStreamingChoice::ReasoningEnd {
                        id,
                        reasoning,
                        signature,
                        wire_sent,
                    } => {
                        // The completed block is yielded when the wire said
                        // something at the boundary: an end payload (a
                        // restatement or a signature) or a bare end frame
                        // the wire actually sent (anthropic's
                        // `content_block_stop` on an unsigned block). Only a
                        // bare end an adapter *synthesized* stays silent —
                        // the consumer already received every delta, and
                        // fabricating a "completed block" event the wire
                        // never sent would change what downstream history
                        // builders observe.
                        let authoritative = reasoning.is_some() || signature.is_some() || wire_sent;
                        let restated = reasoning.is_some();
                        let completed = stream.parts.reasoning_end(&id, reasoning, signature);
                        // The part is finished: the live delta map is cleared
                        // unconditionally — a suppressed synthesized end must
                        // still make a reused key mint fresh — but the
                        // correlator survives in the finished map, so a
                        // trailing signature-bearing end for this key restates
                        // the identity its deltas carried instead of minting
                        // one the assembler cannot match.
                        let correlator = stream.reasoning_end_correlator(id, restated);
                        match completed {
                            Some(completed) if authoritative => {
                                Poll::Ready(Some(Ok(StreamedAssistantContent::Reasoning {
                                    reasoning: completed,
                                    id: correlator,
                                })))
                            }
                            _ => continue,
                        }
                    }
                    RawStreamingChoice::TextEnd { id } => {
                        stream.parts.text_end(&id);
                        continue;
                    }
                    RawStreamingChoice::ReasoningDelta {
                        id,
                        provider_id,
                        reasoning,
                    } => {
                        stream
                            .parts
                            .reasoning_delta(&id, provider_id.as_ref(), &reasoning);
                        // The public delta carries a rig-generated correlator
                        // (stable per part, unique per run) plus the durable
                        // provider id when one exists. The opaque
                        // accumulation key is never observable.
                        let correlator = stream
                            .reasoning_correlators
                            .entry(id)
                            .or_insert_with(crate::id::generate)
                            .clone();
                        Poll::Ready(Some(Ok(StreamedAssistantContent::ReasoningDelta {
                            id: correlator,
                            provider_id: provider_id.map(WireId::into_string),
                            reasoning,
                        })))
                    }
                    RawStreamingChoice::ToolCall(raw_tool_call) => {
                        let minted_internal_call_id = raw_tool_call.internal_call_id;
                        let part_id = raw_tool_call.id.clone();
                        let tool_call: ToolCall = raw_tool_call.into();
                        // A wire that fragmented this call's input already
                        // published an internal id on its deltas; the
                        // accumulator adopts it so the completed call stays
                        // correlated with them (the contract on
                        // `StreamedAssistantContent::ToolCall`). With no open
                        // assembly the emitter's minted id is kept.
                        let internal_call_id = stream.parts.tool_call(
                            &part_id,
                            tool_call.clone(),
                            minted_internal_call_id,
                        );
                        Poll::Ready(Some(Ok(StreamedAssistantContent::ToolCall {
                            tool_call,
                            internal_call_id,
                        })))
                    }
                    RawStreamingChoice::FinalResponse(mut response) => {
                        // Assembled tool calls never pass `normalize_stream` as
                        // `RawStreamingChoice::ToolCall`, so the finish-reason
                        // reconciliation runs here too, against the accumulator's
                        // authoritative view of completed calls. Idempotent over
                        // the reconciliation `normalize_stream` already applied.
                        response.finish_reason = response.finish_reason.map(|reason| {
                            reason.reconcile_with_output(stream.parts.saw_tool_call())
                        });
                        if stream
                            .final_response_yielded
                            .load(std::sync::atomic::Ordering::SeqCst)
                        {
                            continue;
                        } else {
                            // Set the final response field and return the next item in the stream.
                            // An explicit `MessageId` event keeps precedence; the
                            // terminal record only fills a gap.
                            if stream.message_id.is_none() {
                                stream.message_id.clone_from(&response.message_id);
                            }
                            stream.response = Some(response.clone());
                            stream
                                .final_response_yielded
                                .store(true, std::sync::atomic::Ordering::SeqCst);
                            let final_response = StreamedAssistantContent::final_response(response);
                            Poll::Ready(Some(Ok(final_response)))
                        }
                    }
                    RawStreamingChoice::MessageId(id) => {
                        stream.message_id = Some(id);
                        continue;
                    }
                    RawStreamingChoice::Unknown(value) => {
                        // Pass an unmodeled provider item straight through to the
                        // consumer; it is intentionally not pushed into
                        // `assistant_items` (no `AssistantContent::Unknown` exists).
                        // No exclusion warning here: everything reaching this arm
                        // is a live wire frame a provider adapter chose not to
                        // model (adapters warn on those themselves) — a persisted
                        // item that failed the strict `Text` decode is created by
                        // consumer-side serde and never re-enters this stream.
                        // The agent assembler, which does ingest such items,
                        // carries that warning.
                        Poll::Ready(Some(Ok(StreamedAssistantContent::Unknown(value))))
                    }
                },
            };
        }
    }
}

// Test module
#[cfg(test)]
mod tests;

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
        internal_call_id: InternalCallId,
    },
    /// Partial tool call data emitted by the assistant.
    ToolCallDelta {
        /// Rig-generated correlator for this call: stable across the call's
        /// fragments, matches the eventual
        /// [`StreamedAssistantContent::ToolCall`], and unique per run.
        /// Provider-issued ids arrive on the completed [`ToolCall`]; no
        /// stream-internal key is ever rendered here.
        internal_call_id: InternalCallId,
        content: ToolCallDeltaContent,
    },
    /// Complete reasoning block emitted by the assistant.
    ///
    /// Supersedes any prior [`StreamedAssistantContent::ReasoningDelta`]s
    /// carrying the same correlator `id`: render it as a *replacement* for
    /// the accumulated delta text, not an addition. The match key is this
    /// variant's `id`, not [`Reasoning::id`](crate::message::Reasoning::id).
    /// The aggregated [`StreamingCompletionResponse::choice`] already
    /// applies this replacement.
    Reasoning {
        reasoning: Reasoning,
        /// Rig-generated correlator: matches the `id` on this part's prior
        /// [`StreamedAssistantContent::ReasoningDelta`]s and is unique per
        /// run. The durable provider handle is `reasoning.id`; this value
        /// never enters replayable history.
        id: String,
    },
    /// Partial reasoning text emitted by the assistant.
    ReasoningDelta {
        /// Rig-generated correlator for the reasoning part this delta
        /// extends: stable across the part's deltas and unique per run.
        /// Never a stream-internal key and never a fabricated provider
        /// value.
        id: String,
        /// The provider-issued reasoning item id, when one exists — the
        /// durable handle the aggregated
        /// [`Reasoning::id`](crate::message::Reasoning::id) will carry
        /// (`None` on wires that issue no reasoning ids).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_id: Option<String>,
        /// Partial reasoning text.
        reasoning: String,
    },
    /// The provider's normalized terminal record, if yielded by the stream.
    Final(StreamFinal),
    /// A provider-native output item rig does not model, preserved verbatim —
    /// e.g. an OpenAI Responses hosted-tool result (`web_search_call`,
    /// `file_search_call`, `computer_call`, `code_interpreter_call`). It is
    /// yielded to the consumer for inspection/forwarding but is not added to the
    /// accumulated assistant message or persisted history. Kept last because the
    /// enum is `#[serde(untagged)]` and the transparent payload wrapper
    /// matches anything, so earlier (typed) variants must be tried first.
    Unknown(UnknownPayload),
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
        internal_call_id: InternalCallId,
    },
}

impl StreamedUserContent {
    /// Create a streamed tool result correlated to an internal tool call ID.
    pub fn tool_result(tool_result: ToolResult, internal_call_id: InternalCallId) -> Self {
        Self::ToolResult {
            tool_result,
            internal_call_id,
        }
    }
}
