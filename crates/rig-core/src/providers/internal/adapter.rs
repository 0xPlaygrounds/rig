//! The wire-adapter contract and its single-policy-site driver.
//!
//! Every streaming wire family is one [`WireAdapter`]: a sans-IO pair of pure
//! functions — `classify` (delegating to a `wire.rs` classifier) and
//! `interpret` (stateful event → canonical-grammar mapping). The generic
//! [`run_wire_stream`] driver owns the *entire* frame-triage policy, so no
//! adapter can hand-roll its own handling of unknown or corrupt frames:
//!
//! | classify                  | driver action                                |
//! |---------------------------|----------------------------------------------|
//! | [`WireEvent::Known`]      | `adapter.interpret`, yield its outputs       |
//! | [`WireEvent::Unknown`]    | `tracing::warn!` (with the payload) + skip   |
//! | [`WireEvent::Corrupt`]    | in-band `Err` item, keep consuming           |
//! | transport `Err`           | `Err` item, then end (truncation semantics — |
//! |                           | no `finish` flush, no terminal record)       |
//!
//! The trait is public so out-of-tree providers implement it and inherit the
//! shared driver and policy instead of hand-rolling assemblers; like the
//! erased-model precedent, an adapter is constructed once per stream and never
//! stored as a generic.

use std::borrow::Cow;

use futures::{Stream, StreamExt};

use super::wire::WireEvent;
use crate::completion::CompletionError;
use crate::streaming::{RawStreamingChoice, RawStreamingResult};
use crate::wasm_compat::WasmCompatSend;

/// One transport frame, after framing but before decoding.
///
/// The transport layer (SSE framer, NDJSON splitter, websocket reader) owns
/// byte splitting and yields these; adapters never split bytes.
#[derive(Debug, Clone)]
pub enum WireFrame {
    /// A decoded text payload — an SSE `data:` field or a ws message body.
    Text(String),
    /// A raw byte payload — an NDJSON line or a binary SDK frame.
    Bytes(Vec<u8>),
}

impl WireFrame {
    /// The frame payload as text (lossy for byte frames).
    pub fn as_str(&self) -> Cow<'_, str> {
        match self {
            Self::Text(text) => Cow::Borrowed(text),
            Self::Bytes(bytes) => String::from_utf8_lossy(bytes),
        }
    }
}

/// What one adapter step hands back to the driver.
///
/// `Err` items are data-level defects the adapter itself detects while
/// assembling (e.g. accumulated tool-argument JSON that fails to parse);
/// frame-level defects never reach `interpret` — the driver surfaces those
/// from `classify` directly.
pub type AdapterOutput<R> = Vec<Result<RawStreamingChoice<R>, CompletionError>>;

/// One streaming wire family as a thin adapter onto the canonical grammar.
///
/// `classify` and `interpret` are sans-IO by construction: no transport
/// handle, no async — pure `(state, event) → events` functions, testable by
/// feeding events directly with no mock HTTP.
pub trait WireAdapter {
    /// The transport frame this adapter classifies: [`WireFrame`] for byte
    /// wires (SSE, NDJSON, websocket), the SDK's own event type for
    /// typed-transport wires (bedrock's Converse events, gemini-grpc's
    /// protobuf responses, candle's in-process generation events).
    type Frame;
    /// The wire's typed event, produced by the `wire.rs` classifier.
    type Event;
    /// The provider-native terminal record carried by
    /// [`RawStreamingChoice::FinalResponse`].
    type Response;

    /// Decode + classify one transport frame. MUST delegate to a `wire.rs`
    /// classifier (`classify_tagged_frame` / `classify_chat_completions_frame`
    /// / `classify_untyped_line` / `classify_typed_event`) — never raw serde,
    /// so the decode-then-validate policy cannot be re-derived per adapter.
    fn classify(&self, frame: Self::Frame) -> WireEvent<Self::Event>;

    /// Map one `Known` event to canonical grammar events. Stateful: index→id
    /// maps, open-block state, id fabrication, and wire-quirk quarantine live
    /// here — policy for unknown/corrupt frames does not (the driver owns it).
    ///
    /// Pushing a [`RawStreamingChoice::FinalResponse`] marks the provider's
    /// genuine terminal; the driver stops consuming after yielding it.
    fn interpret(&mut self, event: Self::Event, out: &mut AdapterOutput<Self::Response>);

    /// End-of-stream flush on EOF without a terminal (close open blocks).
    ///
    /// Never runs after a transport error (truncation drops partials) or after
    /// a terminal was interpreted. Must not synthesize a terminal record: EOF
    /// without the provider's own end event is truncation, and a fabricated
    /// terminal would read as a successfully completed turn.
    fn finish(&mut self, out: &mut AdapterOutput<Self::Response>);
}

/// Drive one transport stream through an adapter under the shared policy.
///
/// This is the single policy site for every wire family (see the module table).
/// Adapters contain no `match WireEvent`.
pub fn run_wire_stream<A, S>(transport: S, mut adapter: A) -> RawStreamingResult<A::Response>
where
    A: WireAdapter + WasmCompatSend + 'static,
    A::Frame: WasmCompatSend,
    A::Event: WasmCompatSend,
    A::Response: WasmCompatSend + 'static,
    S: Stream<Item = Result<A::Frame, CompletionError>> + WasmCompatSend + 'static,
{
    Box::pin(async_stream::stream! {
        let mut transport = Box::pin(transport);
        let mut out: AdapterOutput<A::Response> = Vec::new();

        while let Some(frame) = transport.next().await {
            let frame = match frame {
                Ok(frame) => frame,
                Err(error) => {
                    // Truncation semantics: the error is the last item — no
                    // finish flush (partials drop), no terminal record.
                    yield Err(error);
                    return;
                }
            };

            match adapter.classify(frame) {
                WireEvent::Known(event) => adapter.interpret(event, &mut out),
                WireEvent::Unknown { event_type, value } => {
                    tracing::warn!(
                        event_type,
                        payload = %value,
                        "skipping unrecognized stream event"
                    );
                }
                WireEvent::Corrupt(error) => {
                    yield Err(CompletionError::JsonError(error));
                }
            }

            let saw_terminal = out
                .iter()
                .any(|item| matches!(item, Ok(RawStreamingChoice::FinalResponse(_))));
            for item in out.drain(..) {
                yield item;
            }
            if saw_terminal {
                return;
            }
        }

        adapter.finish(&mut out);
        for item in out.drain(..) {
            yield item;
        }
    })
}

/// Fabricated per-stream identity for wires that carry none.
///
/// Every id-less wire mints ids the same way — a namespace plus a counter or
/// wire index — and the namespaces are the reserved set the boundary-minted
/// provenance gate recognizes (`streaming::MINTED_ID_NAMESPACES`),
/// so a minted id can never serialize upstream as a wire-genuine one.
#[derive(Debug)]
pub struct SyntheticIds {
    namespace: &'static str,
    next: u64,
}

impl SyntheticIds {
    /// Ids in the `reasoning-` namespace (constant-id wires: gemini REST,
    /// ollama, chat-compat, candle).
    pub fn reasoning() -> Self {
        Self::new("reasoning-")
    }

    /// Ids in the `block-` namespace (index-as-id wires: anthropic, bedrock).
    pub fn block() -> Self {
        Self::new("block-")
    }

    /// Ids in the `output-` namespace (the Responses `output_index` fallback).
    pub fn output() -> Self {
        Self::new("output-")
    }

    /// Ids in the `tool-` namespace (chat-compat tool-call fragments whose
    /// wire supplies no tool-call id; identity derives from the chunk index).
    pub fn tool() -> Self {
        Self::new("tool-")
    }

    fn new(namespace: &'static str) -> Self {
        Self { namespace, next: 0 }
    }

    /// Mint the next counter-based id (vercel's `blockCounter++` pattern).
    pub fn mint(&mut self) -> String {
        let id = self.for_index(self.next);
        self.next += 1;
        id
    }

    /// The id for a stable wire-supplied index (anthropic's content-block
    /// index pattern).
    pub fn for_index(&self, index: impl std::fmt::Display) -> String {
        format!("{}{}", self.namespace, index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every id this helper can mint must be recognized by the F7 provenance
    /// gate, so fabricated reasoning identities never serialize upstream.
    #[test]
    fn minted_ids_are_recognized_as_boundary_minted() {
        for mut ids in [
            SyntheticIds::reasoning(),
            SyntheticIds::block(),
            SyntheticIds::output(),
            SyntheticIds::tool(),
        ] {
            let counter_id = ids.mint();
            assert!(
                crate::streaming::is_boundary_minted_id(&counter_id),
                "{counter_id} must be provenance-gated"
            );
            let index_id = ids.for_index(7usize);
            assert!(
                crate::streaming::is_boundary_minted_id(&index_id),
                "{index_id} must be provenance-gated"
            );
        }
    }

    #[test]
    fn mint_counts_up_per_stream() {
        let mut ids = SyntheticIds::reasoning();
        assert_eq!(ids.mint(), "reasoning-0");
        assert_eq!(ids.mint(), "reasoning-1");
    }
}
