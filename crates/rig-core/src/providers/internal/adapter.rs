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
//! | [`WireEvent::Unknown`]    | `tracing::warn!` (metadata only), skip on    |
//! |                           | the semantic path, and yield the raw value   |
//! |                           | as [`RawStreamingChoice::Unknown`] (the      |
//! |                           | passthrough channel — never aggregated)      |
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
///
/// # Contract for implementors (in-tree and out-of-tree)
///
/// This trait is public so companion provider crates (rig-bedrock,
/// rig-gemini-grpc, rig-candle) and out-of-tree providers implement it and
/// inherit the shared [`run_wire_stream`] / [`run_wire_buffered`] drivers.
/// An implementation must uphold:
///
/// - **Classify delegation**: [`WireAdapter::classify`] delegates to a
///   `wire.rs` classifier — never raw serde — so decode-then-validate policy
///   is stated once per wire family.
/// - **Driver-owns-policy**: unknown/corrupt-frame handling belongs to the
///   driver (module policy table); adapters contain no `match WireEvent`.
/// - **Mandatory identity**: every `Reasoning`/`ReasoningDelta`,
///   `ToolCallDelta`, and `TextStart` event carries a
///   [`StreamPartId`](crate::streaming::StreamPartId) — the wire's own identity
///   (`StreamPartId::Wire`) when it exists, else
///   an identity minted via [`SyntheticIds`]
///   (`StreamPartId::Minted`). Provenance
///   travels in the type: a minted identity keys stream accumulation and
///   structurally cannot become a durable provider handle or reach a request
///   serializer, so no per-provider gate exists or is needed.
/// - **Finish/flush obligations**: see [`WireAdapter::finish`] (EOF-only,
///   never synthesizes a terminal) and
///   [`WireAdapter::flush_before_terminal_error`] (fully-delivered content
///   only, no terminal record).
/// - **[`WireAdapter::is_finished`]**: `true` only after `interpret`
///   consumed the wire's own in-band terminal failure, having pushed the
///   flush-then-`Err` sequence itself.
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
    /// terminal would read as a successfully completed turn. (A terminal the
    /// provider *did* signal earlier — e.g. the chat-completions `[DONE]`
    /// sentinel or a `finish_reason` chunk, whose usage trailer arrives later —
    /// may be emitted here; that is deferral, not synthesis.)
    fn finish(&mut self, out: &mut AdapterOutput<Self::Response>);

    /// Flush content the provider fully delivered before a terminal error item
    /// (a transport failure or an in-band provider error envelope) reaches the
    /// consumer.
    ///
    /// Default: nothing — truncation drops partials. Wires that buffer
    /// fully-delivered tool calls (the chat-completions compat family, the
    /// Responses SSE loop) override this so a first-`Err`-stop consumer still
    /// sees them. Must not push a terminal record.
    fn flush_before_terminal_error(&mut self, _out: &mut AdapterOutput<Self::Response>) {}

    /// Whether `interpret` consumed the wire's own in-band terminal failure.
    ///
    /// When true after an `interpret` call, the driver stops consuming without
    /// running the EOF `finish` flush — the adapter has already pushed the
    /// flush-then-`Err` sequence itself. Default: never.
    fn is_finished(&self) -> bool {
        false
    }
}

/// One frame after [`triage_frame`]: a modeled event for `interpret`, or an
/// unknown frame's raw payload for the passthrough channel.
#[derive(Debug)]
pub enum TriagedFrame<T> {
    /// A modeled event, ready for [`WireAdapter::interpret`].
    Event(T),
    /// An unknown frame's raw payload. Already warned; the caller forwards it
    /// as [`RawStreamingChoice::Unknown`] where the surface has a raw channel
    /// (openai-agents' raw-event precedent), and never interprets it — the
    /// semantic path skips it.
    Unknown(crate::streaming::UnknownPayload),
}

/// Triage one classified frame under the shared policy table (see the module
/// docs): `Known` passes through, `Unknown` is warned (structural metadata
/// only) and handed back raw for the passthrough channel, `Corrupt` is a
/// [`CompletionError::JsonError`].
///
/// This is [`run_wire_stream`]'s per-frame policy factored out for the
/// non-stream surfaces that classify frames one at a time (the websocket
/// pre-dispatch, the interactions typed-event stream), so they share the
/// driver's table instead of restating it.
pub fn triage_frame<T>(event: WireEvent<T>) -> Result<TriagedFrame<T>, CompletionError> {
    match event {
        WireEvent::Known(event) => Ok(TriagedFrame::Event(event)),
        WireEvent::Unknown { event_type, value } => {
            // Structural metadata only — see `warn_unmodeled`. The full
            // payload survives on the `Unknown` raw passthrough channel;
            // that channel IS the opt-in for consumers who want the content.
            warn_unmodeled(&event_type, &value);
            Ok(TriagedFrame::Unknown(value))
        }
        WireEvent::Corrupt(error) => Err(CompletionError::JsonError(error)),
    }
}

/// Warn about an unmodeled wire payload with **structural metadata only** —
/// its kind and serialized byte size, never the payload itself. Unmodeled
/// frames and parts can carry model output or other sensitive provider
/// data, which must not leak into production WARN logs; the one redaction
/// policy lives here, used by the driver's Unknown arm and by adapters that
/// skip an unmodeled part kind. `driver_adoption.rs` scans streaming
/// modules for direct `warn!(?...)` payload captures, so bypassing this
/// helper fails CI.
pub fn warn_unmodeled(kind: &str, payload: &impl serde::Serialize) {
    tracing::warn!(
        kind,
        payload_bytes = unknown_payload_bytes(payload),
        "skipping unmodeled wire payload"
    );
}

/// Serialized byte size of an unknown frame's payload, for the structural
/// warn log (the log never carries the payload itself).
fn unknown_payload_bytes(value: &impl serde::Serialize) -> u64 {
    /// Counter sink: measures how many bytes serialization would write
    /// without buffering them.
    struct CountingWriter(u64);

    impl std::io::Write for CountingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0 += buf.len() as u64;
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let mut counter = CountingWriter(0);
    // A `Value` cannot fail to serialize; degrade to 0 rather than panic.
    let _ = serde_json::to_writer(&mut counter, value);
    counter.0
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
        // Debug-mode sequence laws over the raw adapter output: every
        // conformance fixture and cassette replay checks what the adapter
        // ACTUALLY emits, not just what accumulator fixtures spell.
        // Compiled out of release builds.
        #[cfg(any(test, debug_assertions))]
        let mut sequence_laws = super::sequence_law::SequenceLaws::default();

        while let Some(frame) = transport.next().await {
            let frame = match frame {
                Ok(frame) => frame,
                Err(error) => {
                    // Truncation semantics: the error is the last item — no
                    // finish flush (partials drop), no terminal record. Content
                    // the provider fully delivered (an adapter's buffered tool
                    // calls) still flushes first, so a first-`Err`-stop
                    // consumer sees it.
                    adapter.flush_before_terminal_error(&mut out);
                    for item in out.drain(..) {
                        yield item;
                    }
                    yield Err(error);
                    return;
                }
            };

            match triage_frame(adapter.classify(frame)) {
                Ok(TriagedFrame::Event(event)) => adapter.interpret(event, &mut out),
                // Skipped semantically, but surfaced verbatim on the raw
                // passthrough channel so consumers who want unmodeled frames
                // can observe them; aggregation never folds `Unknown` into
                // the assistant choice.
                Ok(TriagedFrame::Unknown(value)) => {
                    out.push(Ok(RawStreamingChoice::Unknown(value)));
                }
                Err(error) => {
                    yield Err(error);
                }
            }

            #[cfg(any(test, debug_assertions))]
            sequence_laws.check_batch(&out);

            let saw_terminal = out
                .iter()
                .any(|item| matches!(item, Ok(RawStreamingChoice::FinalResponse(_))));
            for item in out.drain(..) {
                yield item;
            }
            if saw_terminal || adapter.is_finished() {
                return;
            }
        }

        adapter.finish(&mut out);
        #[cfg(any(test, debug_assertions))]
        sequence_laws.check_batch(&out);
        for item in out.drain(..) {
            yield item;
        }
    })
}

/// Drive an already-buffered frame sequence through an adapter under the
/// no-stream policy.
///
/// This is the driver's buffered/unary mode, for replayed SSE bodies decoded
/// after the fact (the Responses unary path, ChatGPT's replayed bodies). There
/// is no stream to carry in-band `Err` items, so the policy table tightens —
/// everything else is identical to [`run_wire_stream`]:
///
/// | classify                  | buffered action                              |
/// |---------------------------|----------------------------------------------|
/// | [`WireEvent::Known`]      | `adapter.interpret`; an `Err` item it pushes |
/// |                           | fails the whole operation                    |
/// | [`WireEvent::Unknown`]    | `tracing::warn!` + skip (a buffered result   |
/// |                           | is a finished completion — there is no       |
/// |                           | stream to carry the raw passthrough item)    |
/// | [`WireEvent::Corrupt`]    | fail the whole operation — the alternative   |
/// |                           | is a successful-but-incomplete completion    |
///
/// The `Corrupt` error's own message is surfaced verbatim (as a
/// [`CompletionError::ResponseError`]), so a classifier can attach
/// frame-naming context for the operation error.
pub fn run_wire_buffered<A>(
    frames: impl IntoIterator<Item = A::Frame>,
    mut adapter: A,
) -> Result<Vec<RawStreamingChoice<A::Response>>, CompletionError>
where
    A: WireAdapter,
{
    let mut out: AdapterOutput<A::Response> = Vec::new();
    let mut choices = Vec::new();
    // Same debug-mode sequence laws as `run_wire_stream` (see there).
    #[cfg(any(test, debug_assertions))]
    let mut sequence_laws = super::sequence_law::SequenceLaws::default();

    for frame in frames {
        match adapter.classify(frame) {
            WireEvent::Known(event) => adapter.interpret(event, &mut out),
            WireEvent::Unknown { event_type, value } => {
                // Structural metadata only, matching [`triage_frame`]: unknown
                // payloads can carry sensitive provider data and must not leak
                // into WARN logs. (The stream driver additionally surfaces the
                // full payload on the `Unknown` raw channel — the opt-in for
                // consumers who want the content; a buffered result has no
                // such channel, so here the payload is simply skipped.)
                tracing::warn!(
                    event_type,
                    payload_bytes = unknown_payload_bytes(&value),
                    "skipping unrecognized stream event"
                );
            }
            WireEvent::Corrupt(error) => {
                return Err(CompletionError::ResponseError(error.to_string()));
            }
        }

        #[cfg(any(test, debug_assertions))]
        sequence_laws.check_batch(&out);

        let saw_terminal = drain_buffered(&mut out, &mut choices)?;
        if saw_terminal || adapter.is_finished() {
            return Ok(choices);
        }
    }

    adapter.finish(&mut out);
    #[cfg(any(test, debug_assertions))]
    sequence_laws.check_batch(&out);
    drain_buffered(&mut out, &mut choices)?;
    Ok(choices)
}

/// Move one buffered step's output into `choices`, failing the operation on
/// the first `Err` item; reports whether a terminal record was appended.
fn drain_buffered<R>(
    out: &mut AdapterOutput<R>,
    choices: &mut Vec<RawStreamingChoice<R>>,
) -> Result<bool, CompletionError> {
    let mut saw_terminal = false;
    for item in out.drain(..) {
        let choice = item?;
        saw_terminal |= matches!(choice, RawStreamingChoice::FinalResponse(_));
        choices.push(choice);
    }
    Ok(saw_terminal)
}

pub use crate::streaming::SyntheticIds;
