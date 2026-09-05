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
//! |                           | as [`StreamEvent::Unknown`] (the             |
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
use crate::streaming::{
    BlockClose, BlockId, BlockKind, Delta, MintKind, StreamEvent, StreamFinal, StreamingResult,
    SyntheticIds, ToolCallEnd, UnknownPayload,
};
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

/// What one `interpret` step emitted: the canonical events, with in-band
/// errors, plus the text-block bookkeeping every adapter needs.
///
/// Adapters push through the helpers so the grammar is stated once: a bare
/// text delta lands in the active text block (minted on demand, and a new
/// one after any non-text block — a completed tool call or a reasoning
/// block is a boundary for anonymous text), a tool/reasoning delta for an
/// unseen id is preceded by its `BlockStart`, and a whole call or whole
/// reasoning block is its start and its end. Frame-level defects never reach
/// `interpret` — the driver surfaces those from `classify` directly.
#[derive(Debug, Default)]
pub struct AdapterOutput {
    items: Vec<Result<StreamEvent, CompletionError>>,
    /// Minter for text blocks opened by a bare text delta.
    text_ids: Option<SyntheticIds>,
    /// The block receiving bare text deltas and metadata, until a boundary
    /// or an explicit text start/end switches it.
    active_text: Option<BlockId>,
    /// Minter for reasoning blocks opened by a bare reasoning delta.
    reasoning_ids: Option<SyntheticIds>,
    /// The block receiving bare reasoning deltas, until a boundary or an
    /// explicit reasoning end switches it.
    active_reasoning: Option<BlockId>,
    /// The blocks this output opened itself (minted on demand for a bare
    /// delta) and therefore closes itself — at the boundary that ends
    /// them, or at [`close_active_blocks`](Self::close_active_blocks). A
    /// block a provider opened explicitly is the provider's to close.
    auto_text: Option<BlockId>,
    auto_reasoning: Option<BlockId>,
    /// Whether a block this output opened itself is closed at its boundary
    /// (the bus's `StreamWriter`: a handler that says `text` then
    /// `tool_call` means the text block ended). Off for provider adapters,
    /// whose wires say where their blocks end — their event sequences are
    /// unchanged.
    self_closing: bool,
    /// Blocks a start was emitted for (or that a delta opened leniently),
    /// so a delta never precedes its block's start on the wire we emit.
    opened: std::collections::HashSet<BlockId>,
}

impl AdapterOutput {
    /// An empty output buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// An output that closes the blocks it opened itself at their boundary
    /// and at [`close_active_blocks`](Self::close_active_blocks): what a
    /// bus handler writes through, where nothing else will close them.
    pub fn self_closing() -> Self {
        Self {
            self_closing: true,
            ..Self::default()
        }
    }

    /// Push one event verbatim. A block this output opened itself for a
    /// bare delta is closed first when `item` is its boundary.
    pub fn push(&mut self, item: Result<StreamEvent, CompletionError>) {
        if self.self_closing
            && let Ok(event) = &item
            && event.block_id().is_some()
            && !Self::is_message_start(event)
        {
            if !Self::is_text_event(event)
                && let Some(id) = self.auto_text.take()
            {
                self.active_text = None;
                self.push_raw(Ok(StreamEvent::BlockEnd {
                    id,
                    end: BlockClose::Text,
                    block: None,
                }));
            }
            if !Self::is_reasoning_event(event)
                && let Some(id) = self.auto_reasoning.take()
            {
                self.active_reasoning = None;
                self.push_raw(Ok(StreamEvent::BlockEnd {
                    id,
                    end: BlockClose::Reasoning {
                        reasoning: None,
                        signature: None,
                        wire_sent: false,
                    },
                    block: None,
                }));
            }
        }
        self.push_raw(item);
    }

    fn is_message_start(event: &StreamEvent) -> bool {
        matches!(
            event,
            StreamEvent::BlockStart {
                kind: BlockKind::Message,
                ..
            }
        )
    }

    fn is_text_event(event: &StreamEvent) -> bool {
        matches!(
            event,
            StreamEvent::BlockStart {
                kind: BlockKind::Text { .. },
                ..
            } | StreamEvent::BlockDelta {
                delta: Delta::Text { .. } | Delta::TextMeta { .. },
                ..
            } | StreamEvent::BlockEnd {
                end: BlockClose::Text,
                ..
            }
        )
    }

    fn is_reasoning_event(event: &StreamEvent) -> bool {
        matches!(
            event,
            StreamEvent::BlockStart {
                kind: BlockKind::Reasoning { .. },
                ..
            } | StreamEvent::BlockDelta {
                delta: Delta::Reasoning { .. },
                ..
            } | StreamEvent::BlockEnd {
                end: BlockClose::Reasoning { .. },
                ..
            }
        )
    }

    fn push_raw(&mut self, item: Result<StreamEvent, CompletionError>) {
        if let Ok(event) = &item
            && let Some(id) = event.block_id()
        {
            match event {
                StreamEvent::BlockStart { .. } => {
                    self.opened.insert(id.clone());
                }
                StreamEvent::BlockEnd { .. } => {
                    self.opened.remove(id);
                }
                // A delta neither opens nor closes; `Final`/`Unknown` carry
                // no block id and never reach this arm. Exhaustive on
                // purpose: a future block-carrying variant must land here,
                // not bypass the `opened` bookkeeping.
                StreamEvent::BlockDelta { .. }
                | StreamEvent::Final(_)
                | StreamEvent::Unknown(_) => {}
            }
            // Any non-text block event is a boundary for anonymous text, any
            // non-reasoning one for anonymous reasoning.
            if !Self::is_text_event(event) && !Self::is_message_start(event) {
                self.active_text = None;
            }
            if !Self::is_reasoning_event(event) && !Self::is_message_start(event) {
                self.active_reasoning = None;
            }
        }
        self.items.push(item);
    }

    /// Push an in-band error item.
    pub fn error(&mut self, error: CompletionError) {
        self.items.push(Err(error));
    }

    /// Iterate the buffered items.
    pub fn iter(&self) -> std::slice::Iter<'_, Result<StreamEvent, CompletionError>> {
        self.items.iter()
    }

    /// Drain the buffered items, keeping the block bookkeeping.
    pub fn drain(&mut self) -> std::vec::Drain<'_, Result<StreamEvent, CompletionError>> {
        self.items.drain(..)
    }

    /// Whether nothing is buffered.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Number of buffered items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Take the buffered items.
    pub fn into_items(self) -> Vec<Result<StreamEvent, CompletionError>> {
        self.items
    }

    fn open_if_unseen(&mut self, id: &BlockId, kind: BlockKind) {
        if !self.opened.contains(id) {
            self.push(Ok(StreamEvent::BlockStart {
                id: id.clone(),
                kind,
            }));
        }
    }

    /// A bare text delta: lands in the active text block, opening a minted
    /// one if none is active.
    pub fn text(&mut self, text: impl Into<String>) {
        let id = self.active_text_id();
        self.push(Ok(StreamEvent::BlockDelta {
            id,
            delta: Delta::Text { text: text.into() },
        }));
    }

    /// A bare reasoning delta: lands in the active reasoning block, opening
    /// a minted one if none is active (the reasoning counterpart of
    /// [`text`](Self::text); any text or tool block is a boundary, and a
    /// block opened this way is closed at its boundary).
    pub fn reasoning(&mut self, text: impl Into<String>) {
        let id = match &self.active_reasoning {
            Some(id) => id.clone(),
            None => {
                let id = self
                    .reasoning_ids
                    .get_or_insert_with(|| SyntheticIds::new(MintKind::Reasoning))
                    .mint();
                self.push(Ok(StreamEvent::BlockStart {
                    id: id.clone(),
                    kind: BlockKind::Reasoning { provider_id: None },
                }));
                self.active_reasoning = Some(id.clone());
                self.auto_reasoning = Some(id.clone());
                id
            }
        };
        self.push(Ok(StreamEvent::BlockDelta {
            id,
            delta: Delta::Reasoning { text: text.into() },
        }));
    }

    /// Close the blocks bare deltas opened (text, reasoning), so a terminal
    /// record never follows a block this output opened.
    pub fn close_active_blocks(&mut self) {
        if let Some(id) = self.auto_reasoning.take() {
            self.active_reasoning = None;
            self.push_raw(Ok(StreamEvent::BlockEnd {
                id,
                end: BlockClose::Reasoning {
                    reasoning: None,
                    signature: None,
                    wire_sent: false,
                },
                block: None,
            }));
        }
        if let Some(id) = self.auto_text.take() {
            self.active_text = None;
            self.push_raw(Ok(StreamEvent::BlockEnd {
                id,
                end: BlockClose::Text,
                block: None,
            }));
        }
    }

    /// Provider metadata for the active text block (opening a minted one if
    /// none is active).
    pub fn text_meta(&mut self, additional_params: crate::message::AdditionalParams) {
        let id = self.active_text_id();
        self.push(Ok(StreamEvent::BlockDelta {
            id,
            delta: Delta::TextMeta { additional_params },
        }));
    }

    /// Open (or reactivate) the text block identified by `id`; later bare
    /// text deltas extend it.
    pub fn text_start(
        &mut self,
        id: BlockId,
        additional_params: Option<crate::message::AdditionalParams>,
    ) {
        self.push(Ok(StreamEvent::BlockStart {
            id: id.clone(),
            kind: BlockKind::Text { additional_params },
        }));
        self.active_text = Some(id);
    }

    /// Close the text block identified by `id`: later bare text deltas open
    /// a fresh block instead of extending it.
    pub fn text_end(&mut self, id: BlockId) {
        if self.active_text.as_ref() == Some(&id) {
            self.active_text = None;
        }
        if self.auto_text.as_ref() == Some(&id) {
            self.auto_text = None;
        }
        self.push(Ok(StreamEvent::BlockEnd {
            id,
            end: BlockClose::Text,
            block: None,
        }));
    }

    fn active_text_id(&mut self) -> BlockId {
        if let Some(id) = &self.active_text {
            return id.clone();
        }
        let id = self.text_ids.get_or_insert_with(SyntheticIds::text).mint();
        self.push(Ok(StreamEvent::BlockStart {
            id: id.clone(),
            kind: BlockKind::Text {
                additional_params: None,
            },
        }));
        self.active_text = Some(id.clone());
        self.auto_text = Some(id.clone());
        id
    }

    /// Open the tool-call block `id` (a no-op when already open).
    pub fn tool_start(&mut self, id: &BlockId) {
        self.open_if_unseen(id, BlockKind::ToolCall);
    }

    /// A streamed tool-name fragment for the call `id`.
    pub fn tool_name(&mut self, id: &BlockId, name: impl Into<String>) {
        self.open_if_unseen(id, BlockKind::ToolCall);
        self.push(Ok(StreamEvent::BlockDelta {
            id: id.clone(),
            delta: Delta::ToolName { name: name.into() },
        }));
    }

    /// A streamed argument fragment for the call `id`.
    pub fn tool_arguments(&mut self, id: &BlockId, arguments: impl Into<String>) {
        self.open_if_unseen(id, BlockKind::ToolCall);
        self.push(Ok(StreamEvent::BlockDelta {
            id: id.clone(),
            delta: Delta::ToolArguments {
                arguments: arguments.into(),
            },
        }));
    }

    /// End the call `id`: the accumulator finalizes the assembled fragments
    /// (or `end`'s authoritative payload) into a completed call.
    pub fn tool_end(&mut self, id: BlockId, end: ToolCallEnd) {
        self.open_if_unseen(&id, BlockKind::ToolCall);
        self.push(Ok(StreamEvent::BlockEnd {
            id,
            end: BlockClose::ToolCall(end),
            block: None,
        }));
    }

    /// A tool call the wire delivered whole: its start and its authoritative
    /// end in one step.
    pub fn tool_call(&mut self, id: BlockId, end: ToolCallEnd) {
        self.tool_end(id, end);
    }

    /// Open the reasoning block `id` (a no-op when already open).
    pub fn reasoning_start(&mut self, id: &BlockId, provider_id: Option<String>) {
        self.open_if_unseen(id, BlockKind::Reasoning { provider_id });
    }

    /// A reasoning text fragment for the block `id`, opening it (with
    /// `provider_id`) if unseen.
    pub fn reasoning_delta(
        &mut self,
        id: &BlockId,
        provider_id: Option<String>,
        text: impl Into<String>,
    ) {
        self.open_if_unseen(id, BlockKind::Reasoning { provider_id });
        self.push(Ok(StreamEvent::BlockDelta {
            id: id.clone(),
            delta: Delta::Reasoning { text: text.into() },
        }));
    }

    /// Close the reasoning block `id`. `reasoning` is the wire's
    /// authoritative restatement, `signature` a provider signature closing
    /// the block, `wire_sent` whether the wire itself sent the end.
    pub fn reasoning_end(
        &mut self,
        id: BlockId,
        reasoning: Option<crate::message::Reasoning>,
        signature: Option<String>,
        wire_sent: bool,
    ) {
        // A restatement is a whole block: open it under its provider id so
        // every published block has a start. A payload-less or
        // signature-only end for an unseen id gets none — the accumulator
        // creates no part for the former, and a start would publish an
        // empty block ahead of the latter's signature-only part.
        if let Some(reasoning) = &reasoning {
            self.open_if_unseen(
                &id,
                BlockKind::Reasoning {
                    provider_id: reasoning.id.clone(),
                },
            );
        }
        self.push(Ok(StreamEvent::BlockEnd {
            id,
            end: BlockClose::Reasoning {
                reasoning,
                signature,
                wire_sent,
            },
            block: None,
        }));
    }

    /// A whole reasoning block: open + authoritative restatement + close.
    pub fn reasoning_block(
        &mut self,
        id: BlockId,
        provider_id: Option<String>,
        content: crate::message::ReasoningContent,
    ) {
        self.open_if_unseen(
            &id,
            BlockKind::Reasoning {
                provider_id: provider_id.clone(),
            },
        );
        self.push(Ok(StreamEvent::BlockEnd {
            id,
            end: BlockClose::Reasoning {
                reasoning: Some(crate::message::Reasoning {
                    id: provider_id,
                    content: vec![content],
                }),
                signature: None,
                wire_sent: true,
            },
            block: None,
        }));
    }

    /// The provider-assigned message id (a `Message` block start).
    pub fn message_id(&mut self, id: impl Into<String>) {
        self.push(Ok(StreamEvent::BlockStart {
            id: BlockId::wire(id),
            kind: BlockKind::Message,
        }));
    }

    /// The provider's terminal record; the driver stops consuming after it.
    pub fn final_record(&mut self, record: StreamFinal) {
        self.push(Ok(StreamEvent::Final(record)));
    }

    /// An unmodeled provider item on the passthrough channel.
    pub fn unknown(&mut self, payload: UnknownPayload) {
        self.push(Ok(StreamEvent::Unknown(payload)));
    }
}

pub trait WireAdapter {
    /// The transport frame this adapter classifies: [`WireFrame`] for byte
    /// wires (SSE, NDJSON, websocket), the SDK's own event type for
    /// typed-transport wires (bedrock's Converse events, gemini-grpc's
    /// protobuf responses, candle's in-process generation events).
    type Frame;
    /// The wire's typed event, produced by the `wire.rs` classifier.
    type Event;

    /// Decode + classify one transport frame. MUST delegate to a `wire.rs`
    /// classifier (`classify_tagged_frame` / `classify_chat_completions_frame`
    /// / `classify_untyped_line` / `classify_typed_event`) — never raw serde,
    /// so the decode-then-validate policy cannot be re-derived per adapter.
    fn classify(&self, frame: Self::Frame) -> WireEvent<Self::Event>;

    /// Map one `Known` event to canonical grammar events. Stateful: index→id
    /// maps, open-block state, id fabrication, and wire-quirk quarantine live
    /// here — policy for unknown/corrupt frames does not (the driver owns it).
    ///
    /// Pushing a [`StreamEvent::Final`] (the adapter maps its native
    /// terminal record itself, serializing it onto [`StreamFinal::raw`])
    /// marks the provider's genuine terminal; the driver stops consuming
    /// after yielding it.
    fn interpret(&mut self, event: Self::Event, out: &mut AdapterOutput);

    /// End-of-stream flush on EOF without a terminal (close open blocks).
    ///
    /// Never runs after a transport error (truncation drops partials) or after
    /// a terminal was interpreted. Must not synthesize a terminal record: EOF
    /// without the provider's own end event is truncation, and a fabricated
    /// terminal would read as a successfully completed turn. (A terminal the
    /// provider *did* signal earlier — e.g. the chat-completions `[DONE]`
    /// sentinel or a `finish_reason` chunk, whose usage trailer arrives later —
    /// may be emitted here; that is deferral, not synthesis.)
    fn finish(&mut self, out: &mut AdapterOutput);

    /// Flush content the provider fully delivered before a terminal error item
    /// (a transport failure or an in-band provider error envelope) reaches the
    /// consumer.
    ///
    /// Default: nothing — truncation drops partials. Wires that buffer
    /// fully-delivered tool calls (the chat-completions compat family, the
    /// Responses SSE loop) override this so a first-`Err`-stop consumer still
    /// sees them. Must not push a terminal record.
    fn flush_before_terminal_error(&mut self, _out: &mut AdapterOutput) {}

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
    /// as [`StreamEvent::Unknown`] where the surface has a raw channel
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
pub fn run_wire_stream<A, S>(transport: S, mut adapter: A) -> StreamingResult
where
    A: WireAdapter + WasmCompatSend + 'static,
    A::Frame: WasmCompatSend,
    A::Event: WasmCompatSend,
    S: Stream<Item = Result<A::Frame, CompletionError>> + WasmCompatSend + 'static,
{
    Box::pin(async_stream::stream! {
        let mut transport = Box::pin(transport);
        let mut out = AdapterOutput::new();
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
                    for item in out.drain() {
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
                    out.unknown(value);
                }
                Err(error) => {
                    yield Err(error);
                }
            }

            #[cfg(any(test, debug_assertions))]
            sequence_laws.check_batch(&out);

            let saw_terminal = out
                .iter()
                .any(|item| matches!(item, Ok(StreamEvent::Final(_))));
            for item in out.drain() {
                yield item;
            }
            if saw_terminal || adapter.is_finished() {
                return;
            }
        }

        adapter.finish(&mut out);
        #[cfg(any(test, debug_assertions))]
        sequence_laws.check_batch(&out);
        for item in out.drain() {
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
) -> Result<Vec<StreamEvent>, CompletionError>
where
    A: WireAdapter,
{
    let mut out = AdapterOutput::new();
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
fn drain_buffered(
    out: &mut AdapterOutput,
    choices: &mut Vec<StreamEvent>,
) -> Result<bool, CompletionError> {
    let mut saw_terminal = false;
    for item in out.drain() {
        let choice = item?;
        saw_terminal |= matches!(choice, StreamEvent::Final(_));
        choices.push(choice);
    }
    Ok(saw_terminal)
}
