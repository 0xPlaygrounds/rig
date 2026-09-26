//! Completion event output and response folding. [`AdapterOutput`] manages
//! block boundaries for provider decoders, including typed transports.
//!
//! ```
//! use rig_core::operation::AdapterOutput;
//!
//! let mut output = AdapterOutput::self_closing();
//! output.text("Hello");
//! output.close_active_blocks();
//! assert_eq!(output.len(), 3);
//! ```

use crate::completion::{CompletionRequest, CompletionResponse};
use crate::error::ProviderError;
use crate::streaming::{
    BlockAccumulator, BlockClose, BlockId, BlockKind, Delta, Finalized, MintKind, StreamEvent,
    StreamFinal, SyntheticIds, ToolCallEnd, UnknownPayload,
};
use crate::telemetry::{GenAiOperation, SpanBuilder, SpanCombinator};
use crate::wire::{Fold, Mode, Operation, Reply, Sink};

/// Generating an assistant turn, unary or streamed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Completion;

/// Debug-mode sequence laws over what a decoder actually emitted.
type Laws = crate::providers::internal::sequence_law::SequenceLaws;

impl Operation for Completion {
    type Request = CompletionRequest;
    type Event = StreamEvent;
    type Response = CompletionResponse;
    type Capabilities = crate::completion::ProviderCapabilities;
    type Output = AdapterOutput;
    type Fold = CompletionFold;
    type Telemetry = GenAiOperation;

    const NAME: &'static str = "completion";

    fn is_terminal(event: &Self::Event) -> bool {
        matches!(event, StreamEvent::Final(_))
    }

    fn telemetry(mode: Mode) -> Self::Telemetry {
        match mode {
            Mode::Streaming => GenAiOperation::ChatStreaming,
            Mode::Unary => GenAiOperation::Chat,
        }
    }

    /// The fold names the provider and the reasoning issuer up front, so
    /// reasoning streamed before the terminal record records its issuer.
    fn fold<W: crate::wire::Wire<Op = Self>>(
        request: &Self::Request,
        wire: &W,
        mode: Mode,
    ) -> Self::Fold {
        let issuer = wire
            .reasoning_issuer(request.model.as_deref().or(wire.id()))
            .map(str::to_owned);
        CompletionFold::opened(wire.name(), issuer, mode)
    }

    /// The transport request id fills a gap on the terminal record.
    fn stamp_event(event: &mut Self::Event, reply: &Reply) {
        if let StreamEvent::Final(terminal) = event
            && terminal.provider_request_id.is_none()
        {
            terminal
                .provider_request_id
                .clone_from(&reply.provider_request_id);
        }
    }

    /// The terminal record is what a stream records.
    fn record_event(span: &tracing::Span, event: &Self::Event) {
        if let StreamEvent::Final(terminal) = event {
            span.record_response(
                terminal
                    .response_id
                    .as_deref()
                    .or(terminal.message_id.as_deref()),
                terminal.model.as_deref(),
                &terminal.usage,
            );
        }
    }

    /// Reasoning another wire issued is omitted; see
    /// [`crate::message::retain_replayable_reasoning`]. The request's model
    /// override, when it names one, is the model the wire replays for.
    fn scope_to_wire<W: crate::wire::Wire<Op = Self>>(request: &mut Self::Request, wire: &W) {
        let model = request.model.as_deref().or(wire.id());
        let Some(issuers) = wire.replay_issuers(model) else {
            return;
        };
        let issuers: Vec<&str> = issuers.iter().map(String::as_str).collect();
        crate::message::retain_replayable_reasoning(&mut request.chat_history, &issuers);
    }

    fn span(
        provider: &str,
        model: Option<&str>,
        telemetry: Self::Telemetry,
        request: &Self::Request,
    ) -> tracing::Span {
        // The request's override is the model actually sent (every wire
        // honours it on encode), so it is the one the span names.
        let model = request.model.as_deref().or(model).unwrap_or_default();
        debug_assert!(telemetry.is_completion());
        SpanBuilder::new(provider, model, telemetry)
            .system_instructions(
                request.system_instructions(),
                request.record_telemetry_content,
            )
            .build()
    }

    // Both prefer the response identity, falling back to the message identity.
    fn record(span: &tracing::Span, response: &Self::Response) {
        span.record_response(
            response
                .response_id
                .as_deref()
                .or(response.message_id.as_deref()),
            response.model.as_deref(),
            &response.usage,
        );
    }
}

impl Sink<Completion> for AdapterOutput {
    type Laws = Laws;

    fn push(&mut self, item: Result<StreamEvent, ProviderError>) {
        AdapterOutput::push(self, item);
    }

    fn drain(&mut self) -> std::vec::Drain<'_, Result<StreamEvent, ProviderError>> {
        AdapterOutput::drain(self)
    }

    fn items(&self) -> &[Result<StreamEvent, ProviderError>] {
        AdapterOutput::items(self)
    }

    /// Forwarded on the passthrough channel, never folded into the choice.
    fn unknown(&mut self, payload: UnknownPayload) {
        AdapterOutput::unknown(self, payload);
    }

    fn check_laws(&self, laws: &mut Self::Laws) {
        #[cfg(any(test, debug_assertions))]
        laws.check_batch(self);
        #[cfg(not(any(test, debug_assertions)))]
        let _ = laws;
    }

    /// EOF without a terminal: close what is still open.
    fn finish(&mut self) {
        self.close_open_blocks();
    }
}

/// The fold from a completion reply's canonical events to its response.
/// It collects the blocks their ends carry, the terminal record and the
/// message id, and assembles nothing: the sink did.
#[derive(Debug, Default)]
pub struct CompletionFold {
    /// One slot per block, in the order the blocks started; filled by the
    /// end that carries the block.
    slots: Vec<Option<crate::message::AssistantContent>>,
    /// Keys whose slot is open (started, not ended).
    open: std::collections::HashMap<BlockId, usize>,
    /// Keys whose slot is filled: a later end under the key restates it.
    filled: std::collections::HashMap<BlockId, usize>,
    terminal: Option<StreamFinal>,
    message_id: Option<String>,
    /// The provider a streamed response names: the opener's, or the
    /// terminal record's for a relayed stream. A unary response names the
    /// reply's.
    provider: String,
    /// Whether the terminal record names the provider: a stream relayed
    /// over the bus is opened under the handler's label.
    provider_from_terminal: bool,
    /// The issuer of this reply's reasoning when a wire names it before the
    /// terminal record.
    reasoning_issuer: Option<String>,
    /// Whether the reply arrived whole. A whole reply that named no
    /// terminal is the provider answering with nothing; a stream that ended
    /// the same way was cut short and is refused.
    whole: bool,
}

impl CompletionFold {
    /// The fold of a reply `provider` opened in `mode`, naming the reasoning
    /// issuer it knows up front. [`Operation::fold`] builds it from the
    /// wire; a caller folding a reply it decoded itself names the provider.
    pub fn opened(
        provider: impl Into<String>,
        reasoning_issuer: Option<String>,
        mode: Mode,
    ) -> Self {
        Self {
            provider: provider.into(),
            reasoning_issuer,
            whole: mode == Mode::Unary,
            ..Self::default()
        }
    }

    /// The fold of a stream relayed under `label`, whose terminal record
    /// names the provider behind it.
    pub(crate) fn relayed(label: impl Into<String>) -> Self {
        Self {
            provider: label.into(),
            provider_from_terminal: true,
            ..Self::default()
        }
    }

    /// The provider this fold's response names.
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// The provider's normalized terminal record, `None` until it arrives
    /// (and forever on truncation or a terminal error).
    pub fn terminal(&self) -> Option<&StreamFinal> {
        self.terminal.as_ref()
    }

    /// The provider-assigned message id, from a message block or the
    /// terminal record.
    pub fn message_id(&self) -> Option<&str> {
        self.message_id.as_deref()
    }

    /// The blocks finalized so far, in the order they started. An open
    /// block is not in it: its end brings it.
    pub fn snapshot(&self) -> Vec<crate::message::AssistantContent> {
        self.slots.iter().flatten().cloned().collect()
    }

    /// The slot a block under `id` fills: the one its start opened, else the
    /// one an earlier end under the key filled, else a new one.
    fn slot(&mut self, id: &BlockId) -> usize {
        let index = self
            .open
            .remove(id)
            .or_else(|| self.filled.get(id).copied())
            .unwrap_or_else(|| {
                self.slots.push(None);
                self.slots.len() - 1
            });
        self.filled.insert(id.clone(), index);
        index
    }

    /// Open a slot for `id` unless one is open: the block's place in the
    /// choice is where it started.
    fn open(&mut self, id: &BlockId) {
        if !self.open.contains_key(id) {
            self.slots.push(None);
            self.open.insert(id.clone(), self.slots.len() - 1);
        }
    }

    /// Terminal usage, or [`Usage::default`](crate::completion::Usage)
    /// before a terminal record.
    pub fn usage(&self) -> crate::completion::Usage {
        self.terminal
            .as_ref()
            .map(|terminal| terminal.usage)
            .unwrap_or_default()
    }

    /// Response identity. A message-start id takes precedence over the
    /// terminal's; response and transport ids require a terminal record.
    pub fn identity(&self) -> crate::completion::ResponseIdentity {
        crate::completion::ResponseIdentity {
            message_id: self.message_id.clone(),
            ..self
                .terminal
                .as_ref()
                .map(StreamFinal::identity)
                .unwrap_or_default()
        }
    }

    /// The issuer this reply's reasoning records: the terminal record's
    /// once it has arrived; before it, the issuer named up front, else the
    /// provider that opened the stream. `None` before the terminal of a
    /// relayed stream, whose label names a handler, not an issuer.
    pub fn reasoning_issuer(&self) -> Option<&str> {
        match &self.terminal {
            Some(terminal) => Some(terminal.issuer()),
            None => self
                .reasoning_issuer
                .as_deref()
                .or((!self.provider_from_terminal).then_some(self.provider.as_str())),
        }
    }
}

impl Fold<Completion> for CompletionFold {
    fn absorb(&mut self, event: &StreamEvent) -> Result<(), ProviderError> {
        match event {
            StreamEvent::BlockStart {
                id,
                kind: BlockKind::Message,
            } => {
                // The wire announced the assistant message's own id; it
                // outranks the terminal record's.
                if let Some(message_id) = id.wire_str() {
                    self.message_id = Some(message_id.to_owned());
                }
            }
            StreamEvent::BlockStart { id, .. } | StreamEvent::BlockDelta { id, .. } => {
                self.open(id);
            }
            StreamEvent::BlockEnd { id, block, .. } => {
                let index = self.slot(id);
                if let (Some(block), Some(slot)) = (block, self.slots.get_mut(index)) {
                    *slot = Some(block.clone());
                }
            }
            StreamEvent::Final(response) if self.terminal.is_none() => {
                // An explicit message-id block keeps precedence; the terminal
                // record only fills a gap.
                if self.message_id.is_none() {
                    self.message_id.clone_from(&response.message_id);
                }
                if self.provider_from_terminal && !response.provider.is_empty() {
                    self.provider.clone_from(&response.provider);
                }
                self.terminal = Some(response.clone());
            }
            _ => {}
        }
        Ok(())
    }

    /// The turn: the collected blocks with the terminal record's usage and
    /// metadata. A whole reply's document is the response's `raw`; a
    /// stream's is its terminal record's. A stream that produced no
    /// terminal record was cut short and is refused.
    fn finish(self, reply: Reply) -> Result<CompletionResponse, ProviderError> {
        let terminal = self.terminal.as_ref();
        if !self.whole && terminal.is_none() {
            return Err(ProviderError::Response(
                "provider stream ended without a terminal record; treating the turn as truncated"
                    .to_owned(),
            ));
        }
        let issuer = terminal.map_or(self.provider.clone(), |terminal| {
            terminal.issuer().to_owned()
        });
        let raw = match (self.whole, terminal) {
            (false, Some(terminal)) => terminal.raw.clone(),
            _ => reply.raw,
        };
        let response = crate::streaming::fold_finish(
            self.snapshot(),
            terminal,
            self.message_id.clone(),
            self.provider.clone(),
            &issuer,
            raw,
        );
        // The terminal's own id wins; the reply headers only fill a gap.
        if response.provider_request_id.is_none() {
            Ok(response.with_optional_provider_request_id(reply.provider_request_id))
        } else {
            Ok(response)
        }
    }
}

/// The completion sink: what a decoder writes through, and what makes its
/// events canonical. Helpers open unseen tool and reasoning keys before
/// deltas; bare text uses an active key, minting a new one after non-text
/// block events other than message starts. Every `BlockEnd` it emits
/// carries the block it finalized, the terminal's finish reason is
/// reconciled with the completed tool calls, a second terminal is dropped,
/// blocks still open at the terminal or at EOF are closed, and a malformed
/// block is an in-band error item. Frame-classification errors are handled
/// by the driver.
#[derive(Debug, Default)]
pub struct AdapterOutput {
    items: Vec<Result<StreamEvent, ProviderError>>,
    /// Assembles every block, so its end can carry it.
    accumulator: BlockAccumulator,
    /// Whether the terminal record was emitted; a second one is a provider
    /// defect and is dropped.
    terminal: bool,
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
    /// Automatically opened text block eligible for synthesized closure.
    /// Explicitly opened blocks remain the provider's responsibility.
    auto_text: Option<BlockId>,
    auto_reasoning: Option<BlockId>,
    /// Whether nonmatching block events close automatically opened blocks.
    /// Disabled by default so provider adapters control explicit boundaries.
    self_closing: bool,
    /// Whether events are kept as the helpers wrote them, uncanonicalized:
    /// what a transport scripts a reply with, leaving canonicalization to
    /// the driver's sink.
    scripted: bool,
    /// Blocks a start was emitted for (or that a delta opened leniently),
    /// in order, so a delta never precedes its block's start on the wire we
    /// emit and the reply's end can close what is still open.
    opened: Vec<(BlockId, Opened)>,
}

/// What kind of block an open key is, for closing it at the reply's end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Opened {
    Message,
    Text,
    Reasoning,
    ToolCall,
}

impl Opened {
    fn of(event: &StreamEvent) -> Option<Self> {
        match event {
            StreamEvent::BlockStart { kind, .. } => Some(match kind {
                BlockKind::Message => Self::Message,
                BlockKind::Text { .. } => Self::Text,
                BlockKind::Reasoning { .. } => Self::Reasoning,
                BlockKind::ToolCall => Self::ToolCall,
            }),
            StreamEvent::BlockDelta { delta, .. } => Some(match delta {
                Delta::Text { .. } | Delta::TextMeta { .. } => Self::Text,
                Delta::Reasoning { .. } => Self::Reasoning,
                Delta::ToolName { .. } | Delta::ToolArguments { .. } => Self::ToolCall,
            }),
            StreamEvent::BlockEnd { .. } | StreamEvent::Final(_) | StreamEvent::Unknown(_) => None,
        }
    }
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

    /// An output whose events are the helpers' own, uncanonicalized: what
    /// a transport behind a [`Local`](crate::driver::Local) wire scripts a
    /// reply with. The driver's sink canonicalizes them once, so a malformed
    /// block scripted here reaches the consumer as the in-band error it
    /// would be on a provider's stream.
    pub fn scripted() -> Self {
        Self {
            scripted: true,
            ..Self::default()
        }
    }

    /// Push one event verbatim. A block this output opened itself for a
    /// bare delta is closed first when `item` is its boundary.
    pub fn push(&mut self, item: Result<StreamEvent, ProviderError>) {
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

    fn push_raw(&mut self, item: Result<StreamEvent, ProviderError>) {
        if let Ok(event) = &item
            && let Some(id) = event.block_id()
        {
            match event {
                StreamEvent::BlockStart { .. } | StreamEvent::BlockDelta { .. } => {
                    if let Some(kind) = Opened::of(event)
                        && !self.opened.iter().any(|(open, _)| open == id)
                    {
                        self.opened.push((id.clone(), kind));
                    }
                }
                StreamEvent::BlockEnd { .. } => {
                    self.opened.retain(|(open, _)| open != id);
                }
                // `Final`/`Unknown` carry no block id and never reach this
                // arm. Exhaustive on purpose: a future block-carrying
                // variant must land here, not bypass the bookkeeping.
                StreamEvent::Final(_) | StreamEvent::Unknown(_) => {}
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
        let item = if self.scripted {
            Some(item)
        } else {
            self.canonical(item)
        };
        if let Some(item) = item {
            self.items.push(item);
        }
    }

    /// The event as every consumer sees it: an end carries the block it
    /// finalized, a terminal follows every close and reconciles its finish
    /// reason with the completed tool calls, a second terminal is dropped,
    /// and a malformed block is an error item.
    fn canonical(
        &mut self,
        item: Result<StreamEvent, ProviderError>,
    ) -> Option<Result<StreamEvent, ProviderError>> {
        let event = match item {
            Ok(event) => event,
            Err(error) => return Some(Err(error)),
        };
        match event {
            StreamEvent::Final(mut response) => {
                if self.terminal {
                    return None;
                }
                self.close_open_blocks();
                self.terminal = true;
                response.finish_reason = response
                    .finish_reason
                    .map(|reason| reason.reconcile_with_output(self.accumulator.saw_tool_call()));
                Some(Ok(StreamEvent::Final(response)))
            }
            StreamEvent::BlockStart {
                kind: BlockKind::Message,
                ..
            }
            | StreamEvent::Unknown(_) => Some(Ok(event)),
            event => match self.accumulator.apply(&event) {
                // An end that finalized a block publishes it under the key
                // its deltas carried.
                Ok(Some(Finalized { id, block, fresh })) => match event {
                    StreamEvent::BlockEnd { end, .. } => {
                        // A sibling under a finished key is a new block: it
                        // opens before it closes, so a consumer keeps the
                        // key's earlier block as well.
                        if fresh && !self.opened.iter().any(|(open, _)| open == &id) {
                            let provider_id = match &block {
                                crate::message::AssistantContent::Reasoning(reasoning) => {
                                    reasoning.id.clone()
                                }
                                _ => None,
                            };
                            self.opened.push((id.clone(), Opened::Reasoning));
                            self.items.push(Ok(StreamEvent::BlockStart {
                                id: id.clone(),
                                kind: BlockKind::Reasoning { provider_id },
                            }));
                        }
                        self.opened.retain(|(open, _)| open != &id);
                        Some(Ok(StreamEvent::BlockEnd {
                            id,
                            end,
                            block: Some(block),
                        }))
                    }
                    // Only ends finalize; the accumulator upholds it.
                    event => Some(Ok(event)),
                },
                Ok(None) => Some(Ok(event)),
                Err(error) => Some(Err(error)),
            },
        }
    }

    /// Close every text and reasoning block still open, so a terminal
    /// record never follows an open block and nothing a decoder wrote is
    /// lost. An open tool call stays open: its input never fully arrived.
    fn close_open_blocks(&mut self) {
        let open: Vec<(BlockId, Opened)> = self.opened.clone();
        for (id, kind) in open {
            match kind {
                Opened::Text => {
                    if self.active_text.as_ref() == Some(&id) {
                        self.active_text = None;
                    }
                    if self.auto_text.as_ref() == Some(&id) {
                        self.auto_text = None;
                    }
                    self.push_raw(Ok(StreamEvent::BlockEnd {
                        id,
                        end: BlockClose::Text,
                        block: None,
                    }));
                }
                Opened::Reasoning => {
                    if self.active_reasoning.as_ref() == Some(&id) {
                        self.active_reasoning = None;
                    }
                    if self.auto_reasoning.as_ref() == Some(&id) {
                        self.auto_reasoning = None;
                    }
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
                Opened::Message | Opened::ToolCall => {}
            }
        }
    }

    /// Push an in-band error item.
    pub fn error(&mut self, error: ProviderError) {
        self.items.push(Err(error));
    }

    /// What this output holds, without taking it.
    pub fn items(&self) -> &[Result<StreamEvent, ProviderError>] {
        &self.items
    }

    /// Iterate the buffered items.
    pub fn iter(&self) -> std::slice::Iter<'_, Result<StreamEvent, ProviderError>> {
        self.items.iter()
    }

    /// Drain the buffered items, keeping the block bookkeeping.
    pub fn drain(&mut self) -> std::vec::Drain<'_, Result<StreamEvent, ProviderError>> {
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
    pub fn into_items(self) -> Vec<Result<StreamEvent, ProviderError>> {
        self.items
    }

    fn open_if_unseen(&mut self, id: &BlockId, kind: BlockKind) {
        if !self.opened.iter().any(|(open, _)| open == id) {
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

    /// Appends to active reasoning, minting a block when needed.
    /// Automatic boundary closure requires [`Self::self_closing`].
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

    /// Close the active text block, if any: the next text opens a new one.
    /// For wires whose part boundaries matter, such as a signed part that
    /// must return on its own.
    pub fn end_active_text(&mut self) {
        if let Some(id) = self.active_text.clone() {
            self.text_end(id);
        }
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
        // Restatements need a start; bare or signature-only ends must not
        // introduce a separate empty reasoning part.
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
                    provider: None,
                    id: provider_id,
                    content: vec![content],
                }),
                signature: None,
                wire_sent: true,
            },
            block: None,
        }));
    }

    /// Emit a whole reply's parts as the events a stream sends for them: one
    /// complete block per part, in order. `images` says how an image part
    /// travels.
    pub fn content(&mut self, choice: &[crate::message::AssistantContent], images: ImagePart) {
        use crate::message::AssistantContent;

        for (index, content) in choice.iter().enumerate() {
            let index = index as u64;
            match content {
                AssistantContent::Text(text) => {
                    let id = BlockId::minted(MintKind::Text, index);
                    self.text_start(id.clone(), text.additional_params.clone());
                    self.text(text.text.clone());
                    self.text_end(id);
                }
                AssistantContent::Reasoning(reasoning) => {
                    let id = reasoning
                        .id
                        .as_deref()
                        .map(BlockId::wire)
                        .unwrap_or_else(|| BlockId::minted(MintKind::Reasoning, index));
                    self.reasoning_end(id, Some(reasoning.clone()), None, true);
                }
                AssistantContent::Image(image) => match images {
                    ImagePart::Block => self.push(Ok(StreamEvent::BlockEnd {
                        id: BlockId::minted(MintKind::Block, index),
                        end: BlockClose::Image(image.clone()),
                        block: None,
                    })),
                    ImagePart::Unknown => match serde_json::to_value(image) {
                        Ok(value) => self.unknown(UnknownPayload::new(value)),
                        Err(error) => self.error(ProviderError::Json(error)),
                    },
                },
                AssistantContent::ToolCall(call) => {
                    // The durable handle is separate from the assembly key and
                    // provider metadata. Local names are never inferred to be
                    // wire IDs merely because they do not look minted.
                    let mut end = ToolCallEnd::whole(
                        call.function.name.clone(),
                        call.function.arguments.clone(),
                    )
                    .with_durable_id(call.id.clone())
                    .with_signature(call.signature.clone())
                    .with_additional_params(call.additional_params.clone());
                    if let Some(provider) = &call.provider {
                        end = match &provider.item_id {
                            Some(item_id) => end
                                .with_call_id(provider.call_id.clone())
                                .with_tool_id(item_id.clone()),
                            None => end.with_tool_id(provider.call_id.clone()),
                        };
                    }
                    // Re-emission creates a fresh assembly occurrence; durable
                    // identity and provider handles are preserved on `end`.
                    self.tool_end(BlockId::minted(MintKind::Tool, index), end);
                }
            }
        }
    }

    /// Emit a whole `response` as the events a stream sends for it: its
    /// message id, its parts and its terminal record.
    pub fn response(&mut self, response: &CompletionResponse, images: ImagePart) {
        if let Some(message_id) = &response.message_id {
            self.message_id(message_id.clone());
        }
        self.content(&response.choice, images);
        self.final_record(terminal_of(response));
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

/// How [`AdapterOutput::content`] emits an image part.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImagePart {
    /// Closed as a whole image block ([`BlockClose::Image`]): what a decoder
    /// emits for a reply it reads itself.
    Block,
    /// Forwarded as an unknown payload: what a bus relay emits for a
    /// completed turn, as relayed streams have always carried it.
    Unknown,
}

/// The terminal record restating `response`'s metadata.
fn terminal_of(response: &CompletionResponse) -> StreamFinal {
    let mut terminal = StreamFinal::new(
        response.provider.clone(),
        response.usage,
        response.raw.clone(),
    )
    .with_optional_finish_reason(response.finish_reason());
    terminal.message_id = response.message_id.clone();
    terminal.response_id = response.response_id.clone();
    terminal.provider_request_id = response.provider_request_id.clone();
    terminal.model = response.model.clone();
    terminal
}

#[cfg(test)]
mod tests;
