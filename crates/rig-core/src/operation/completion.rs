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

use std::collections::{HashMap, HashSet};

use crate::completion::{CompletionRequest, CompletionResponse};
use crate::error::ProviderError;
use crate::message::AssistantContent;
use crate::streaming::{
    BlockClose, BlockId, BlockKind, Delta, MintKind, StreamEvent, StreamFinal, SyntheticIds,
    ToolCallEnd, UnknownPayload,
};
use crate::telemetry::{GenAiOperation, SpanBuilder, SpanCombinator};
use crate::wire::{Fold, Mode, Operation, Reply, Sink};

mod accumulator;

use accumulator::BlockAccumulator;

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

    fn fold<W: crate::wire::Wire<Op = Self>>(
        request: &Self::Request,
        wire: &W,
        mode: Mode,
    ) -> Self::Fold {
        let issuer = wire.reasoning_issuer(request.model.as_deref().or(wire.id()));
        CompletionFold::opened(wire.name(), issuer.map(str::to_owned), mode)
    }

    /// The transport request id reaches the terminal record, unless the
    /// wire already put one there.
    fn stamp_event(event: &mut Self::Event, reply: &Reply) {
        if let StreamEvent::Final(terminal) = event
            && terminal.provider_request_id.is_none()
        {
            terminal
                .provider_request_id
                .clone_from(&reply.provider_request_id);
        }
    }

    /// A streamed call records the terminal record as it passes.
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

    fn telemetry(mode: Mode) -> Self::Telemetry {
        match mode {
            Mode::Unary => GenAiOperation::Chat,
            Mode::Streaming => GenAiOperation::ChatStreaming,
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

    /// Closes every text and reasoning block still open, so each carries
    /// its content on an end. The ends precede the terminal failure a
    /// decoder pushed last.
    fn finish(&mut self) {
        if self.terminated {
            return;
        }
        let kept = self
            .items
            .iter()
            .rposition(Result::is_ok)
            .map_or(0, |last| last + 1);
        let failures = self.items.split_off(kept);
        self.close_open_blocks();
        self.items.extend(failures);
    }

    fn check_laws(&self, laws: &mut Self::Laws) {
        #[cfg(any(test, debug_assertions))]
        laws.check_batch(self);
        #[cfg(not(any(test, debug_assertions)))]
        let _ = laws;
    }
}

/// The fold from a completion reply's canonical events to its response.
///
/// It collects what the sink already finalized: every block from its
/// `BlockEnd`, in the order the blocks began, the message id and the
/// terminal record. It assembles nothing, so a block still open is not in
/// [`Self::snapshot`]. The default is the fold of a stream relayed under no
/// label.
pub struct CompletionFold {
    /// Finalized blocks in the order they began; `None` holds the place of
    /// a block that has not ended.
    blocks: Vec<Option<AssistantContent>>,
    /// The latest slot each block key holds in `blocks`.
    slots: HashMap<BlockId, usize>,
    /// Reasoning keys whose block began and has not ended.
    open_reasoning: HashSet<BlockId>,
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
    /// What the reply's end means: a streamed reply without a terminal
    /// record was truncated, while a whole one is the provider's answer.
    mode: Mode,
}

impl Default for CompletionFold {
    fn default() -> Self {
        Self::relayed("")
    }
}

impl CompletionFold {
    /// The fold of a reply a wire opened in `mode`, under its provider name
    /// and the reasoning issuer it names up front.
    pub(crate) fn opened(
        provider: impl Into<String>,
        reasoning_issuer: Option<String>,
        mode: Mode,
    ) -> Self {
        Self {
            blocks: Vec::new(),
            slots: HashMap::new(),
            open_reasoning: HashSet::new(),
            terminal: None,
            message_id: None,
            provider: provider.into(),
            provider_from_terminal: false,
            reasoning_issuer,
            mode,
        }
    }

    /// The fold of a stream relayed under `label`, whose terminal record
    /// names the provider behind it.
    pub(crate) fn relayed(label: impl Into<String>) -> Self {
        Self {
            provider_from_terminal: true,
            ..Self::opened(label, None, Mode::Streaming)
        }
    }

    /// Hold the place of the block `id` where the sink's assembly puts it:
    /// text at its first content, reasoning when it begins, calls and
    /// images at their end.
    fn reserve(&mut self, id: &BlockId) {
        self.slots.insert(id.clone(), self.blocks.len());
        self.blocks.push(None);
    }

    fn collect(&mut self, id: &BlockId, end: &BlockClose, block: &AssistantContent) {
        let slot = match end {
            BlockClose::Text => self.slots.get(id).copied(),
            BlockClose::Reasoning { .. } => {
                self.open_reasoning.remove(id);
                self.slots.get(id).copied()
            }
            BlockClose::ToolCall(_) | BlockClose::Image(_) => None,
        };
        match slot.and_then(|slot| self.blocks.get_mut(slot)) {
            Some(held) => *held = Some(block.clone()),
            None => {
                self.slots.insert(id.clone(), self.blocks.len());
                self.blocks.push(Some(block.clone()));
            }
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

    /// The blocks finalized so far, in the order they began. A block that
    /// has not ended is not among them.
    pub fn snapshot(&self) -> Vec<AssistantContent> {
        self.blocks.iter().flatten().cloned().collect()
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
            // The wire announced the assistant message's own id; it
            // outranks the terminal record's.
            StreamEvent::BlockStart {
                id,
                kind: BlockKind::Message,
            } => {
                if let Some(message_id) = id.wire_str() {
                    self.message_id = Some(message_id.to_owned());
                }
            }
            StreamEvent::BlockStart {
                id,
                kind:
                    BlockKind::Text {
                        additional_params: Some(_),
                    },
            }
            | StreamEvent::BlockDelta {
                id,
                delta: Delta::Text { .. } | Delta::TextMeta { .. },
            } if !self.slots.contains_key(id) => self.reserve(id),
            StreamEvent::BlockStart {
                id,
                kind: BlockKind::Reasoning { .. },
            }
            | StreamEvent::BlockDelta {
                id,
                delta: Delta::Reasoning { .. },
            } if !self.open_reasoning.contains(id) => {
                self.open_reasoning.insert(id.clone());
                self.reserve(id);
            }
            StreamEvent::BlockEnd {
                id,
                end,
                block: Some(block),
            } => self.collect(id, end, block),
            // The stream's terminal record is its first: a relay may carry
            // items past it.
            StreamEvent::Final(terminal) if self.terminal.is_none() => {
                // An explicit message-id block keeps precedence; the terminal
                // record only fills a gap.
                if self.message_id.is_none() {
                    self.message_id.clone_from(&terminal.message_id);
                }
                if self.provider_from_terminal && !terminal.provider.is_empty() {
                    self.provider.clone_from(&terminal.provider);
                }
                self.terminal = Some(terminal.clone());
            }
            _ => {}
        }
        Ok(())
    }

    /// A whole reply carries its document as `raw` and names the reply's
    /// provider. A streamed one carries the terminal record's document and
    /// names the provider that opened it (the terminal's, for a relayed
    /// stream); without a terminal record it was truncated and is refused.
    fn finish(self, reply: Reply) -> Result<CompletionResponse, ProviderError> {
        let choice = self.snapshot();
        let response = match self.mode {
            Mode::Unary => {
                let issuer = self
                    .terminal
                    .as_ref()
                    .map_or(reply.provider.clone(), |terminal| {
                        terminal.issuer().to_owned()
                    });
                crate::streaming::fold_finish(
                    choice,
                    self.terminal.as_ref(),
                    self.message_id,
                    reply.provider,
                    &issuer,
                    reply.raw,
                )
            }
            Mode::Streaming => {
                let Some(terminal) = self.terminal.as_ref() else {
                    return Err(ProviderError::Response(
                        "provider stream ended without a terminal record; treating the turn \
                         as truncated"
                            .to_owned(),
                    ));
                };
                crate::streaming::fold_finish(
                    choice,
                    Some(terminal),
                    self.message_id,
                    self.provider,
                    terminal.issuer(),
                    terminal.raw.clone(),
                )
            }
        };
        // The terminal's own id wins; the reply headers only fill a gap.
        if response.provider_request_id.is_none() {
            Ok(response.with_optional_provider_request_id(reply.provider_request_id))
        } else {
            Ok(response)
        }
    }
}

/// The completion sink: buffers a decoder's events and in-band errors and
/// makes them canonical.
///
/// Helpers open unseen tool and reasoning keys before deltas. Bare text uses
/// an active key, minting a new one after non-text block events other than
/// message starts. Every event is applied to the one block assembly as it is
/// pushed, so what drains is final: each `BlockEnd` carries the block it
/// finalized (text included), a malformed complete tool input is an error
/// item in its place, the text and reasoning blocks still open are closed
/// before the terminal record, the terminal's finish reason agrees with the
/// completed tool calls, and a second terminal record is dropped. A sink
/// driven by hand rather than by the driver ends a reply with
/// [`Sink::finish`], as the driver does. Canonical items pass a second sink
/// unchanged: an error item reporting a malformed tool input finishes the
/// call it reports, as the end it replaced did. Frame-classification errors
/// are handled by the driver.
#[derive(Debug, Default)]
pub struct AdapterOutput {
    items: Vec<Result<StreamEvent, ProviderError>>,
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
    /// Blocks a start was emitted for (or that a delta opened leniently),
    /// so a delta never precedes its block's start on the wire we emit.
    opened: HashSet<BlockId>,
    /// The assembly that finalizes each block on its end.
    blocks: BlockAccumulator,
    /// Whether the terminal record was pushed.
    terminated: bool,
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
        let item = match item {
            Ok(StreamEvent::Final(mut terminal)) => {
                // A second terminal is a provider defect; the first stands.
                if self.terminated {
                    return;
                }
                self.close_open_blocks();
                self.terminated = true;
                // A `stop` that was really a tool call reads as one.
                terminal.finish_reason = terminal
                    .finish_reason
                    .map(|reason| reason.reconcile_with_output(self.blocks.saw_tool_call()));
                Ok(StreamEvent::Final(terminal))
            }
            Ok(StreamEvent::BlockEnd { id, end, .. }) => {
                // A sibling part under a finished key begins where it ends,
                // so a collector keeps both.
                if self.blocks.ends_a_sibling(&id, &end) {
                    self.push_raw(Ok(StreamEvent::BlockStart {
                        id: id.clone(),
                        kind: BlockKind::Reasoning { provider_id: None },
                    }));
                }
                let event = StreamEvent::BlockEnd {
                    id,
                    end,
                    block: None,
                };
                match (self.blocks.apply(&event), event) {
                    (Ok(Some((id, block))), StreamEvent::BlockEnd { end, .. }) => {
                        Ok(StreamEvent::BlockEnd {
                            id,
                            end,
                            block: Some(block),
                        })
                    }
                    (Ok(_), event) => Ok(event),
                    (Err(error), _) => Err(error),
                }
            }
            Ok(event) => self.blocks.apply(&event).map(|_| event),
            Err(error) => {
                self.finish_reported_call(&error);
                Err(error)
            }
        };
        self.items.push(item);
    }

    /// A malformed-input error item stands in place of the end that raised
    /// it, so the call it reports is finished here as that end finished it.
    fn finish_reported_call(&mut self, error: &ProviderError) {
        let reported = match error {
            ProviderError::MalformedToolInput(reported) => reported,
            ProviderError::Relayed(report) => match &report.detail {
                Some(crate::error::ErrorDetail::MalformedToolInput(reported)) => reported,
                None => return,
            },
            _ => return,
        };
        self.blocks.finish_malformed(reported);
    }

    /// End every text and reasoning block still open, in the order they
    /// opened.
    fn close_open_blocks(&mut self) {
        for (id, end) in self.blocks.unclosed() {
            if self.auto_text.as_ref() == Some(&id) {
                self.auto_text = None;
            }
            if self.auto_reasoning.as_ref() == Some(&id) {
                self.auto_reasoning = None;
            }
            self.push_raw(Ok(StreamEvent::BlockEnd {
                id,
                end,
                block: None,
            }));
        }
    }

    /// Push an in-band error item.
    pub fn error(&mut self, error: ProviderError) {
        self.finish_reported_call(&error);
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

/// Canonicalizing is idempotent, and a relay canonicalizes what it carries.
#[cfg(test)]
mod property_tests;
