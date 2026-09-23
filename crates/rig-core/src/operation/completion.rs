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
    Absorbed, BlockAccumulator, BlockClose, BlockId, BlockKind, Delta, FoldStep, MintKind,
    StreamEvent, StreamFinal, SyntheticIds, ToolCallEnd, UnknownPayload,
};
use crate::telemetry::{GenAiOperation, SpanBuilder, SpanCombinator};
use crate::wire::{Fold, Operation, Reply, Sink};

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

    fn telemetry(streaming: bool) -> Self::Telemetry {
        if streaming {
            GenAiOperation::ChatStreaming
        } else {
            GenAiOperation::Chat
        }
    }

    /// Forwards unmodeled payloads without adding them to aggregated content.
    fn unknown(payload: crate::streaming::UnknownPayload) -> Option<Self::Event> {
        Some(StreamEvent::Unknown(payload))
    }

    /// Reasoning another wire issued is omitted; see
    /// [`crate::message::retain_replayable_reasoning`].
    fn scope_to_wire(request: &mut Self::Request, issuers: &[&str]) {
        crate::message::retain_replayable_reasoning(&mut request.chat_history, issuers);
    }

    fn request_model(request: &Self::Request) -> Option<&str> {
        request.model.as_deref()
    }

    fn stamp_request_id(event: &mut Self::Event, request_id: &Option<String>) {
        // The terminal's own id wins: it saw the reply that carried it.
        if let StreamEvent::Final(terminal) = event
            && terminal.provider_request_id.is_none()
        {
            terminal.provider_request_id = request_id.clone();
        }
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

    fn check_laws(&self, laws: &mut Self::Laws) {
        #[cfg(any(test, debug_assertions))]
        laws.check_batch(self);
        #[cfg(not(any(test, debug_assertions)))]
        let _ = laws;
    }
}

/// The fold from a completion reply's events to its response.
///
/// The same step [`StreamingCompletionResponse`](crate::streaming::StreamingCompletionResponse)
/// runs while it yields events, so a unary reply and a streamed one agree by
/// construction.
#[derive(Default)]
pub struct CompletionFold {
    accumulator: BlockAccumulator,
    terminal: Option<StreamFinal>,
    message_id: Option<String>,
    /// Only written by the fold step; the response's provider is the wire's.
    provider: String,
}

impl Fold<Completion> for CompletionFold {
    fn absorb(&mut self, event: StreamEvent) -> Result<(), ProviderError> {
        let step = FoldStep {
            accumulator: &mut self.accumulator,
            response: &mut self.terminal,
            message_id: &mut self.message_id,
            provider: &mut self.provider,
            provider_from_terminal: false,
        };
        match crate::streaming::absorb(step, event) {
            Absorbed::Yield(_) | Absorbed::Skip => Ok(()),
            // A buffered reply has no stream to carry an in-band defect, so
            // a block the wire promised and then malformed fails the call.
            Absorbed::Failed(report) => Err(ProviderError::Response(report.message)),
        }
    }

    fn finish(self, reply: Reply) -> Result<CompletionResponse, ProviderError> {
        // The buffered reply's document is the response's `raw`, not the
        // terminal record's: the wire decoded the whole body at once.
        let issuer = self
            .terminal
            .as_ref()
            .map_or(reply.provider.clone(), |terminal| {
                terminal.issuer().to_owned()
            });
        let response = crate::streaming::fold_finish(
            self.accumulator,
            self.terminal.as_ref(),
            self.message_id,
            reply.provider,
            &issuer,
            reply.raw,
        );
        // The terminal's own id wins; the reply headers only fill a gap.
        if response.provider_request_id.is_none() {
            Ok(response.with_optional_provider_request_id(reply.provider_request_id))
        } else {
            Ok(response)
        }
    }
}

/// Buffers completion events and in-band errors with block bookkeeping.
/// Helpers open unseen tool and reasoning keys before deltas. Bare text uses
/// an active key, minting a new one after non-text block events other than
/// message starts. Frame-classification errors are handled by the driver.
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
        self.items.push(item);
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

#[cfg(test)]
mod tests;
