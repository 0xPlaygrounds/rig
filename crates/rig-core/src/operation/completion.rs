//! The completion operation: the one operation whose replies stream.

use crate::completion::{CompletionError, CompletionRequest, CompletionResponse};
use crate::providers::internal::adapter::AdapterOutput;
use crate::streaming::{Absorbed, BlockAccumulator, FoldStep, StreamEvent, StreamFinal};
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder, SpanCombinator};
use crate::wire::{Fold, Operation, Reply, Sink};

/// One decoded step of a completion reply.
pub type CompletionEvent = StreamEvent;

/// Generating an assistant turn, unary or streamed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Completion;

/// Debug-mode sequence laws over what a decoder actually emitted.
type Laws = crate::providers::internal::sequence_law::SequenceLaws;

impl Operation for Completion {
    type Request = CompletionRequest;
    type Event = StreamEvent;
    type Response = CompletionResponse;
    type Error = CompletionError;
    type Capabilities = crate::completion::ProviderCapabilities;
    type Output = AdapterOutput;
    type Fold = CompletionFold;
    type Telemetry = CompletionOperation;

    const NAME: &'static str = "completion";

    fn is_terminal(event: &Self::Event) -> bool {
        matches!(event, StreamEvent::Final(_))
    }

    fn telemetry(streaming: bool) -> Self::Telemetry {
        if streaming {
            CompletionOperation::ChatStreaming
        } else {
            CompletionOperation::Chat
        }
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
        CompletionSpanBuilder::new(provider, model.unwrap_or_default(), telemetry)
            .system_instructions(
                request.system_instructions(),
                request.record_telemetry_content,
            )
            .build()
    }

    fn record(span: &tracing::Span, response: &Self::Response) {
        if span.is_disabled() {
            return;
        }
        // Recorded off the normalized response, so no provider has to
        // implement a telemetry projection of its wire type. An
        // Anthropic-style wire puts its message id in `message_id`; an
        // OpenAI-style one puts its `chatcmpl-` id in `response_id`.
        if let Some(id) = response
            .response_id
            .as_deref()
            .or(response.message_id.as_deref())
        {
            span.record("gen_ai.response.id", id);
        }
        if let Some(model) = response.model.as_deref() {
            span.record("gen_ai.response.model", model);
        }
        span.record_token_usage(&response.usage);
    }

    fn record_event(span: &tracing::Span, event: &Self::Event) {
        let StreamEvent::Final(terminal) = event else {
            return;
        };
        if span.is_disabled() {
            return;
        }
        if let Some(id) = terminal
            .response_id
            .as_deref()
            .or(terminal.message_id.as_deref())
        {
            span.record("gen_ai.response.id", id);
        }
        if let Some(model) = terminal.model.as_deref() {
            span.record("gen_ai.response.model", model);
        }
        span.record_token_usage(&terminal.usage);
    }
}

impl Sink<Completion> for AdapterOutput {
    type Laws = Laws;

    fn push(&mut self, item: Result<StreamEvent, CompletionError>) {
        AdapterOutput::push(self, item);
    }

    fn drain(&mut self) -> std::vec::Drain<'_, Result<StreamEvent, CompletionError>> {
        AdapterOutput::drain(self)
    }

    fn items(&self) -> &[Result<StreamEvent, CompletionError>] {
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
    fn absorb(&mut self, event: StreamEvent) -> Result<(), CompletionError> {
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
            Absorbed::Failed(report) => Err(CompletionError::ResponseError(report.message)),
        }
    }

    fn finish(self, reply: Reply) -> Result<CompletionResponse, CompletionError> {
        let response = crate::streaming::fold_finish(
            self.accumulator,
            self.terminal.as_ref(),
            self.message_id,
            reply.provider,
        );
        // The terminal's own id wins; the reply headers only fill a gap.
        let response = if response.provider_request_id.is_none() {
            response.with_optional_provider_request_id(reply.provider_request_id)
        } else {
            response
        };
        Ok(response.with_raw(reply.raw))
    }
}
