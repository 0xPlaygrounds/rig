use crate::completion::{
    CompletionError, CompletionRequest, CompletionResponse, ProviderCapabilities,
};
use crate::streaming::{BlockAccumulator, StreamEvent, StreamFinal};

use super::{Fold, Operation};

/// The completion operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Completion;

impl Operation for Completion {
    type Request = CompletionRequest;
    type Event = StreamEvent;
    type Response = CompletionResponse;
    type Error = CompletionError;
    type Capabilities = ProviderCapabilities;
    type Fold = CompletionFold;
}

/// Accumulates stream events into a completed completion response.
#[derive(Default)]
pub struct CompletionFold {
    accumulator: BlockAccumulator,
    terminal: Option<StreamFinal>,
    message_id: Option<String>,
}

impl Fold<StreamEvent> for CompletionFold {
    type Response = CompletionResponse;
    type Error = CompletionError;

    fn fold(&mut self, event: StreamEvent) {
        match &event {
            StreamEvent::BlockStart {
                id,
                kind: crate::streaming::BlockKind::Message,
            } => {
                if let Some(msg_id) = id.wire_str() {
                    self.message_id = Some(msg_id.to_owned());
                }
            }
            StreamEvent::Final(terminal) => {
                self.terminal = Some(terminal.clone());
            }
            _ => {}
        }
        let _ = self.accumulator.apply(&event);
    }

    fn finish(mut self) -> Result<CompletionResponse, CompletionError> {
        let choice = self.accumulator.finish();
        let terminal = self.terminal.as_ref();
        let mut resp = CompletionResponse::new(
            choice,
            terminal.map(|r| r.usage).unwrap_or_default(),
            String::new(),
        )
        .with_optional_message_id(
            self.message_id
                .or_else(|| terminal.and_then(|r| r.message_id.clone())),
        )
        .with_optional_response_id(terminal.and_then(|r| r.response_id.clone()))
        .with_optional_provider_request_id(terminal.and_then(|r| r.provider_request_id.clone()))
        .with_optional_finish_reason(terminal.and_then(|r| r.finish_reason.clone()))
        .with_optional_model(terminal.and_then(|r| r.model.clone()));

        if let Some(terminal) = terminal {
            resp.raw = terminal.raw.clone();
        }

        Ok(resp)
    }
}
