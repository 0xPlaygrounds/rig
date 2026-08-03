//! Transactional ownership for one provider model-call attempt.
//!
//! Drivers move a request patch into this record before preparation. The patch
//! is consumed only by a successful commit transition. The driver owns this
//! record while provider I/O is in flight, so dropping an async operation keeps
//! the exact answered attempt available for a later reissue. Explicit failures
//! either retain it in the retryable phase or call [`ModelCallAttempt::abandon`]
//! to restore its patch and return [`AgentRun`] to the pre-call state.

use rig_core::completion::{CompletionError, Message};

use super::config::AgentConfig;
use super::hook::RequestPatch;
use super::prepare::{PreparedRequest, ToolCatalog, prepare_request_with_inherited_output};
use super::run::{
    AcceptedModelTurn, AgentRun, ModelAttemptContext, ModelAttemptId, ModelTurn, ModelTurnOutcome,
    StreamedTurn, ToolOutputContract,
};
use crate::completion::PromptError;

/// Provisional state shared by the blocking and streaming model-call drivers.
///
/// Provider transports and streamed assemblers remain phase-specific driver
/// data. This record owns the inputs and mutations that must commit or roll
/// back together, including across cancellation of the future currently
/// borrowing the driver.
pub(crate) struct ModelCallAttempt {
    context: ModelAttemptContext,
    history: Vec<Message>,
    patch: RequestPatch,
    turn: usize,
    phase: ModelCallAttemptPhase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelCallAttemptPhase {
    Preparing,
    InFlight,
    Retryable,
}

impl ModelCallAttempt {
    /// Move the next-turn patch into a new attempt.
    pub(crate) fn begin(
        prompt: Message,
        history: Vec<Message>,
        turn: usize,
        attempt_id: ModelAttemptId,
        inherited_output_contract: Option<ToolOutputContract>,
        next_patch: &mut Option<RequestPatch>,
    ) -> Self {
        Self {
            context: ModelAttemptContext {
                attempt_id,
                prompt,
                output_contract: inherited_output_contract,
            },
            history,
            patch: next_patch.take().unwrap_or_default(),
            turn,
            phase: ModelCallAttemptPhase::Preparing,
        }
    }

    /// Fold a genuinely new host patch into the retained retry input.
    pub(crate) fn merge_patch(&mut self, patch: RequestPatch) {
        self.patch = std::mem::take(&mut self.patch).merge(patch);
    }

    /// Mark the provider operation as cancellation-sensitive and in flight.
    pub(crate) fn mark_in_flight(&mut self) {
        self.phase = ModelCallAttemptPhase::InFlight;
    }

    /// Roll a preparing or in-flight call back once and retain this exact
    /// attempt for a later reissue.
    pub(crate) fn make_retryable(&mut self, run: &mut AgentRun) {
        if self.phase != ModelCallAttemptPhase::Retryable {
            run.abandon_pending_model_call();
            self.phase = ModelCallAttemptPhase::Retryable;
        }
    }

    /// Re-enter preparation after AgentRun reissued the logical model call.
    pub(crate) fn reissue(&mut self, turn: usize, attempt_id: ModelAttemptId) {
        self.turn = turn;
        self.context.attempt_id = attempt_id;
        self.phase = ModelCallAttemptPhase::Preparing;
    }

    pub(crate) fn attempt_id(&self) -> &str {
        self.context.attempt_id.as_str()
    }

    pub(crate) fn attempt_identity(&self) -> &ModelAttemptId {
        &self.context.attempt_id
    }

    /// Prepare the provider request without mutating durable run state.
    pub(crate) fn prepare(
        &mut self,
        config: &AgentConfig,
        catalog: &ToolCatalog,
        composes_native_output_with_tools: bool,
        committed_output_tool: Option<&str>,
    ) -> Result<PreparedRequest, CompletionError> {
        let prepared = prepare_request_with_inherited_output(
            config,
            catalog,
            composes_native_output_with_tools,
            self.context.prompt.clone(),
            &self.history,
            committed_output_tool,
            self.context.output_contract.as_ref(),
            Some(&self.context.attempt_id),
            Some(&self.patch),
        )?;
        self.context = prepared.model_attempt.attempt_context.clone();
        Ok(prepared)
    }

    /// Atomically feed a completed unary turn and promote provisional
    /// attempt metadata. A rejected turn is abandoned through the same
    /// rollback path as transport and preparation failures.
    pub(crate) fn commit_unary(
        self,
        run: &mut AgentRun,
        next_patch: &mut Option<RequestPatch>,
        turn: ModelTurn,
    ) -> Result<ModelTurnOutcome, PromptError> {
        match run.model_response(turn.with_attempt_context(self.context.clone())) {
            Ok(outcome) => Ok(outcome),
            Err(error) => {
                self.abandon(run, next_patch);
                Err(error)
            }
        }
    }

    /// Atomically feed a successfully exhausted streamed turn and promote
    /// provisional attempt metadata.
    pub(crate) fn commit_streamed(
        self,
        run: &mut AgentRun,
        next_patch: &mut Option<RequestPatch>,
        turn: StreamedTurn,
    ) -> Result<AcceptedModelTurn, PromptError> {
        let mut turn = turn;
        turn.attempt_context = Some(self.context.clone());
        match run.streamed_turn(turn) {
            Ok(accepted) => Ok(accepted),
            Err(error) => {
                self.abandon(run, next_patch);
                Err(error)
            }
        }
    }

    /// Restore the retry patch and roll back the pending logical turn. The
    /// consumed model-call attempt remains charged to the run budget.
    pub(crate) fn abandon(self, run: &mut AgentRun, next_patch: &mut Option<RequestPatch>) {
        if !self.patch.is_empty() {
            *next_patch = Some(match next_patch.take() {
                Some(later) => self.patch.merge(later),
                None => self.patch,
            });
        }
        run.abandon_pending_model_call();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reissued_provider_operation_gets_a_fresh_attempt_identity() {
        let mut next_patch = None;
        let mut attempt = ModelCallAttempt::begin(
            Message::user("prompt"),
            Vec::new(),
            1,
            ModelAttemptId::new(),
            None,
            &mut next_patch,
        );
        let first = attempt.attempt_id().to_owned();

        attempt.reissue(1, ModelAttemptId::new());

        assert_ne!(attempt.attempt_id(), first);
    }
}
