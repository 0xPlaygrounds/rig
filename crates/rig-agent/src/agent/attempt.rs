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
use super::prepare::{PreparedRequest, ToolCatalog, prepare_request};
use super::run::{AcceptedModelTurn, AgentRun, ModelTurn, ModelTurnOutcome, StreamedTurn};
use crate::completion::PromptError;

/// Provisional state shared by the blocking and streaming model-call drivers.
///
/// Provider transports and streamed assemblers remain phase-specific driver
/// data. This record owns the inputs and mutations that must commit or roll
/// back together, including across cancellation of the future currently
/// borrowing the driver.
pub(crate) struct ModelCallAttempt {
    prompt: Message,
    history: Vec<Message>,
    patch: RequestPatch,
    turn: usize,
    output_tool_name: Option<String>,
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
        next_patch: &mut Option<RequestPatch>,
    ) -> Self {
        Self {
            prompt,
            history,
            patch: next_patch.take().unwrap_or_default(),
            turn,
            output_tool_name: None,
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
    pub(crate) fn reissue(&mut self, turn: usize) {
        self.turn = turn;
        self.phase = ModelCallAttemptPhase::Preparing;
    }

    /// Prepare the provider request without mutating durable run state.
    pub(crate) fn prepare(
        &mut self,
        config: &AgentConfig,
        catalog: &ToolCatalog,
        composes_native_output_with_tools: bool,
        committed_output_tool: Option<&str>,
    ) -> Result<PreparedRequest, CompletionError> {
        let prepared = prepare_request(
            config,
            catalog,
            composes_native_output_with_tools,
            self.prompt.clone(),
            &self.history,
            committed_output_tool,
            Some(&self.patch),
        )?;
        self.output_tool_name = prepared.output_tool_name.clone();
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
        match run.model_response(turn) {
            Ok(outcome) => {
                run.set_output_tool_name(self.output_tool_name);
                Ok(outcome)
            }
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
        match run.streamed_turn(turn) {
            Ok(accepted) => {
                run.set_output_tool_name(self.output_tool_name);
                Ok(accepted)
            }
            Err(error) => {
                self.abandon(run, next_patch);
                Err(error)
            }
        }
    }

    /// Promote provisional metadata after invalid-call recovery deliberately
    /// commits the provider attempt's usage while abandoning its turn.
    pub(crate) fn commit_recovered(self, run: &mut AgentRun) {
        run.set_output_tool_name(self.output_tool_name);
    }

    /// Restore the retry patch and refund the pending model call.
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
