//! Redacted ECS explanations derived from authoritative facts.

use crate::{
    AgentId, CanonicalTranscript, RunAccounting, RunEvent, RunHandle, RunId, RunPhase,
    RuntimeError, TerminalReason,
    components::{RunEvents, RunNode, TerminalState},
    runtime::LocalRuntime,
};

/// Content policy for an explicit run explanation request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ContentVisibility {
    /// Redact prompts, model output, tool arguments/results, and memory content.
    #[default]
    Redacted,
    /// Include the canonical committed transcript only.
    Canonical,
}

/// Stable explanation derived from ECS facts without provider finals or implementation handles.
#[derive(Clone, Debug)]
pub struct RunExplanation {
    /// Stable run identity.
    pub run_id: RunId,
    /// Owning agent identity.
    pub agent_id: AgentId,
    /// Current lifecycle phase.
    pub phase: RunPhase,
    /// Terminal reason when present.
    pub terminal_reason: Option<TerminalReason>,
    /// Number of model calls dispatched.
    pub model_calls_dispatched: usize,
    /// Number of completed model-call records.
    pub model_calls_completed: usize,
    /// Number of rejected effect messages.
    pub rejected_effects: usize,
    /// Event category names with content removed.
    pub event_kinds: Vec<&'static str>,
    /// Canonical transcript included only after explicit opt-in.
    pub canonical_transcript: Option<Vec<rig_core::completion::Message>>,
}

impl LocalRuntime {
    /// Explain one retained run, redacting content unless explicitly requested.
    pub fn explain(
        &self,
        handle: RunHandle,
        visibility: ContentVisibility,
    ) -> Result<RunExplanation, RuntimeError> {
        let entity = self.validate_handle(handle)?;
        let run = self
            .world
            .get::<RunNode>(entity)
            .ok_or(RuntimeError::UnknownRun(handle.run_id))?;
        let accounting = self
            .world
            .get::<RunAccounting>(entity)
            .ok_or(RuntimeError::UnknownRun(handle.run_id))?;
        let terminal_reason = self
            .world
            .get::<TerminalState>(entity)
            .map(|terminal| terminal.reason.clone());
        let event_kinds = self
            .world
            .get::<RunEvents>(entity)
            .map(|events| events.events.iter().map(event_kind).collect())
            .unwrap_or_default();
        let canonical_transcript = matches!(visibility, ContentVisibility::Canonical).then(|| {
            self.world
                .get::<CanonicalTranscript>(entity)
                .map(|transcript| transcript.messages.clone())
                .unwrap_or_default()
        });
        Ok(RunExplanation {
            run_id: run.id,
            agent_id: run.agent_id,
            phase: run.phase,
            terminal_reason,
            model_calls_dispatched: accounting.model_calls_dispatched,
            model_calls_completed: accounting.model_calls.len(),
            rejected_effects: accounting.rejected_effects,
            event_kinds,
            canonical_transcript,
        })
    }
}

fn event_kind(event: &RunEvent) -> &'static str {
    match event {
        RunEvent::ModelDispatched(_) => "model_dispatched",
        RunEvent::Provisional { .. } => "provisional",
        RunEvent::ProviderFinal { .. } => "provider_final",
        RunEvent::ToolDispatched { .. } => "tool_dispatched",
        RunEvent::ToolCommitted { .. } => "tool_committed",
        RunEvent::ResponseRetried => "response_retried",
        RunEvent::ToolSuppressed { .. } => "tool_suppressed",
        RunEvent::Terminal(_) => "terminal",
        RunEvent::EffectRejected(_) => "effect_rejected",
    }
}
