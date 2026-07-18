//! Observable runtime events. Provisional observations are distinct from commits.

use crate::{CorrelationId, OperationId, RunId, TerminalReason};
use bevy_ecs::prelude::Component;
use rig_core::completion::AssistantContent;
use rig_core::tool::ToolResult;

/// Non-persisted hosted/erased provider-final diagnostics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderFinalEnvelope {
    pub provider: String,
    pub type_name: String,
    pub diagnostic_json: serde_json::Value,
}

/// Ordered observations emitted by one run.
#[derive(Clone, Debug)]
pub enum RuntimeEvent {
    ProvisionalDelta {
        run_id: RunId,
        operation_id: OperationId,
        correlation_id: CorrelationId,
        sequence: u64,
        content: AssistantContent,
    },
    AcceptedFinal {
        run_id: RunId,
        content: Vec<AssistantContent>,
    },
    Rollback {
        run_id: RunId,
        reason: String,
    },
    ProviderFinal {
        run_id: RunId,
        final_response: ProviderFinalEnvelope,
    },
    ProviderFailure {
        run_id: RunId,
        message: String,
    },
    RejectedEffect {
        run_id: RunId,
        reason: String,
    },
    ToolCommitted {
        run_id: RunId,
        call_index: usize,
        name: String,
        result: ToolResult,
    },
    Terminal {
        run_id: RunId,
        reason: TerminalReason,
    },
}

/// Retained ordered event history for subscribers and diagnostics.
#[derive(Component, Clone, Debug, Default)]
pub(crate) struct EventLog(pub Vec<RuntimeEvent>);
