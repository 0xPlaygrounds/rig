//! Focused authoritative ECS state for the Bevy runtime.

use crate::{
    AgentId, CapabilityId, CorrelationId, InvalidToolResolution, OperationId, RunId, TenantId,
    WorldId,
};
use bevy_ecs::prelude::Component;
use rig_core::completion::{AssistantContent, CompletionRequest, Message, Usage};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Why a run became terminal.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TerminalReason {
    Completed,
    Cancelled { reason: String },
    Failed { message: String },
    BudgetExhausted,
    Livelock,
}

/// Observable lifecycle state. Terminal state remains present until retention cleanup.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RunStatus {
    Ready,
    AwaitingModel,
    AwaitingTools,
    Quiescent,
    Terminal(TerminalReason),
}

/// Stable ownership and identity, deliberately separate from Bevy [`Entity`](bevy_ecs::entity::Entity).
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
pub struct RunIdentity {
    pub world_id: WorldId,
    pub tenant_id: TenantId,
    pub agent_id: AgentId,
    pub run_id: RunId,
}

/// Generation used to invalidate superseded or cancelled effects.
#[derive(Component, Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Generation(pub u64);

/// Total model-call budget and committed call count.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CallBudget {
    pub max_calls: usize,
    pub completed_calls: usize,
}

/// Lifecycle progression owned by ECS state.
#[derive(Component, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Lifecycle(pub RunStatus);

/// Next owned provider request, consumed exactly once when dispatched.
#[derive(Component, Clone, Debug)]
pub struct PendingRequest(pub Option<CompletionRequest>);

/// Request template retained while a tool batch is being resolved.
#[derive(Component, Clone, Debug, Default)]
pub(crate) struct ToolContinuation(pub Option<CompletionRequest>);

/// Last policy result, produced by the ordered policy system and consumed by a driver.
#[derive(Component, Clone, Debug, Default)]
pub(crate) struct InvalidResolution(pub Option<InvalidToolResolution>);

#[derive(Component, Clone, Copy, Debug, Default)]
pub(crate) struct InvalidRetryCount(pub usize);

/// Canonical committed transcript. Provisional streaming data never enters it.
#[derive(Component, Clone, Debug, Default, Serialize, Deserialize)]
pub struct Transcript {
    pub history: Vec<Message>,
    pub final_output: Vec<AssistantContent>,
}

/// Idempotent usage and completion accounting.
#[derive(Component, Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Accounting {
    pub usage: Usage,
    pub rejected_effects: usize,
}

impl Default for Accounting {
    fn default() -> Self {
        Self {
            usage: Usage::new(),
            rejected_effects: 0,
        }
    }
}

/// Correlation identity for the one authoritative in-flight model operation.
#[derive(Component, Clone, Debug)]
pub(crate) struct PendingOperation {
    pub operation_id: OperationId,
    pub correlation_id: CorrelationId,
    pub generation: u64,
    pub request: CompletionRequest,
}

/// Next accepted provisional sequence for deterministic streaming observation.
#[derive(Component, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct StreamCursor(pub u64);

/// Consecutive progress passes that could not advance a nonterminal run.
#[derive(Component, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct StalledPasses(pub u16);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PendingTool {
    pub operation_id: OperationId,
    pub correlation_id: CorrelationId,
    pub generation: u64,
    pub capability_id: CapabilityId,
    pub revision: u64,
}

/// In-flight tool calls and out-of-order completions for one canonical batch.
#[derive(Component, Debug, Default)]
pub(crate) struct ToolBatch {
    pub pending: BTreeMap<usize, PendingTool>,
    pub completed: BTreeMap<usize, (String, rig_core::tool::ToolResult)>,
    pub next_commit: usize,
}

/// Terminal records remain until at least one observer acknowledges them and
/// cleanup is explicitly requested.
#[derive(Component, Clone, Copy, Debug)]
pub(crate) struct Retention {
    pub observations_remaining: usize,
    pub cleanup_requested: bool,
}

#[derive(Component, Clone, Copy, Debug)]
pub(crate) struct PendingVector {
    pub operation_id: OperationId,
    pub correlation_id: CorrelationId,
    pub generation: u64,
    pub store_id: crate::StoreId,
}

impl Default for Retention {
    fn default() -> Self {
        Self {
            observations_remaining: 1,
            cleanup_requested: false,
        }
    }
}
