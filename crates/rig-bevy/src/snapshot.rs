use crate::{AgentId, CapabilityId, RunId, RunStatus, StoreId, TenantId, WorldId};
use rig_core::completion::{AssistantContent, CompletionRequest, Message, ToolDefinition, Usage};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunSnapshot {
    pub schema_version: u32,
    pub world_id: WorldId,
    pub tenant_id: TenantId,
    pub agent_id: AgentId,
    pub run_id: RunId,
    pub generation: u64,
    pub max_calls: usize,
    pub completed_calls: usize,
    pub status: RunStatus,
    /// Owned request required to resume a run at a stable `Ready` boundary.
    pub pending_request: Option<CompletionRequest>,
    pub history: Vec<Message>,
    pub output: Vec<AssistantContent>,
    pub usage: Usage,
    pub rejected_effects: usize,
    pub required_tools: Vec<PersistedToolBinding>,
    pub required_memory: Option<PersistedMemoryBinding>,
}

/// Persisted capability metadata; executable implementations are deliberately excluded.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistedToolBinding {
    pub capability_id: CapabilityId,
    pub tenant_id: TenantId,
    pub revision: u64,
    pub definition: ToolDefinition,
    pub retired: bool,
}

/// Persisted memory identity. Backend objects and conversation contents are
/// deliberately excluded and must be explicitly rebound.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistedMemoryBinding {
    pub store_id: StoreId,
    pub conversation_id: String,
    pub loaded: bool,
    pub persist_from: usize,
    pub appended_generation: Option<u64>,
}

#[derive(Debug, Error)]
pub enum SnapshotError {
    #[error("unsupported snapshot schema version {0}")]
    UnsupportedVersion(u32),
    #[error("snapshot contains a non-terminal in-flight operation")]
    InFlightOperation,
    #[error("snapshot integrity check failed: {0}")]
    InvalidIntegrity(String),
    #[error("missing implementation rebinding for capability {0:?} revision {1}")]
    MissingImplementation(CapabilityId, u64),
    #[error("implementation rebinding mismatches capability {0:?} revision {1}")]
    MismatchedImplementation(CapabilityId, u64),
    #[error("missing implementation rebinding for memory store {0:?}")]
    MissingMemoryImplementation(StoreId),
}
