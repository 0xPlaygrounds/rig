//! Experimental ECS-native runtime for Rig.
//!
//! Authoritative run progression lives in Bevy components and ordered systems.
//! Async executors exchange owned effect values with the world; they never
//! borrow ECS state across an await point.

#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::panic_in_result_fn,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]

mod capability;
mod events;
mod identity;
mod local;
mod policy;
mod runtime;
mod schedule;
mod snapshot;
mod state;
mod store;
mod vector;

pub use capability::{
    CapabilityGrant, PortableTool, ToolCapability, ToolEffectRequest, ToolImplementation,
    ToolIngress, ToolRebinding, ToolSnapshot, ToolSnapshotEntry,
};
pub use events::{ProviderFinalEnvelope, RuntimeEvent};
pub use identity::{
    AgentId, CapabilityId, CorrelationId, GrantId, OperationId, RunId, StoreId, TenantId, WorldId,
};
pub use local::{
    LocalBlockingResult, LocalRunError, LocalRuntime, LocalStreamingResult, LocalToolOutputResult,
};
pub use policy::{
    ConcurrencyLimit, InvalidToolPolicy, InvalidToolResolution, OutputMode, RecoveryPolicy,
    select_output_mode, synthetic_output_tool_name,
};
pub use runtime::{
    CompletionIngress, EffectRequest, HostedEffect, HostedHandle, HostedPoll, RunExplanation,
    Runtime, RuntimeError, RuntimeHandle, StreamingIngress,
};
pub use schedule::{RuntimeSchedule, RuntimeSet};
pub use snapshot::{PersistedMemoryBinding, PersistedToolBinding, RunSnapshot, SnapshotError};
pub use state::{
    Accounting, CallBudget, Generation, Lifecycle, PendingRequest, RunIdentity, RunStatus,
    TerminalReason, Transcript,
};
pub use store::{
    MemoryAppendEffect, MemoryAppendIngress, MemoryEffectError, MemoryImplementation,
    MemoryLoadEffect, MemoryLoadIngress, MemoryRebinding,
};
pub use vector::{VectorSearchEffect, VectorSearchIngress};

/// Common imports for the experimental ECS runtime.
pub mod prelude {
    pub use crate::{
        AgentId, CompletionIngress, EffectRequest, OperationId, RunId, RunSnapshot, RunStatus,
        Runtime, RuntimeError, RuntimeHandle, TenantId, TerminalReason, WorldId,
    };
    pub use rig_core::completion::{AssistantContent, CompletionRequest, Message, Usage};
}
