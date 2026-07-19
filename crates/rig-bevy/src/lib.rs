//! Experimental native-only Bevy ECS runtime for Rig.
//!
//! The ECS world is authoritative for agent topology and run progression.
//! Provider, tool, and memory futures receive owned effect values and return
//! through validated ingress carrying runtime, run, operation, generation,
//! tenant, capability, grant, revision, and correlation identity.
//!
//! The classic runtime remains the supported default. This crate is opt-in,
//! native-only, and intentionally does not depend on `rig-agent`.

#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(target_family = "wasm")]
compile_error!("rig-bevy is an experimental native-only runtime; use rig-agent on WebAssembly");

mod components;
mod config;
mod debug;
mod effects;
mod error;
mod identity;
mod persistence;
mod policy;
mod runtime;
mod schedule;

/// Common imports for constructing and operating the ECS runtime.
pub mod prelude;

pub use components::{
    AgentNode, CanonicalTranscript, CapabilityKind, CapabilityNode, GrantNode, ModelCallRecord,
    RunAccounting, RunEvent, RunPhase, TerminalObservation, TerminalReason, TerminalState,
};
pub use config::{AgentSpec, ResponseRetryPolicy, RuntimeConfig, StreamingMode};
pub use debug::{ContentVisibility, RunExplanation};
pub use effects::{
    EffectCompletion, EffectHeader, EffectIngress, EffectRejection, EffectRejectionReason,
    HostedProviderDiagnostic, MemoryEffectError, MemoryEffectOutput, ModelEffectError,
    ModelEffectOutput, ProvisionalDelta, ToolEffectOutput,
};
pub use error::{RuntimeError, SnapshotError};
pub use identity::{
    AgentId, BindingIdentity, CapabilityId, CorrelationId, Generation, GrantId, MemoryId, ModelId,
    OperationId, RunId, RuntimeId, TenantId,
};
pub use persistence::{
    ProtectedSnapshot, RebindRegistry, SnapshotContentPolicy, SnapshotProtector,
};
pub use policy::{InvalidToolPolicy, OutputMode, StructuredOutputPolicy};
pub use runtime::{
    BevyAgentBuilder, BevyAgentDefinition, BevyClientExt, BevyModelExt, HostedRuntime,
    LocalRunResult, LocalRuntime, LocalStreamingRun, RunHandle, RunStepStatus, StreamingRunEvent,
    StreamingRunResult, ToolGrant,
};
