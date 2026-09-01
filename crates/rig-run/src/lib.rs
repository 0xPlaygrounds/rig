//! The sans-IO agent-run protocol for Rig.
//!
//! [`AgentRun`] owns every *decision* the agent loop makes and performs no IO;
//! a driver steps it with [`AgentRun::next_step`] and feeds results back. This
//! crate depends on `rig-core` only — no async runtime, no hooks, no tool
//! registry — so the same state machine can be driven by a futures loop
//! (`rig-agent`) or by ECS systems. See [`run`] for the protocol, [`prepare`]
//! for the pure `(spec, tools, patch) → request` step, [`policy`] for
//! decisions-as-data, [`response`] for outputs, [`transcript`] for how
//! messages are threaded, and [`error`] for [`PromptError`].

#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]
#![forbid(unsafe_code)]

pub mod run;

/// Errors of the run protocol (rig-core).
pub mod error {
    pub use rig_core::completion::response::PromptError;
}
/// Structured-output mode (rig-core).
pub mod output_mode {
    pub use rig_core::completion::output::*;
}
/// Decisions as data (rig-core).
pub mod policy {
    pub use rig_core::completion::policy::*;
}
/// Pure request preparation (rig-core).
pub mod prepare {
    pub use rig_core::completion::prepare::*;
}
/// Run outputs (rig-core).
pub mod response {
    pub use rig_core::completion::response::*;
}
/// Protocol-facing run configuration (rig-core).
pub mod spec {
    pub use rig_core::completion::spec::*;
}
/// Streamed-turn assembly (rig-core).
pub mod streamed {
    pub use rig_core::streaming::assemble::*;
}
/// Canonical-transcript helpers (rig-core).
pub mod transcript {
    pub use rig_core::transcript::*;
}

pub use error::PromptError;
pub use output_mode::OutputMode;
pub use policy::{InvalidToolCallAction, InvalidToolCallContext, RequestPatch, RetryRequest};
pub use prepare::{PrepareError, PreparedRequest, prepare_request};
pub use response::{CompletionCall, PromptResponse};
pub use rig_core::id::RunId;
pub use run::{
    AgentRun, AgentRunStep, ModelTurn, ModelTurnOutcome, PendingToolCall, RunEntry, TurnTools,
};
pub use spec::RunSpec;
pub use streamed::{
    PartialStreamedTurn, StreamedInvalidToolCall, StreamedResolution, StreamedTurn,
    StreamedTurnAssembler, StreamedTurnEvent,
};
pub use transcript::{TranscriptError, validate_canonical};

// Compile-time contract: protocol state is plain, owned data a host can keep in
// shared state (worker pools, ECS components) on native targets.
#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<AgentRun>();
    assert_send_sync_static::<AgentRunStep>();
    assert_send_sync_static::<PendingToolCall>();
    assert_send_sync_static::<ModelTurn>();
    assert_send_sync_static::<StreamedTurn>();
    assert_send_sync_static::<StreamedTurnAssembler>();
    assert_send_sync_static::<PromptResponse>();
    assert_send_sync_static::<CompletionCall>();
    assert_send_sync_static::<PromptError>();
    assert_send_sync_static::<RunSpec>();
    assert_send_sync_static::<TurnTools>();
    assert_send_sync_static::<RunEntry>();
    assert_send_sync_static::<RequestPatch>();
    assert_send_sync_static::<PreparedRequest>();
};
