//! The sans-IO run protocol, re-exported from [`rig_run`].
//!
//! `AgentRun` and its step/turn types live in the `rig-run` crate so that any
//! driver — this crate's futures loop or an ECS plugin — steps the same state
//! machine. Every path under `rig_agent::agent::run` resolves as before.

pub use rig_run::run::*;

/// Streamed-turn accumulation, re-exported from [`rig_run::streamed`].
pub mod streamed {
    pub use rig_run::streamed::*;
}

/// Structured-output mode, re-exported from [`rig_run::output_mode`].
pub mod output_mode {
    pub use rig_run::output_mode::*;
}
