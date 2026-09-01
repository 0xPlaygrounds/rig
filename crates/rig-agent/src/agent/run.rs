//! The sans-IO run protocol at its 0.42 path: [`crate::run`] re-exported.
//!
//! `AgentRun` and its step/turn types live at [`crate::run`]; the data and
//! policy vocabulary they are built from lives in rig-core
//! (`rig_core::{transcript, completion::{output, policy, prepare, spec,
//! response}, streaming::assemble}`). Every path under `rig_agent::agent::run`
//! resolves as before.

pub use crate::run::*;

/// Streamed-turn accumulation, re-exported from `rig_core::streaming::assemble`.
pub mod streamed {
    pub use rig_core::streaming::assemble::*;
}

/// Structured-output mode, re-exported from `rig_core::completion::output`.
pub mod output_mode {
    pub use rig_core::completion::output::*;
}
