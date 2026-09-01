//! The run protocol at its 0.42 path: [`crate::run`] re-exported.
//!
//! `AgentRun`, its step/turn types, the run's response and error types, the
//! invalid-call decision data and the streamed-turn assembler live at
//! [`crate::run`]; the request vocabulary they are built from is rig-core's
//! (`rig_core::{transcript, completion::{output, patch, prepare, spec}}`).
//! Every path under `rig_agent::agent::run` resolves as before.

pub use crate::run::*;

/// Streamed-turn accumulation, re-exported from [`crate::run::streamed`].
pub mod streamed {
    pub use crate::run::streamed::*;
}

/// Structured-output mode, re-exported from `rig_core::completion::output`.
pub mod output_mode {
    pub use rig_core::completion::output::*;
}
