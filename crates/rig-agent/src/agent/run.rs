//! The run protocol at its 0.42 path: [`crate::run`] re-exported.
//!
//! `AgentRun`, its step/turn types, the run's response and error types, the
//! invalid-call decision data and the streamed-turn assembler live at
//! [`crate::run`], as do the request-assembly vocabulary (`spec`, `prepare`,
//! `output`, `patch`) and the loop-side transcript helpers.
//! Every path under `rig_agent::agent::run` resolves as before.

pub use crate::run::*;

/// Streamed-turn accumulation, re-exported from [`crate::run::streamed`].
pub mod streamed {
    pub use crate::run::streamed::*;
}

/// Structured-output mode, re-exported from [`crate::run::output`].
pub mod output_mode {
    pub use crate::run::output::*;
}
