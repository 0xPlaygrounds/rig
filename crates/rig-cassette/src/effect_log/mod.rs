//! Durable effect logs, checkpoints, recorders, and replay handlers.
//!
//! Replayers validate recorded handler identities and publish saved tool outputs
//! before resolving outcomes. Callers must reapply executable middleware and
//! exclude secrets and live capabilities from durable tool outputs. Exact
//! delivery replay requires runtime delivery metadata and retained stream items.
//!
//! ```
//! use rig_cassette::effect_log::EffectLog;
//!
//! let log = EffectLog::default();
//! let (checkpoint, tail) = log.checkpoint(0, ());
//! let resumed = EffectLog::from_checkpoint(&checkpoint, tail)?;
//! assert!(resumed.is_empty());
//! # Ok::<(), rig_core::error::ErrorReport>(())
//! ```

mod log;
mod recorder;
mod replay;

pub use log::{
    CHECKPOINT_FORMAT, Checkpoint, EffectLog, LogHeader, ProgramIdentity, RecordedStreamError,
    stable_hash,
};
pub use recorder::EffectLogRecorder;
pub use replay::{EffectLogReplayer, ReplayRefusals, RequestCheck};

#[cfg(test)]
mod tests;
