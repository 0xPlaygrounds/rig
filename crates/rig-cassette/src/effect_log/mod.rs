//! Durable effect logs, checkpoints, recorders, replay handlers, and
//! [`RecordedAnswers`], a log as answers a host looks up.
//!
//! Replayers validate recorded handler identities and publish saved tool outputs
//! before resolving outcomes. Callers must reapply executable middleware and
//! exclude secrets and live capabilities from durable tool outputs.
//!
//! A log alone does not restore an application. A host that runs effects
//! on its own runtime saves its own state in a [`Checkpoint`] with its place
//! in the log; to resume, it restores that state, checks that the log's tail
//! follows the checkpoint, and replays the tail, by id when it issues the
//! recorded ids again:
//!
//! ```
//! use rig_cassette::effect_log::{Checkpoint, EffectLog, EffectLogReplayer};
//!
//! #[derive(serde::Serialize, serde::Deserialize)]
//! struct Host {
//!     turn: u32,
//! }
//!
//! let log = EffectLog::default();
//! let (checkpoint, tail) = log.checkpoint(0, Host { turn: 3 });
//! let saved = serde_json::to_string(&checkpoint)?;
//!
//! let checkpoint: Checkpoint<Host> = serde_json::from_str(&saved)?;
//! let tail = EffectLog::from_checkpoint(&checkpoint, tail)?;
//! let replayers = EffectLogReplayer::for_log_by_id(&tail)?;
//! assert_eq!(checkpoint.state.turn, 3);
//! assert!(replayers.is_empty());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod answers;
mod filter;
#[cfg(feature = "jsonl")]
pub mod jsonl;
mod log;
mod recorder;
mod replay;

pub use answers::{Divergence, RecordedAnswers};
pub use filter::EffectFilter;
pub use log::{
    CHECKPOINT_FORMAT, Checkpoint, EffectLog, LogHeader, ProgramIdentity, RecordedStreamError,
    canonical_local_ids, stable_hash,
};
pub use recorder::EffectLogRecorder;
pub use replay::{EffectLogReplayer, ReplayRefusals, RequestCheck};

#[cfg(test)]
mod tests;
