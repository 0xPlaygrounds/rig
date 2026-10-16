//! Durable effect logs, recorders, replay handlers, and
//! [`RecordedAnswers`], a log as answers a host looks up.
//!
//! Replayers validate recorded handler identities and publish saved tool outputs
//! before resolving outcomes. Callers must reapply executable middleware and
//! exclude secrets and live capabilities from durable tool outputs.
//!
//! A run interrupted after `at` records resumes from its saved state with
//! the log's [`tail`](EffectLog::tail) replayed.
//!
//! ```
//! use rig_cassette::effect_log::{EffectLog, EffectLogReplayer};
//!
//! let log = EffectLog::default();
//! let replayers = EffectLogReplayer::for_log(&log.tail(0))?;
//! assert!(replayers.is_empty());
//! # Ok::<(), rig_core::error::ErrorReport>(())
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
pub use log::{EffectLog, LogHeader, RecordedStreamError, canonical_local_ids, stable_hash};
pub use recorder::EffectLogRecorder;
pub use replay::{EffectLogReplayer, RequestCheck};

#[cfg(test)]
mod tests;
