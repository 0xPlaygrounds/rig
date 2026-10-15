//! What the agent records about itself, each part a plugin of its own:
//! - [`UsagePlugin`]: what model calls used and cost, in `/usage` and as the
//!   session's `spent` label in `/resume`;
//! - [`EffectLogPlugin`]: every settled effect in the session's
//!   `effects.jsonl`;
//! - [`DiagnosticsPlugin`]: the process's warnings and errors as data.

pub mod diagnostics;
pub mod effect_log;
mod resume;
pub mod usage;

pub use diagnostics::{Diagnostics, DiagnosticsPlugin};
pub use effect_log::{EffectLog, EffectLogPlugin};
pub use usage::{AgentUsage, CallUsage, SessionUsage, TurnUsage, UsagePlugin, UsageRow};
