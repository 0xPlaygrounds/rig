//! What the agent records about itself, each part a plugin of its own:
//! what model calls cost, in `/usage` and as the session's `spent` label
//! in `/resume` ([`UsagePlugin`]), every model and
//! tool call in the session's `effects.jsonl` ([`EffectLogPlugin`]), and the
//! process's warnings and errors as data ([`DiagnosticsPlugin`]).

pub mod diagnostics;
pub mod effect_log;
mod resume;
pub mod usage;

pub use diagnostics::{Diagnostics, DiagnosticsPlugin};
pub use effect_log::{EffectFilter, EffectLog, EffectLogPlugin};
pub use usage::{CallUsage, SessionSpending, Spending, TurnSpending, UsagePlugin, UsageRow};
