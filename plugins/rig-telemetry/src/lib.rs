//! What the agent records about itself, each part a plugin of its own:
//! what model calls cost and `/usage` ([`UsagePlugin`]), every model and
//! tool call in the session's `effects.jsonl` ([`EffectLogPlugin`]), and the
//! process's warnings and errors as data ([`DiagnosticsPlugin`]).

pub mod diagnostics;
pub mod effect_log;
pub mod usage;

pub use diagnostics::{Diagnostics, DiagnosticsPlugin};
pub use effect_log::EffectLogPlugin;
pub use usage::{Spending, TurnSpending, UsagePlugin};
