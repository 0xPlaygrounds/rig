//! What the agent records about itself, each part a plugin of its own:
//! - [`UsagePlugin`](usage::UsagePlugin): what model calls used and cost,
//!   in `/usage` and as the session's `spent` label in `/resume`;
//! - [`EffectLogPlugin`](effect_log::EffectLogPlugin): every settled
//!   effect in the session's `effects.jsonl`;
//! - [`DiagnosticsPlugin`](diagnostics::DiagnosticsPlugin): the process's
//!   warnings and errors as data.

pub mod diagnostics;
pub mod effect_log;
mod resume;
pub mod usage;
