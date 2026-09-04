#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]
//! Record and replay for the effect bus.
//!
//! An [`EffectLog`] is a recorded run: a [`LogHeader`] — the format, the
//! run-spec hash an agent stamps, the handlers registered when recording
//! began, the effect signature — and every exchange in dispatch order
//! (`rig_core::effect::EffectRecord`). The [`EffectLogRecorder`] is the
//! [`Recorder`](rig_core::serve::Recorder) a driver such as rig-agent's `BusDriver`
//! writes to (its `record_to`); the
//! [`EffectLogReplayer`] is a handler that answers the same dispatches from
//! the record instead of a provider, one per key (a runtime registers
//! them: `rig_agent::bus::replay::register_all`, rig-ecs's `Replay`). A
//! log with another
//! [`EFFECT_LOG_FORMAT`] does not load: there is no tolerant decoder.
//!
//! The vocabulary is rig-core's, the runtimes are rig-agent's bus and rig-ecs; this crate is
//! the persistence story over both, and what a host that saves and restores
//! in-flight effects (a scene, a durable run) depends on.

mod log;
mod recorder;
mod replay;

pub use log::{Checkpoint, EFFECT_LOG_FORMAT, EffectLog, LogHeader, ProgramIdentity, stable_hash};
pub use recorder::EffectLogRecorder;
pub use replay::{EffectLogReplayer, RequestCheck};

#[cfg(test)]
mod tests;
