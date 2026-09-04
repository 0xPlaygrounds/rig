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
//! An [`EffectLog`] is a recorded run: a [`LogHeader`] — the
//! run-spec hash an agent stamps, the handlers registered when recording
//! began, the effect signature — and every exchange in dispatch order
//! (`rig_core::effect::EffectRecord`). The [`EffectLogRecorder`] is the
//! [`Recorder`](rig_core::serve::Recorder) a driver such as rig-agent's `BusDriver`
//! writes to (its `record_to`); the
//! [`EffectLogReplayer`] is a handler that answers the same dispatches from
//! the record instead of a provider, one per key (a runtime registers
//! them: `rig_agent::bus::replay::register_all`, rig-ecs's `Replay`). A
//! log is checked by its data, not a global format number in its header.
//!
//! Records hold explicitly published tool result values separately from
//! the ordinary outcome, including error outcomes. Replayers publish these
//! values before resolving the outcome. The recorded map excludes inbound
//! tool context and live scopes; handlers must publish before resolving and
//! must not put secrets or live capabilities into durable result values.
//! The `tool_output` field is required: `null` explicitly records no
//! publication, an object records published values (possibly empty). Omission
//! cannot establish whether values were lost and is rejected when decoding.
//! Legacy header `format` fields are ignored and are never written. The
//! separate [`Checkpoint`] envelope still has its own [`CHECKPOINT_FORMAT`].
//!
//! The vocabulary is rig-core's, the runtimes are rig-agent's bus and rig-ecs; this crate is
//! the persistence story over both, and what a host that saves and restores
//! in-flight effects (a scene, a durable run) depends on.

mod log;
mod recorder;
mod replay;

pub use log::{CHECKPOINT_FORMAT, Checkpoint, EffectLog, LogHeader, ProgramIdentity, stable_hash};
pub use recorder::EffectLogRecorder;
pub use replay::{EffectLogReplayer, RequestCheck};

#[cfg(test)]
mod tests;
