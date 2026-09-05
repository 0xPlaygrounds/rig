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
//! Custom outcomes use an explicit `payload` field beneath the `outcome`
//! tag, so strings, numbers, booleans, null, arrays and objects retain their
//! original JSON value. The previous flattened custom outcome form is not
//! emitted; regenerate affected fixtures through their owning producers.
//!
//! A replayer preserves a recorded handler's semantic family, including model
//! identity and capabilities, even when that key has records. Required keys
//! include every scoped [`ProgramIdentity`] row. Conflicting declarations are
//! refused. Executable middleware is reapplied by the caller; replayers do not
//! claim to have executed the recorded descriptor's layers. Descriptor-less
//! legacy exchanges may infer a fallback description, which cannot establish
//! verified semantic compatibility for a program.
//!
//! [`LogHeader::deliveries`] optionally records consumer-visible outcome and
//! stream batches. rig-ecs records and enforces these schedule boundaries;
//! the shared replayer alone only supplies exchanges and event sequences.
//! rig-agent's bus does not record ECS schedule boundaries. A log without
//! this metadata, or a folded stream without event bytes, cannot prove exact
//! partial-state or first-visible-answer policy replay. Batch numbers are
//! not a clock, a whole-world snapshot, or an external-side-effect guarantee.
//! Kept streams also retain error items in [`LogHeader::stream_errors`], with
//! positions among all items. Errors before or after `Final` remain in their
//! original order; the folded outcome alone cannot reconstruct them. Empty
//! metadata is omitted on the wire. Historical logs lacking error positions
//! can replay their successful events and a folded error, but cannot prove
//! the original sequence of error items.
//!
//! The vocabulary is rig-core's, the runtimes are rig-agent's bus and rig-ecs; this crate is
//! the persistence story over both, and what a host that saves and restores
//! in-flight effects (a scene, a durable run) depends on.

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
