//! The synchronisation primitives the agent's driver protocol is written
//! against: `std`'s, or `loom`'s under `--cfg rig_loom`, so the same code
//! is what the model checker explores (`agent/bus/loom_models.rs`).

#[cfg(rig_loom)]
pub(crate) use loom::sync::{Mutex, MutexGuard};
#[cfg(not(rig_loom))]
pub(crate) use std::sync::{Mutex, MutexGuard};
