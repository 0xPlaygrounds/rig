//! The synchronisation primitives the crate's protocols are written
//! against — the bus's (`bus/`) and the agent's driver's (`agent/drive.rs`):
//! `std`'s, or `loom`'s under `--cfg rig_loom`, so the same code is what
//! the model checker explores (`bus/loom_models.rs`,
//! `agent/drive/loom_models.rs`).

#[cfg(rig_loom)]
pub(crate) use loom::sync::{Mutex, MutexGuard, RwLock, atomic};
#[cfg(not(rig_loom))]
pub(crate) use std::sync::{Mutex, MutexGuard, RwLock, atomic};
