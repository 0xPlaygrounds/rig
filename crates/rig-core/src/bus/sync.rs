//! The synchronisation primitives the bus's protocols are written against:
//! `std`'s, or `loom`'s under `--cfg rig_loom`, so the same code is what the
//! model checker explores (`bus/loom_models.rs`).

#[cfg(rig_loom)]
pub(crate) use loom::sync::{Mutex, RwLock, atomic};
#[cfg(not(rig_loom))]
pub(crate) use std::sync::{Mutex, RwLock, atomic};
