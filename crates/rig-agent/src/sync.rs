//! Synchronization primitives shared by bus and driver protocols. `rig_loom`
//! selects Loom primitives so model checking exercises the production protocol.

#[cfg(rig_loom)]
pub(crate) use loom::sync::{Mutex, MutexGuard, RwLock, atomic};
#[cfg(not(rig_loom))]
pub(crate) use std::sync::{Mutex, MutexGuard, RwLock, atomic};
