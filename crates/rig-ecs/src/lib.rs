//! rig inside a Bevy `World`.
//!
//! One module today, [`bus`]: the effect bus as a plugin — effects are
//! entities, handlers are entities, the driver is a system, an outcome is a
//! component, causality is `ChildOf`, a scene is a checkpoint. Nothing in
//! this crate awaits, blocks, or holds a future for a host to probe.
//!
//! The `bus` module is written as if it were already its own crate (every
//! item `pub`, no import from a sibling module, no agent-shaped item, its
//! tests in `tests/bus_*.rs`): the agent runtime the later modules add
//! consumes it through its public items only, and it becomes `rig-bevy` by
//! a `git mv` when a second consumer exists.

pub mod bus;
