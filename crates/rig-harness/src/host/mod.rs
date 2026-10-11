//! What the binary needs to run as a process and be relaunched: the
//! session directory and its log, the loop and its signals, and the `rig`
//! launcher protocol.

pub mod launcher;
pub mod runner;
pub mod session;
pub mod signals;
