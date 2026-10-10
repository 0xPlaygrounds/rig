//! What the binary needs to run as a process and be relaunched: the
//! session directory and its log, the loop and its signals, the `rig`
//! launcher protocol and `/reload`.

pub mod launcher;
pub mod reload;
pub mod runner;
pub mod session;
pub mod signals;
