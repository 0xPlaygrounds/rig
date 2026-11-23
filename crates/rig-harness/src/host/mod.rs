//! What the binary needs to run as a process and be relaunched: the
//! session directory and its log, the loop and its signals, the `rig`
//! launcher protocol with the plugins' options and fronts, and why the app
//! exits.

pub mod cli;
pub mod exit;
pub mod launcher;
pub mod pipes;
pub mod runner;
pub mod session;
pub mod signals;
