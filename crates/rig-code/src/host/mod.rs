//! How this app runs as a process: the session directory and its log, the
//! child processes it starts, `/reload`, and the `rig` launcher protocol.
//! The host depends on the core; the core never depends on the host.

pub mod launcher;
pub(crate) mod process;
pub mod reload;
pub mod session;
pub mod signals;
