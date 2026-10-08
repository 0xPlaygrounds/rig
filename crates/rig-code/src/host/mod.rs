//! How this app runs as a process: the session directory and its log, the
//! project context it reads, the child processes it starts, `/reload`, and
//! the `rig` launcher protocol.
//! The host depends on the core; the core never depends on the host.

pub mod context;
pub mod launcher;
pub(crate) mod process;
pub mod reload;
pub mod runner;
pub mod session;
pub mod sessions;
pub mod signals;
