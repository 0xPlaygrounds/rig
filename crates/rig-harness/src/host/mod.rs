//! How this app runs as a process: the session directory and its log, the
//! project context it reads, `/reload`, and
//! the `rig` launcher protocol, with what the model knows about extending itself.
//! The host depends on the core; the core never depends on the host.

pub mod context;
pub mod defaults;
mod extending;
pub mod headless;
pub mod launcher;
pub mod reload;
pub mod runner;
pub mod session;
pub mod sessions;
pub mod signals;
