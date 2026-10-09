//! How this app runs as a process: the session directory and its log, the
//! project context it reads, `/reload`, and
//! the `rig` launcher protocol.
//! The host depends on the core; the core never depends on the host.

pub mod compaction;
pub mod context;
pub mod defaults;
pub mod headless;
pub mod launcher;
pub mod reload;
pub mod session;
pub mod sessions;
pub mod signals;
