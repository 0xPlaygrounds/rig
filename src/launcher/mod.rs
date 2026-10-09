//! The launcher's parts: the locks on `RIG_HOME` (whose layout, shared with
//! the agent, is [`rig::harness_protocol`]), the `plugins.toml` plugin list and
//! `rig plugin`, the generated agent project, its build, and the run loop.

pub mod build;
pub mod config;
pub mod home;
pub mod plugin;
pub mod project;
pub mod run;

/// Every launcher failure is reported to the user as its message.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The launcher's version. It stamps the generated project, so a new
/// launcher regenerates and rebuilds the agent.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub use rig::harness_protocol::BEVY_VERSION;
