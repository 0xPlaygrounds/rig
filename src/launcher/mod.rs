//! The launcher's parts: the `RIG_HOME` layout, the `rig.toml` plugin list,
//! the generated agent project, its build, and the run loop.

pub mod build;
pub mod config;
pub mod home;
pub mod project;
pub mod run;

/// Every launcher failure is reported to the user as its message.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The launcher's version. It stamps the generated project, so a new
/// launcher regenerates and rebuilds the agent.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The one Bevy version of the agent and of every plugin.
pub const BEVY_VERSION: &str = "0.20.0-rc.2";

/// The exit code with which the agent asks to be restarted on the staged
/// build.
pub const RELOAD_EXIT_CODE: i32 = 75;
