//! The launcher's parts: the locks on `RIG_HOME` (whose layout, shared with
//! the agent, is [`rig_harness_protocol`]), the `plugins.toml` plugin list and
//! `rig plugin`, the generated agent project, its build, and the run loop.

pub mod build;
pub mod cli;
pub mod locks;
pub mod plugin;
pub mod plugin_list;
pub mod project;
pub mod run;

/// Every launcher failure is reported to the user as its message.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub use rig_harness_protocol::{BEVY_VERSION, VERSION};
