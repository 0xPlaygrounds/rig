//! A small coding agent built on Rig and Bevy.
//!
//! The agent is a Bevy app. Agents are entities whose conversation, model
//! and settings are components; model and tool calls run on Bevy's task
//! pools and are recorded through rig-core's effect types. Tools, slash
//! commands and the terminal view are ordinary Bevy plugins, registered
//! through [`AgentAppExt`] exactly as a third-party plugin registers its
//! own.
//!
//! The agent reads its data directory from `RIG_DATA_DIR`, or
//! `$RIG_HOME/data`, and refuses to start without one.
//!
//! ```no_run
//! fn main() -> rig_code::bevy::app::AppExit {
//!     rig_code::run(|app| {
//!         app.add_plugins(rig_code::BuiltinTools);
//!         app.add_plugins(rig_code::BuiltinCommands);
//!         app.add_plugins(rig_code::TuiPlugin);
//!     })
//! }
//! ```

pub use bevy;

mod commands;
mod core;
mod tools;
mod tui;

pub use crate::commands::BuiltinCommands;
pub use crate::core::*;
pub use crate::tools::BuiltinTools;
pub use crate::tui::TuiPlugin;

use bevy::app::{App, AppExit};

/// Builds the agent app, lets `add_plugins` add the plugin list, and runs
/// it until it exits. Returns an error exit, without starting, when no data
/// directory is set.
pub fn run(add_plugins: impl FnOnce(&mut App)) -> AppExit {
    let Some(data) = crate::core::app::data_dir() else {
        // The terminal is not set up yet; this is the one message the agent
        // prints itself.
        eprintln!("rig-code: set RIG_DATA_DIR or RIG_HOME; the rig launcher sets both");
        return AppExit::error();
    };
    let mut app = crate::core::app::base_app(data);
    app.add_plugins(crate::core::CorePlugin);
    add_plugins(&mut app);
    app.run()
}
