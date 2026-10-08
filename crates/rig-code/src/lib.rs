//! rig-code: a terminal coding agent built as a Bevy app on rig-core.
//!
//! An agent is an entity whose components hold its conversation, model,
//! reasoning setting, system prompt, tool access and status. Model calls and
//! tool calls are entities owned by their agent and run on Bevy's task
//! pools; every one goes through the one recorded dispatch path. Tools and
//! slash commands are registered by Bevy plugins, the built-in ones exactly
//! as a third-party plugin registers its own.
//!
//! ```no_run
//! use rig_code::prelude::*;
//!
//! fn main() -> AppExit {
//!     App::new().add_plugins(RigCodePlugins).run()
//! }
//! ```

pub mod builtin;
pub mod core;
mod process;
pub mod reload;
#[cfg(feature = "tui")]
pub mod tui;

use std::time::Duration;

use bevy_app::{PluginGroup, PluginGroupBuilder, ScheduleRunnerPlugin, TaskPoolPlugin};
use bevy_log::LogPlugin;

pub use bevy_app::{App, AppExit};

/// What a plugin needs: Bevy's app and ECS preludes, the agent components
/// and requests, and the tool and command registries.
pub mod prelude {
    pub use bevy_app::prelude::*;
    pub use bevy_ecs::prelude::*;
    pub use bevy_reflect::prelude::*;

    pub use crate::RigCodePlugins;
    pub use crate::core::agent::{
        Agent, AgentId, AgentStatus, Conversation, Effort, Interrupt, ModelChoice, Notice,
        SetEffort, SetModel, Submit, SystemPrompt, ToolAccess, TurnFinished,
    };
    pub use crate::core::commands::{AppCommandsExt, CommandArgs};
    pub use crate::core::session::ReflectSaved;
    pub use crate::core::tools::AppToolsExt;
    pub use rig_core::tool::{PortableTool, Tool, ToolExecutionError};
}

/// The rig-code app: the session and its log, Bevy's task pools and a
/// 60 Hz loop, the agent core, the built-in tools and commands, `/reload`,
/// and the terminal view (feature `tui`).
pub struct RigCodePlugins;

impl PluginGroup for RigCodePlugins {
    fn build(self) -> PluginGroupBuilder {
        let group = PluginGroupBuilder::start::<Self>()
            .add(core::session::SessionPlugin)
            .add(LogPlugin {
                fmt_layer: core::session::log_layer,
                ..LogPlugin::default()
            })
            .add(TaskPoolPlugin::default())
            .add(ScheduleRunnerPlugin::run_loop(Duration::from_millis(16)))
            .add(core::AgentPlugin)
            .add(builtin::BuiltinToolsPlugin)
            .add(builtin::BuiltinCommandsPlugin)
            .add(reload::ReloadPlugin);
        #[cfg(feature = "tui")]
        let group = group.add(tui::TuiPlugin);
        group
    }
}
