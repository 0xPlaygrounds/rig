//! rig-code: a minimal coding agent built from Bevy plugins.
//!
//! Agents are entities, their model and tool calls are entities tied to
//! them, and every call goes through one recorded dispatch path. Tools and
//! slash commands are registered by plugins through [`RigCodeAppExt`]; the
//! built-in ones use the same methods a third-party plugin does.
//!
//! ```no_run
//! fn main() -> bevy_app::AppExit {
//!     rig_code::app().run()
//! }
//! ```

pub mod agent;
pub mod commands;
pub mod effects;
pub mod model;
mod process;
pub mod reload;
pub mod session;
pub mod tools;
#[cfg(feature = "tui")]
pub mod tui;
pub mod turn;

use std::time::Duration;

use bevy_app::{App, PluginGroup, PluginGroupBuilder, ScheduleRunnerPlugin, TaskPoolPlugin};
use bevy_ecs::{prelude::*, system::IntoSystem};
use bevy_log::LogPlugin;
use rig_core::{
    serve::{ErasedHandler, adapters::ToolAdapter},
    tool::{Tool, tool_definition},
};

pub use bevy_app;
pub use bevy_ecs;
pub use bevy_log;
pub use bevy_reflect;
pub use bevy_tasks;
pub use rig_core;

use crate::{
    commands::{CommandArgs, CommandsPlugin, SlashCommand},
    reload::ReloadPlugin,
    session::SessionPlugin,
    tools::{ToolDef, ToolsPlugin},
    turn::AgentPlugin,
};

/// What a plugin usually needs.
pub mod prelude {
    pub use crate::{
        RigCodeAppExt, RigCodePlugins,
        agent::{
            Agent, AgentId, AgentStatus, Choice, Choose, Conversation, Notice, RigSet, Submit,
            TurnEnded,
        },
        commands::CommandArgs,
    };
    pub use bevy_app::prelude::*;
    pub use bevy_ecs::prelude::*;
}

/// How long the app waits between frames when idle.
const FRAME: Duration = Duration::from_millis(16);

/// The app with every built-in plugin, logging to the session's file and
/// logging failing systems instead of panicking.
pub fn app() -> App {
    session::log_panics();
    let mut app = App::new();
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins(RigCodePlugins);
    app
}

/// The built-in plugins, in order: session, log, task pools and the frame
/// loop, the agent loop, tools, commands, `/reload`, and the terminal view.
pub struct RigCodePlugins;

impl PluginGroup for RigCodePlugins {
    fn build(self) -> PluginGroupBuilder {
        let group = PluginGroupBuilder::start::<Self>()
            .add(SessionPlugin)
            .add(LogPlugin {
                fmt_layer: session::file_log_layer,
                ..Default::default()
            })
            .add(TaskPoolPlugin::default())
            .add(ScheduleRunnerPlugin::run_loop(FRAME))
            .add(AgentPlugin)
            .add(ToolsPlugin)
            .add(CommandsPlugin)
            .add(ReloadPlugin);
        #[cfg(feature = "tui")]
        let group = group.add(tui::TuiPlugin);
        group
    }
}

/// Registration of tools and slash commands, for plugins.
pub trait RigCodeAppExt {
    /// Offer `tool` to the model. Its calls are recorded like every effect.
    fn add_tool<T: Tool + 'static>(&mut self, tool: T) -> &mut Self;

    /// Add `/name`, running `system` with the agent and the text after the
    /// name.
    fn add_command<M>(
        &mut self,
        name: &str,
        description: &str,
        system: impl IntoSystem<In<CommandArgs>, (), M> + 'static,
    ) -> &mut Self;
}

impl RigCodeAppExt for App {
    fn add_tool<T: Tool + 'static>(&mut self, tool: T) -> &mut Self {
        let definition = tool_definition(&tool);
        let handler = ErasedHandler::new(ToolAdapter::new(tool));
        self.world_mut().spawn(ToolDef {
            definition,
            handler,
        });
        self
    }

    fn add_command<M>(
        &mut self,
        name: &str,
        description: &str,
        system: impl IntoSystem<In<CommandArgs>, (), M> + 'static,
    ) -> &mut Self {
        let world = self.world_mut();
        let mut existing = world.query::<&SlashCommand>();
        if existing.iter(world).any(|command| command.name == name) {
            bevy_log::warn!("not adding /{name}: another plugin already added it");
            return self;
        }
        let run = self.register_system(system);
        self.world_mut().spawn(SlashCommand {
            name: name.to_owned(),
            description: description.to_owned(),
            run,
        });
        self
    }
}
