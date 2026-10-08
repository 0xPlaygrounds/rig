//! The slash command registry. A command is an entity holding a registered
//! one-shot system; plugins add commands with
//! [`AppCommandsExt::add_command`].

use bevy_app::App;
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemId;

/// What a command system receives: the agent the command was typed for and
/// the text after the command name.
#[derive(Clone, Debug)]
pub struct CommandArgs {
    /// The agent.
    pub agent: Entity,
    /// The arguments, trimmed.
    pub args: String,
}

/// A slash command: its name without the `/`, its help line, and its
/// system.
#[derive(Component, Clone)]
pub struct SlashCommand {
    /// The name typed after `/`.
    pub name: String,
    /// One line of help.
    pub help: String,
    /// The system run with the command's [`CommandArgs`].
    pub system: SystemId<In<CommandArgs>>,
}

/// Registers slash commands on an [`App`].
pub trait AppCommandsExt {
    /// Register `/name`, described by `help`, that runs `system` with the
    /// [`CommandArgs`] of each use.
    fn add_command<M>(
        &mut self,
        name: &str,
        help: &str,
        system: impl IntoSystem<In<CommandArgs>, (), M> + 'static,
    ) -> &mut Self;
}

impl AppCommandsExt for App {
    fn add_command<M>(
        &mut self,
        name: &str,
        help: &str,
        system: impl IntoSystem<In<CommandArgs>, (), M> + 'static,
    ) -> &mut Self {
        let system = self.register_system(system);
        self.world_mut().spawn((
            Name::new(format!("command:/{name}")),
            SlashCommand {
                name: name.to_owned(),
                help: help.to_owned(),
                system,
            },
        ));
        self
    }
}
