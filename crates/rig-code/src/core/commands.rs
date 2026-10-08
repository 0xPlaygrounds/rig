//! The slash command registry. A command is a registered one-shot system
//! whose own entity carries its [`SlashCommand`], so despawning that entity
//! unregisters the command and its system together. Plugins add commands
//! with [`AppCommandsExt::add_command`].

use bevy_app::App;
use bevy_ecs::prelude::*;
use bevy_log::warn;

/// What a command system receives: the agent the command was typed for and
/// the text after the command name.
#[derive(Clone, Debug)]
pub struct CommandArgs {
    /// The agent.
    pub agent: Entity,
    /// The arguments, trimmed.
    pub args: String,
}

/// A slash command, on the entity of the one-shot system it runs: its name
/// without the `/` and its help line.
#[derive(Component, Clone)]
pub struct SlashCommand {
    /// The name typed after `/`.
    pub name: String,
    /// One line of help.
    pub help: String,
}

/// Registers slash commands on an [`App`].
pub trait AppCommandsExt {
    /// Register `/name`, described by `help`, that runs `system` with the
    /// [`CommandArgs`] of each use. A name already registered is refused
    /// with a warning.
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
        let world = self.world_mut();
        if world
            .query::<&SlashCommand>()
            .iter(world)
            .any(|command| command.name == name)
        {
            warn!("command not registered: /{name} already exists");
            return self;
        }
        let system = self.register_system(system);
        self.world_mut().entity_mut(system.entity()).insert((
            Name::new(format!("command:/{name}")),
            SlashCommand {
                name: name.to_owned(),
                help: help.to_owned(),
            },
        ));
        self
    }
}
