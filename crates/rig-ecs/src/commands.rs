//! The slash command registry. A command is a registered one-shot system
//! whose own entity carries its [`SlashCommand`], so despawning that entity
//! unregisters the command and its system together. Plugins add commands
//! with [`AppCommandsExt::add_command`]. [`RunCommand`] runs one.

use bevy_app::App;
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemId;
use bevy_log::warn;
use bevy_reflect::prelude::*;

use super::agent::Notice;

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

/// Run the slash command `line` (without its `/`) for the agent: its name,
/// then its arguments.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct RunCommand {
    /// The agent.
    pub entity: Entity,
    /// The command line after the `/`.
    pub line: String,
}

/// Runs the command a [`RunCommand`] names, or says it does not exist.
pub(crate) fn on_run_command(
    run: On<RunCommand>,
    slash: Query<(Entity, &SlashCommand)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = run.entity;
    let line = run.line.trim();
    let (name, args) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
    match slash.iter().find(|(_, command)| command.name == name) {
        // The command sits on its system's own entity.
        Some((system, _)) => commands.run_system_with(
            SystemId::<In<CommandArgs>>::from_entity(system),
            CommandArgs {
                agent,
                args: args.trim().to_owned(),
            },
        ),
        None => {
            // /help comes from a plugin, so point at it only when loaded.
            let hint = if slash.iter().any(|(_, command)| command.name == "help") {
                " /help lists the commands."
            } else {
                ""
            };
            notices.write(Notice::error(
                agent,
                format!("Unknown command /{name}.{hint}"),
            ));
        }
    }
}
