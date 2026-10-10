//! Slash commands. A command is an entity: its [`Name`] is the `/name`
//! typed, its [`SlashCommand`] says what it does, and it is the one-shot
//! system that runs it, so despawning it unregisters both. A plugin adds
//! one with [`AppCommandsExt::add_command`], a system that reads the
//! [`CommandArgs`], or with [`AppCommandsExt::add_command_event`], an event
//! filled from the line by its reflected fields.
//!
//! [`RunCommand`] runs a line at once, with everything it triggers. The
//! notices written meanwhile are its reply. An unknown command, arguments
//! its event cannot take, or an error notice about the agent refuse the
//! line: it comes back whole as [`Recalled`] with the reason, in place of
//! that notice, so nothing typed is lost.

use std::mem;

use bevy_app::App;
use bevy_ecs::message::Messages;
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemId;
use bevy_log::warn;
use bevy_reflect::prelude::*;
use bevy_reflect::structs::DynamicStruct;
use bevy_reflect::{TypeInfo, Typed};

use super::agent::{Notice, NoticeLevel};
use super::inbox::Recalled;

/// What a command system receives: the agent the command was typed for and
/// the text after the command name.
#[derive(Clone, Debug)]
pub struct CommandArgs {
    /// The agent.
    pub agent: Entity,
    /// The arguments, trimmed.
    pub args: String,
}

/// A slash command, on the entity of the one-shot system it runs, whose
/// [`Name`] is `/` and the name typed.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, Clone, Debug)]
pub struct SlashCommand {
    /// One line of help.
    pub help: String,
    /// The type path of the event it triggers, for one added with
    /// [`AppCommandsExt::add_command_event`].
    pub event: Option<String>,
}

/// Registers slash commands on an [`App`]. A name already registered is
/// refused with a warning.
pub trait AppCommandsExt {
    /// Register `/name`, described by `help`, that runs `system` with the
    /// [`CommandArgs`] of each use.
    fn add_command<M>(
        &mut self,
        name: &str,
        help: &str,
        system: impl IntoSystem<In<CommandArgs>, (), M> + 'static,
    ) -> &mut Self;

    /// Register `/name`, described by `help`, that triggers the event `E`,
    /// its fields filled by reflection: an [`Entity`] with the agent, a
    /// `String` with the arguments. Arguments an event without a `String`
    /// does not take refuse the line. An event with a field of another
    /// type is refused with a warning.
    fn add_command_event<E>(&mut self, name: &str, help: &str) -> &mut Self
    where
        E: Event + FromReflect + Typed,
        for<'a> E::Trigger<'a>: Default;
}

impl AppCommandsExt for App {
    fn add_command<M>(
        &mut self,
        name: &str,
        help: &str,
        system: impl IntoSystem<In<CommandArgs>, (), M> + 'static,
    ) -> &mut Self {
        register(self, name, help, None, system)
    }

    fn add_command_event<E>(&mut self, name: &str, help: &str) -> &mut Self
    where
        E: Event + FromReflect + Typed,
        for<'a> E::Trigger<'a>: Default,
    {
        let TypeInfo::Struct(info) = E::type_info() else {
            warn!(
                "command not registered: /{name}: {} is not a struct",
                E::type_path()
            );
            return self;
        };
        if let Some(field) = info.iter().find(|f| !f.is::<Entity>() && !f.is::<String>()) {
            let (event, field) = (E::type_path(), field.name());
            warn!("command not registered: /{name}: commands cannot fill {event}'s {field}");
            return self;
        }
        let takes_text = info.iter().any(|field| field.is::<String>());
        let command = format!("/{name}");
        let trigger = move |In(args): In<CommandArgs>,
                            mut commands: Commands,
                            mut notices: MessageWriter<Notice>| {
            if !takes_text && !args.args.is_empty() {
                let why = format!("{command} takes no arguments, not `{}`.", args.args);
                notices.write(Notice::error(args.agent, why));
                return;
            }
            let mut fields = DynamicStruct::default();
            for field in info.iter() {
                match field.is::<Entity>() {
                    true => fields.insert(field.name(), args.agent),
                    false => fields.insert(field.name(), args.args.clone()),
                }
            }
            if let Some(event) = E::from_reflect(&fields) {
                commands.trigger(event);
            }
        };
        register(self, name, help, Some(E::type_path()), trigger)
    }
}

fn register<'a, M>(
    app: &'a mut App,
    name: &str,
    help: &str,
    event: Option<&str>,
    system: impl IntoSystem<In<CommandArgs>, (), M> + 'static,
) -> &'a mut App {
    let name = format!("/{name}");
    if find(app.world_mut(), &name).is_some() {
        warn!("command not registered: {name} already exists");
        return app;
    }
    let system = app.register_system(system);
    app.world_mut().entity_mut(system.entity()).insert((
        Name::new(name),
        SlashCommand {
            help: help.to_owned(),
            event: event.map(str::to_owned),
        },
    ));
    app
}

/// The command named `name`, with its `/`.
fn find(world: &mut World, name: &str) -> Option<Entity> {
    world
        .query_filtered::<(Entity, &Name), With<SlashCommand>>()
        .iter(world)
        .find(|(_, command)| command.as_str() == name)
        .map(|(entity, _)| entity)
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

/// Runs the command a [`RunCommand`] names, with everything it triggers.
pub(crate) fn on_run_command(run: On<RunCommand>, mut commands: Commands) {
    let (agent, line) = (run.entity, run.line.clone());
    commands.queue(move |world: &mut World| run_command(world, agent, line));
}

/// Runs `line` for `agent`, keeping the notices written meanwhile apart,
/// and recalls it when refused.
fn run_command(world: &mut World, agent: Entity, line: String) {
    let typed = line.trim();
    let (name, args) = typed.split_once(char::is_whitespace).unwrap_or((typed, ""));
    let name = format!("/{name}");
    let refusals = match find(world, &name) {
        None => vec![format!("Unknown command {name}.")],
        Some(system) => {
            let earlier = swap_notices(world, Messages::default());
            let ran = world.run_system_with(
                SystemId::<In<CommandArgs>>::from_entity(system),
                CommandArgs {
                    agent,
                    args: args.trim().to_owned(),
                },
            );
            let mut reply = swap_notices(world, earlier);
            let mut refusals: Vec<String> =
                ran.err().map(|ran| ran.to_string()).into_iter().collect();
            for notice in reply.drain() {
                let refusal = notice.level == NoticeLevel::Error
                    && notice.agent.is_none_or(|about| about == agent);
                if refusal {
                    refusals.push(notice.text);
                } else {
                    world.write_message(notice);
                }
            }
            refusals
        }
    };
    if !refusals.is_empty() {
        world.write_message(Recalled {
            agent,
            text: format!("/{line}"),
            why: Some(refusals.join(" ")),
        });
    }
}

/// Puts `notices` in place of the world's, and returns those.
fn swap_notices(world: &mut World, notices: Messages<Notice>) -> Messages<Notice> {
    match world.get_resource_mut::<Messages<Notice>>() {
        Some(mut current) => mem::replace(&mut *current, notices),
        None => notices,
    }
}
