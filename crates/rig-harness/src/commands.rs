//! Slash commands. A command is an entity: its [`Name`] is the `/name`
//! typed, its [`SlashCommand`] says what it does, and it is the one-shot
//! system that runs it, so despawning it unregisters both. A plugin adds
//! one with [`AppCommandsExt::add_command`]. Its system takes
//! [`CommandArgs`] to read text after the name, or [`NoArgs`] to refuse it.
//!
//! What the user types that starts with `/` runs as a command: a Gate
//! system in [`CommandSystems`] takes the user's line over
//! ([`Incoming::command`]) and triggers a [`RunCommand`] with it. A plugin
//! that takes such a line over itself orders its Gate system
//! `.before(CommandSystems)`. [`RunCommand`] runs a line at once, with
//! everything it triggers; the notices written meanwhile are its reply. An
//! unknown command, or an error notice about the agent, such as one
//! refusing arguments, refuses the line: it comes back whole as
//! [`Recalled`] with the reason, in place of that notice, so nothing typed
//! is lost.
//!
//! ```
//! use rig_harness::prelude::*;
//!
//! App::new().add_command("hello", "Say hello", greet);
//! fn greet(In(args): In<NoArgs>, mut notices: MessageWriter<Notice>) {
//!     notices.write(Notice::info(args.agent, "Hello."));
//! }
//! ```

use std::mem;

use bevy_app::prelude::*;
use bevy_ecs::error::BevyError;
use bevy_ecs::message::Messages;
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemId;
use bevy_log::warn;
use bevy_reflect::prelude::*;
use rig_core::effect::Decider;
use rig_ecs::agent::{Notice, NoticeLevel};
use rig_ecs::inbox::{DeliveryMode, Incoming, Recalled};
use rig_ecs::stage::{
    AgentPass, Decision, Effect, ForAgent, IncomingMessages, Outcome, ReadyToStart, Stage,
};

/// What a command system receives: the agent the command was typed for,
/// the text after the command name, and how its line was sent.
#[derive(Clone, Debug)]
pub struct CommandArgs {
    /// The agent.
    pub agent: Entity,
    /// The arguments, trimmed.
    pub args: String,
    /// How the line was sent, such as queued, for a command that sends
    /// text on: as [`RunCommand::mode`] says.
    pub mode: DeliveryMode,
}

/// What a command system receives when it accepts no text after its name.
#[derive(Clone, Debug)]
pub struct NoArgs {
    /// The agent.
    pub agent: Entity,
    /// How the line was sent, as [`RunCommand::mode`] says.
    pub mode: DeliveryMode,
}

impl From<CommandArgs> for NoArgs {
    fn from(args: CommandArgs) -> Self {
        Self {
            agent: args.agent,
            mode: args.mode,
        }
    }
}

/// The text a slash command accepts after its name.
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Args {
    /// No text. Extra text refuses the line before the system runs.
    None,
    /// Optional text, passed to the system trimmed.
    Text,
}

mod sealed {
    pub trait Sealed {}
}

/// A slash command's input, either [`NoArgs`] or [`CommandArgs`].
pub trait CommandInput: sealed::Sealed + From<CommandArgs> + 'static {
    /// The text the input accepts after the command name.
    const ARGS: Args;
}

impl sealed::Sealed for CommandArgs {}
impl CommandInput for CommandArgs {
    const ARGS: Args = Args::Text;
}

impl sealed::Sealed for NoArgs {}
impl CommandInput for NoArgs {
    const ARGS: Args = Args::None;
}

/// A slash command, on the entity of the one-shot system it runs, whose
/// [`Name`] is `/` and the name typed.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, Clone, Debug)]
pub struct SlashCommand {
    /// One line of help.
    pub help: String,
    /// The text accepted after the command name, set by its system's input.
    pub args: Args,
}

/// Runs slash commands: what the user types as one, and each
/// [`RunCommand`]. Registering a command adds it too.
pub struct CommandPlugin;

impl Plugin for CommandPlugin {
    fn build(&self, app: &mut App) {
        runner(app);
    }
}

/// The Gate system that runs the user's `/` lines.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CommandSystems;

/// That the app runs commands.
#[derive(Resource)]
struct Runner;

/// Makes the app run commands, once.
fn runner(app: &mut App) {
    if app.world().contains_resource::<Runner>() {
        return;
    }
    app.insert_resource(Runner)
        .configure_sets(AgentPass, CommandSystems.in_set(Stage::Gate))
        .add_systems(AgentPass, run_typed.in_set(CommandSystems))
        .add_observer(on_run_command);
}

/// Takes each user's `/` line about to start over, in the order they were
/// sent, and runs it for its agent, sent as it was.
fn run_typed(
    mut typed: Query<
        (&Effect, &Incoming, &ForAgent, &mut Decision),
        (ReadyToStart, IncomingMessages),
    >,
    mut commands: Commands,
) {
    let mut lines: Vec<_> = typed
        .iter_mut()
        .filter_map(|(effect, message, agent, decision)| {
            Some((
                effect.id(),
                message.command()?,
                message.mode,
                agent.0,
                decision,
            ))
        })
        .collect();
    lines.sort_unstable_by_key(|(id, ..)| *id);
    for (_, line, mode, agent, mut decision) in lines {
        if decision.decide(Decider::plugin("rig-harness"), Outcome::custom("command")) {
            commands.trigger(RunCommand {
                mode,
                ..RunCommand::new(agent, line)
            });
        }
    }
}

/// Registers slash commands on an [`App`], which then runs them
/// ([`CommandPlugin`]). A name already registered is refused with a
/// warning: to replace another plugin's command, insert Bevy's `Disabled`
/// on its entity first, which frees its name.
pub trait AppCommandsExt {
    /// Register `/name`, described by `help`, that runs `system` with the
    /// [`CommandArgs`] or [`NoArgs`] of each use. The input type declares
    /// whether text after the command name is accepted.
    fn add_command<A: CommandInput, M>(
        &mut self,
        name: &str,
        help: &str,
        system: impl IntoSystem<In<A>, (), M> + 'static,
    ) -> &mut Self;
}

impl AppCommandsExt for App {
    fn add_command<A: CommandInput, M>(
        &mut self,
        name: &str,
        help: &str,
        system: impl IntoSystem<In<A>, (), M> + 'static,
    ) -> &mut Self {
        runner(self);
        let name = format!("/{name}");
        if find(self.world_mut(), &name).is_some() {
            warn!("command not registered: {name} exists; insert `Disabled` on it to replace it");
            return self;
        }
        let system = self.register_system(system);
        let help = help.to_owned();
        let command = (
            Name::new(name),
            SlashCommand {
                help,
                args: A::ARGS,
            },
        );
        self.world_mut().entity_mut(system.entity()).insert(command);
        self
    }
}

/// The command named `name`, with its `/`.
fn find(world: &mut World, name: &str) -> Option<(Entity, Args)> {
    world
        .query::<(Entity, &Name, &SlashCommand)>()
        .iter(world)
        .find(|(_, command, _)| command.as_str() == name)
        .map(|(entity, _, command)| (entity, command.args))
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
    /// How the line was sent: the mode of the message typed, else
    /// [`DeliveryMode::Steer`], as by a key or a picker.
    pub mode: DeliveryMode,
}

impl RunCommand {
    /// Run `line` for `agent`, as if typed and sent at once.
    pub fn new(agent: Entity, line: impl Into<String>) -> Self {
        Self {
            entity: agent,
            line: line.into(),
            mode: DeliveryMode::Steer,
        }
    }
}

/// Runs the command a [`RunCommand`] names, with everything it triggers.
fn on_run_command(run: On<RunCommand>, mut commands: Commands) {
    let (agent, line, mode) = (run.entity, run.line.clone(), run.mode);
    commands.queue(move |world: &mut World| run_command(world, agent, line, mode));
}

/// Runs `line` for `agent`, sent as `mode`, keeping the notices written
/// meanwhile apart, and recalls it when refused.
fn run_command(world: &mut World, agent: Entity, line: String, mode: DeliveryMode) {
    let typed = line.trim();
    let (name, args) = typed.split_once(char::is_whitespace).unwrap_or((typed, ""));
    let args = args.trim();
    let name = format!("/{name}");
    let refusals = match find(world, &name) {
        None => vec![format!("Unknown command {name}.")],
        Some((_, Args::None)) if !args.is_empty() => {
            vec![format!("{name} takes no text after it: “{args}”.")]
        }
        Some((system, kind)) => {
            let earlier = swap_notices(world, Messages::default());
            let args = CommandArgs {
                agent,
                args: args.to_owned(),
                mode,
            };
            world.insert_resource(CommandRuns);
            let ran = match kind {
                Args::None => run::<NoArgs>(world, system, args),
                Args::Text => run::<CommandArgs>(world, system, args),
            };
            world.remove_resource::<CommandRuns>();
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

/// While it exists, a command runs, with everything it triggers.
#[derive(Resource)]
pub(crate) struct CommandRuns;

/// Runs the registered system with its declared input type.
fn run<A: CommandInput>(
    world: &mut World,
    system: Entity,
    args: CommandArgs,
) -> Result<(), BevyError> {
    world
        .run_system_with(SystemId::<In<A>>::from_entity(system), A::from(args))
        .map_err(BevyError::from)
}

/// Puts `notices` in place of the world's, and returns those.
fn swap_notices(world: &mut World, notices: Messages<Notice>) -> Messages<Notice> {
    match world.get_resource_mut::<Messages<Notice>>() {
        Some(mut current) => mem::replace(&mut *current, notices),
        None => notices,
    }
}

#[cfg(test)]
mod tests;
