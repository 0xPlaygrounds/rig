//! The agent core: agents, their loop, tools, commands and effects. It
//! knows nothing of the terminal, so another view can drive it through the
//! same messages.
//!
//! Views write [`Submit`] and [`Interrupt`] and read [`Notice`] and
//! [`ChoiceRequested`]. The loop runs in [`AgentSystems`] in `Update`.

use bevy::{ecs::error::warn, prelude::*};

pub mod agent;
pub mod catalog;
pub mod command;
pub mod dispatch;
pub mod paths;
pub mod session;
pub mod tools;
mod turn;

use agent::Agent;
use catalog::Providers;
use command::CommandInput;
use dispatch::Effects;

/// Text typed for an agent: a message, or a slash command when it starts
/// with `/`.
#[derive(Message, Clone, Debug)]
pub struct Submit {
    /// The agent.
    pub agent: Entity,
    /// The text.
    pub text: String,
}

/// Stop the agent's running turn.
#[derive(Message, Clone, Copy, Debug)]
pub struct Interrupt {
    /// The agent.
    pub agent: Entity,
}

/// How a [`Notice`] reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoticeLevel {
    /// Information.
    Info,
    /// Something failed.
    Error,
}

/// A line for the user about an agent, outside the conversation.
#[derive(Message, Clone, Debug)]
pub struct Notice {
    /// The agent.
    pub agent: Entity,
    /// How it reads.
    pub level: NoticeLevel,
    /// The text.
    pub text: String,
}

impl Notice {
    /// Information about `agent`.
    pub fn info(agent: Entity, text: impl Into<String>) -> Self {
        Self {
            agent,
            level: NoticeLevel::Info,
            text: text.into(),
        }
    }

    /// A failure about `agent`.
    pub fn error(agent: Entity, text: impl Into<String>) -> Self {
        Self {
            agent,
            level: NoticeLevel::Error,
            text: text.into(),
        }
    }
}

/// What a [`ChoiceRequested`] asks the user to pick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChoiceKind {
    /// A model, answered with `/model <reference>`.
    Model,
    /// An effort, answered with `/effort <name>`.
    Effort,
}

/// A command asks the view to let the user pick; the view answers with a
/// [`Submit`] of the command with the choice as its argument.
#[derive(Message, Clone, Copy, Debug)]
pub struct ChoiceRequested {
    /// The agent.
    pub agent: Entity,
    /// What to pick.
    pub kind: ChoiceKind,
}

/// An agent's turn ended: it is idle again, after an answer, an error or an
/// interrupt.
#[derive(EntityEvent, Clone, Copy, Debug)]
pub struct TurnEnded {
    /// The agent.
    pub entity: Entity,
}

/// The agent loop's stages, chained in `Update`.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AgentSystems {
    /// Read submits and interrupts.
    Input,
    /// Start model calls.
    Start,
    /// Collect streamed text and finished calls.
    Collect,
}

/// Registers tools and slash commands from plugins.
pub trait RigAppExt {
    /// Make `tool` callable by agents.
    fn add_tool<T: rig_core::tool::Tool + 'static>(&mut self, tool: T) -> &mut Self;

    /// Make `system` run when `/name` is submitted. `help` is its line in
    /// `/help`.
    fn add_command<M>(
        &mut self,
        name: &str,
        help: &str,
        system: impl IntoSystem<In<CommandInput>, (), M> + 'static,
    ) -> &mut Self;
}

impl RigAppExt for App {
    fn add_tool<T: rig_core::tool::Tool + 'static>(&mut self, tool: T) -> &mut Self {
        tools::register(self, tool);
        self
    }

    fn add_command<M>(
        &mut self,
        name: &str,
        help: &str,
        system: impl IntoSystem<In<CommandInput>, (), M> + 'static,
    ) -> &mut Self {
        command::register(self, name, help, system);
        self
    }
}

/// Agents, their loop and the effect log. A failing system or command is
/// logged as a warning instead of stopping the app, because plugins are
/// untrusted.
pub struct AgentPlugin;

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        app.set_error_handler(warn)
            .add_message::<Submit>()
            .add_message::<Interrupt>()
            .add_message::<Notice>()
            .add_message::<ChoiceRequested>()
            .insert_resource(Providers::from_env())
            .insert_resource(Effects::new(paths::effects_file()))
            .configure_sets(
                Update,
                (
                    AgentSystems::Input,
                    AgentSystems::Start,
                    AgentSystems::Collect,
                )
                    .chain(),
            )
            .add_systems(PostStartup, spawn_first_agent)
            .add_systems(
                Update,
                (
                    (turn::read_submits, turn::read_interrupts)
                        .chain()
                        .in_set(AgentSystems::Input),
                    turn::start_model_calls.in_set(AgentSystems::Start),
                    (
                        turn::drain_streams,
                        turn::finish_model_calls,
                        turn::finish_tool_calls,
                    )
                        .chain()
                        .in_set(AgentSystems::Collect),
                ),
            )
            .add_systems(
                Last,
                flush_effects
                    .in_set(bevy::app::OnAppExitSystems)
                    .run_if(on_message::<AppExit>),
            )
            .add_observer(|_: On<TurnEnded>, effects: ResMut<Effects>| flush_effects(effects));
    }
}

/// Start with one agent when none exists yet.
fn spawn_first_agent(agents: Query<(), With<Agent>>, mut commands: Commands) {
    if agents.is_empty() {
        commands.spawn(Agent);
    }
}

/// Append resolved effects to the log: after every turn, and on exit.
fn flush_effects(mut effects: ResMut<Effects>) {
    if let Err(error) = effects.flush() {
        error!("cannot write the effect log: {error}");
    }
}
