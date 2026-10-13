//! The basic slash commands: `/help`, `/retry` and `/quit`, with the
//! default `tui` feature `/agents` and the agent tree in each agent's
//! status line ([`BasicCommandsPlugin`]), and the project context in the
//! system prompt ([`ProjectContextPlugin`]): rig's own prompt for an agent
//! without one, `AGENTS.md`, the environment and `/context`. Other plugins
//! add to what `/context` lists with [`ContextEntry`]s, and to each
//! agent's line in `/agents` with [`AgentColumn`]s.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;

use rig_ecs::agent::{Notice, Retry};
use rig_harness::commands::{AppCommandsExt, CommandArgs, SlashCommand};
use rig_harness::host::exit::{Exit, ExitReason};

pub mod project_context;
#[cfg(feature = "tui")]
mod tui;

pub use project_context::{
    ContextEntry, ContextLimits, DEFAULT_SYSTEM_PROMPT, ProjectContextPlugin, RefreshProjectContext,
};

/// Registers the basic commands; with the `tui` feature, `/agents` and the
/// agent tree it picks from in each agent's status line.
#[derive(Default)]
pub struct BasicCommandsPlugin;

impl Plugin for BasicCommandsPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "retry",
            "Send the conversation again after a failed model call",
            retry,
        )
        .add_command("help", "List the commands", help)
        .add_command("quit", "Quit; the session stays for /resume", quit);
        #[cfg(feature = "tui")]
        tui::add(app);
    }
}

/// A column of an agent's line in `/agents`, such as what it spent or why
/// it waits: spawn one with a [`ColumnOf`] the agent and keep its `text`
/// current. Columns follow the agent's name, model and state, by `tag`;
/// an empty `text`, or Bevy's `Disabled` on the entity, leaves one out.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Clone, Debug, Default)]
pub struct AgentColumn {
    /// What it is, and what sorts it, such as `spent`.
    pub tag: String,
    /// What the line shows.
    pub text: String,
}

impl AgentColumn {
    /// `text` in an agent's line, sorted by `tag`.
    pub fn new(tag: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            text: text.into(),
        }
    }
}

/// On an [`AgentColumn`]: the agent whose line it is in. Despawning the
/// agent despawns it; a restored agent is a new entity, so a plugin
/// spawns its columns again.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship(relationship_target = AgentColumns)]
pub struct ColumnOf(pub Entity);

/// The [`AgentColumn`]s of this agent.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship_target(relationship = ColumnOf, linked_spawn)]
pub struct AgentColumns(Vec<Entity>);

/// `/retry`: sends the conversation again; it takes no arguments.
fn retry(In(args): In<CommandArgs>, mut commands: Commands, mut notices: MessageWriter<Notice>) {
    if args.args.is_empty() {
        commands.trigger(Retry { entity: args.agent });
    } else {
        let why = format!("/retry takes no arguments, not `{}`.", args.args);
        notices.write(Notice::error(args.agent, why));
    }
}

/// `/help`: every command with its help.
fn help(
    In(args): In<CommandArgs>,
    commands: Query<(&Name, &SlashCommand)>,
    mut notices: MessageWriter<Notice>,
) {
    let mut lines: Vec<String> = commands
        .iter()
        .map(|(name, command)| format!("{:<11} {}", name.as_str(), command.help))
        .collect();
    lines.sort();
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

fn quit(In(_): In<CommandArgs>, mut commands: Commands) {
    commands.trigger(Exit(ExitReason::Quit));
}
