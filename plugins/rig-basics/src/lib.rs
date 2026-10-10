//! The basic slash commands: `/help`, `/retry`, `/agents` and `/quit`,
//! with the agent tree in each agent's status line
//! ([`BasicCommandsPlugin`]), and the project context in the system prompt
//! ([`ProjectContextPlugin`]): `AGENTS.md`, the environment and `/context`.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::query::QueryData;

use rig_ecs::agent::{ActiveTurn, Agent, AgentId, Notice, Retry, Spawned, SpawnedBy};
use rig_ecs::commands::{AppCommandsExt, CommandArgs, SlashCommand};
use rig_ecs::model::ModelChoice;
use rig_harness::front::{Focus, PickItem, PickRequest};

pub mod project_context;
mod status;

pub use project_context::ProjectContextPlugin;

/// Registers the basic commands, and shows the agent tree that `/agents`
/// picks from in each agent's status line.
#[derive(Default)]
pub struct BasicCommandsPlugin;

impl Plugin for BasicCommandsPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "retry",
            "Send the conversation again after a failed model call",
            retry,
        )
        .add_command(
            "agents",
            "Pick an agent or subagent to show; /agents <id> shows it",
            agents,
        )
        .add_command("help", "List the commands", help)
        .add_command("quit", "Quit; the session stays for /resume", quit);
        status::add(app);
    }
}

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

fn quit(In(_): In<CommandArgs>, mut exit: MessageWriter<AppExit>) {
    exit.write(AppExit::Success);
}

/// An agent as `/agents` lists it.
#[derive(QueryData)]
struct Listed {
    entity: Entity,
    id: &'static AgentId,
    name: &'static Name,
    model: Option<&'static ModelChoice>,
    busy: Has<ActiveTurn>,
}

/// `/agents`: picks an agent to show, each listed under the agent that
/// spawned it, or shows the one whose id starts with the argument.
fn agents(
    In(args): In<CommandArgs>,
    agents: Query<Listed, With<Agent>>,
    spawned: Query<&Spawned>,
    parents: Query<&SpawnedBy>,
    mut commands: Commands,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    if !args.args.is_empty() {
        match agents
            .iter()
            .find(|agent| agent.id.0.starts_with(&args.args))
        {
            Some(agent) => commands.trigger(Focus {
                entity: agent.entity,
            }),
            None => {
                let why = format!("No agent's id starts with `{}`.", args.args);
                notices.write(Notice::error(args.agent, why));
            }
        }
        return;
    }
    let mut roots: Vec<_> = agents
        .iter()
        .filter(|agent| !parents.contains(agent.entity))
        .collect();
    roots.sort_by(|a, b| a.id.0.cmp(&b.id.0));
    let listed = roots.iter().flat_map(|root| {
        std::iter::once(root.entity).chain(spawned.iter_descendants_depth_first(root.entity))
    });
    let (mut items, mut selected) = (Vec::new(), 0);
    for agent in listed.filter_map(|agent| agents.get(agent).ok()) {
        if agent.entity == args.agent {
            selected = items.len();
        }
        let depth = parents.iter_ancestors(agent.entity).count();
        let model = agent.model.map_or("no model", |model| model.0.as_str());
        let state = if agent.busy { "working" } else { "idle" };
        items.push(PickItem {
            label: format!("{}{} · {model} · {state}", "  ".repeat(depth), agent.name),
            command: format!("agents {}", agent.id.0),
        });
    }
    picks.write(PickRequest {
        agent: args.agent,
        title: "Show an agent".to_owned(),
        items,
        selected,
    });
}
