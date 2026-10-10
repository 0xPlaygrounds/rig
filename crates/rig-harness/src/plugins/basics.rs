//! The basic slash commands: `/help`, `/retry`, `/agents` and `/quit`.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use crate::front::{Focus, PickItem, PickRequest};
use rig_ecs::agent::{ActiveTurn, Agent, AgentId, Notice, Retry, Spawned, SpawnedBy};
use rig_ecs::commands::{AppCommandsExt, CommandArgs, SlashCommand};
use rig_ecs::model::ModelChoice;

/// Registers the basic commands.
#[derive(Default)]
pub struct BasicCommandsPlugin;

impl Plugin for BasicCommandsPlugin {
    fn build(&self, app: &mut App) {
        app.add_command_event::<Retry>(
            "retry",
            "Send the conversation again after a failed model call",
        )
        .add_command(
            "agents",
            "Pick an agent or subagent to show; /agents <id> shows it",
            agents,
        )
        .add_command("help", "List the commands", help)
        .add_command("quit", "Quit; the session stays for /resume", quit);
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

/// `/agents`: picks an agent to show, each listed under the agent that
/// spawned it, or shows the one whose id starts with the argument.
fn agents(
    In(args): In<CommandArgs>,
    agents: Query<
        (
            Entity,
            &AgentId,
            &Name,
            Option<&ModelChoice>,
            Has<ActiveTurn>,
        ),
        With<Agent>,
    >,
    (spawned, parents): (Query<&Spawned>, Query<&SpawnedBy>),
    mut commands: Commands,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    if !args.args.is_empty() {
        match agents
            .iter()
            .find(|(_, id, ..)| id.0.starts_with(&args.args))
        {
            Some((entity, ..)) => commands.trigger(Focus { entity }),
            None => {
                let why = format!("No agent's id starts with `{}`.", args.args);
                notices.write(Notice::error(args.agent, why));
            }
        }
        return;
    }
    let mut roots: Vec<(&AgentId, Entity)> = agents
        .iter()
        .filter(|(agent, ..)| !parents.contains(*agent))
        .map(|(agent, id, ..)| (id, agent))
        .collect();
    roots.sort_by(|a, b| a.0.0.cmp(&b.0.0));
    let listed = roots.iter().flat_map(|&(_, root)| {
        std::iter::once(root).chain(spawned.iter_descendants_depth_first(root))
    });
    let (mut items, mut selected) = (Vec::new(), 0);
    for (agent, id, name, model, busy) in listed.filter_map(|agent| agents.get(agent).ok()) {
        if agent == args.agent {
            selected = items.len();
        }
        let depth = parents.iter_ancestors(agent).count();
        let model = model.map_or("no model", |model| model.0.as_str());
        let state = if busy { "working" } else { "idle" };
        items.push(PickItem {
            label: format!("{}{name} · {model} · {state}", "  ".repeat(depth)),
            command: format!("agents {}", id.0),
        });
    }
    picks.write(PickRequest {
        agent: args.agent,
        title: "Show an agent".to_owned(),
        items,
        selected,
    });
}
