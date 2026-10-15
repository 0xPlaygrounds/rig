//! The terminal view's part: `/agents`, which shows an agent picked from
//! the agent tree, and the tree in each agent's status line: a spawned
//! agent's name, its subagents at work, and the other agents at work. An
//! agent is at work while it has a turn.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::query::QueryData;

use rig_ecs::agent::{ActiveTurn, Agent, AgentId, Notice, Spawned, SpawnedBy};
use rig_ecs::model::ModelChoice;
use rig_harness::commands::{AppCommandsExt, CommandArgs};
use rig_harness::pick::{PickItem, Picker};
use rig_tui::{ShowAgent, Side, StatusItem, StatusItems, StatusSystems, Tone};

use super::{AgentColumn, AgentColumns};

/// Where a spawned agent's name is: before its model.
const TITLE: StatusItem = StatusItem::new("title", Side::Left, 20, 14);
/// Where its subagents at work are counted: after its status.
const SUBAGENTS: StatusItem = StatusItem::new("subagents", Side::Left, 60, 15);
/// Where the other agents at work are counted.
const OTHERS: StatusItem = StatusItem::new("others", Side::Left, 70, 13);

pub(super) fn add(app: &mut App) {
    app.add_command(
        "agents",
        "Pick an agent or subagent to show; /agents <id> shows it",
        agents,
    )
    .add_systems(
        PostUpdate,
        show.in_set(StatusSystems::Show)
            .run_if(any_component_removed::<ActiveTurn>.or_eager(changed)),
    );
}

/// Whether an agent started a turn, was named or was spawned.
fn changed(
    agents: Query<(), Or<(Changed<ActiveTurn>, Changed<Name>, Added<StatusItems>)>>,
) -> bool {
    !agents.is_empty()
}

fn show(
    busy: Query<Entity, (With<Agent>, With<ActiveTurn>)>,
    mut agents: Query<
        (
            Entity,
            &Name,
            Has<SpawnedBy>,
            Option<&Spawned>,
            &mut StatusItems,
        ),
        With<Agent>,
    >,
) {
    for (agent, name, spawned_by, spawned, mut items) in &mut agents {
        let children: Vec<Entity> = spawned
            .map(|spawned| spawned.iter().collect())
            .unwrap_or_default();
        let mine = busy.iter().filter(|other| children.contains(other)).count();
        let others = busy
            .iter()
            .filter(|other| *other != agent && !children.contains(other));
        let others = others.count();
        let title = if spawned_by {
            format!("⤷ {name}")
        } else {
            String::new()
        };
        let mine = match mine {
            0 => String::new(),
            1 => "1 subagent working (/agents)".to_owned(),
            count => format!("{count} subagents working (/agents)"),
        };
        let others = match others {
            0 => String::new(),
            count => format!("+{count} other agents working (/agents)"),
        };
        items.show(TITLE.says(title, Tone::Magenta));
        items.show(SUBAGENTS.says(mine, Tone::Magenta));
        items.show(OTHERS.says(others, Tone::Magenta));
    }
}

/// An agent as `/agents` lists it.
#[derive(QueryData)]
struct Listed {
    entity: Entity,
    id: &'static AgentId,
    name: &'static Name,
    model: Option<&'static ModelChoice>,
    busy: Has<ActiveTurn>,
    columns: Option<&'static AgentColumns>,
}

/// `/agents`: picks an agent to show, each listed under the agent that
/// spawned it with its [`AgentColumn`]s, or shows the one whose id starts
/// with the argument.
fn agents(
    In(args): In<CommandArgs>,
    agents: Query<Listed, With<Agent>>,
    columns: Query<&AgentColumn>,
    spawned: Query<&Spawned>,
    parents: Query<&SpawnedBy>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if !args.args.is_empty() {
        match agents
            .iter()
            .find(|agent| agent.id.0.starts_with(&args.args))
        {
            Some(agent) => commands.trigger(ShowAgent {
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
        let mut own: Vec<&AgentColumn> = agent
            .columns
            .into_iter()
            .flat_map(|of| of.iter())
            .filter_map(|column| columns.get(column).ok())
            .filter(|column| !column.text.is_empty())
            .collect();
        own.sort_by(|a, b| a.tag.cmp(&b.tag));
        let own: String = own
            .iter()
            .map(|column| format!(" · {}", column.text))
            .collect();
        items.push(PickItem {
            label: format!(
                "{}{} · {model} · {state}{own}",
                "  ".repeat(depth),
                agent.name
            ),
            command: format!("agents {}", agent.id.0),
            detail: None,
            checked: false,
        });
    }
    commands.spawn(Picker {
        items,
        selected,
        ..Picker::new("Show an agent").for_agent(args.agent)
    });
}

#[cfg(test)]
mod tests;
