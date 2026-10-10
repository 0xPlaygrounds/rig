//! The agent tree in each agent's status line, pointing at `/agents`: a
//! spawned agent's name, its subagents at work, and the other agents at
//! work. An agent is at work while it has a turn.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use rig_ecs::agent::{ActiveTurn, Agent, Spawned, SpawnedBy};
use rig_harness::front::{Side, StatusItem, StatusItems, StatusSystems, Tone};

/// Where a spawned agent's name is: before its model.
const TITLE: StatusItem = StatusItem::at(Side::Left, 20, 14);
/// Where its subagents at work are counted: after its status.
const SUBAGENTS: StatusItem = StatusItem::at(Side::Left, 60, 15);
/// Where the other agents at work are counted.
const OTHERS: StatusItem = StatusItem::at(Side::Left, 70, 13);

pub(super) fn add(app: &mut App) {
    app.add_systems(
        PostUpdate,
        show.in_set(StatusSystems)
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
