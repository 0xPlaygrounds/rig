//! The terminal view's part: the agent tree in each agent's status line: a
//! spawned agent's name, its subagents at work, and the other agents at
//! work (`AgentsPlugin`'s `/agents` shows one). An agent is at work while it
//! has a turn.

use rig_harness::prelude::*;

use rig_ecs::agent::{ActiveTurn, Agent, Spawned, SpawnedBy};
use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};

/// Where a spawned agent's name is: before its model.
const TITLE: StatusItem = StatusItem::new("title", Side::Left, 20, 14);
/// Where its subagents at work are counted: after its status.
const SUBAGENTS: StatusItem = StatusItem::new("subagents", Side::Left, 60, 15);
/// Where the other agents at work are counted.
const OTHERS: StatusItem = StatusItem::new("others", Side::Left, 70, 13);

pub(super) fn add(app: &mut App) {
    app.add_systems(
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
