use rig_ecs::agent::Agent;
use rig_ecs::commands::RunCommand;
use rig_ecs::prelude::*;

use crate::{AgentColumn, BasicCommandsPlugin, ColumnOf};

#[test]
fn agents_shows_the_columns_other_plugins_add_to_an_agents_line() {
    let mut app = App::new();
    app.add_plugins((AgentPlugin, BasicCommandsPlugin));
    let world = app.world_mut();
    let agent = world.spawn((Agent, Name::new("main"))).id();
    world.spawn((AgentColumn::new("spent", "$0.10"), ColumnOf(agent)));
    world.spawn((AgentColumn::new("queued", "2 queued"), ColumnOf(agent)));
    world.trigger(RunCommand {
        entity: agent,
        line: "agents".to_owned(),
    });
    world.flush();
    let mut picks = world.resource_mut::<Messages<rig_tui::PickRequest>>();
    let labels: Vec<String> = picks
        .drain()
        .flat_map(|pick| pick.items)
        .map(|item| item.label)
        .collect();
    assert_eq!(labels, ["main · no model · idle · 2 queued · $0.10"]);
}
