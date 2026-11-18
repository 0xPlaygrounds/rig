use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_ecs::AgentPlugin;
use rig_ecs::fork::{Ancestor, ForkedFrom};
use rig_ecs::prelude::{AgentId, MessageId, TurnOf};
use rig_ecs::session_log::SessionLogPlugin;
use rig_ecs::spawn::spawn_root;
use rig_harness::commands::RunCommand;
use rig_harness::pick::{PickPlugin, Picker};
use rig_harness::prelude::ShownAgent;
use rig_harness::settings::Settings;
use rig_harness_test_support::{
    Ended, MemoryStore, app_on, log_ended, press, reply, scripted_models, tui_view,
};
use rig_tui::{ShowAgent, TuiTestScreen};

use super::*;

#[test]
fn agents_shows_the_columns_other_plugins_add_to_an_agents_line() {
    let mut app = App::new();
    app.add_plugins((AgentPlugin, AgentPickerPlugin));
    let world = app.world_mut();
    let agent = spawn_root(world, Name::new("main")).id();
    let short = world.get::<AgentId>(agent).map(|id| id.short().to_owned());
    world.spawn((AgentColumn::new("spent", "$0.10"), ColumnOf(agent)));
    world.spawn((AgentColumn::new("queued", "2 queued"), ColumnOf(agent)));
    world.trigger(RunCommand::new(agent, "agents".to_owned()));
    world.flush();
    let mut pickers = world.query::<&Picker>();
    let labels: Vec<&str> = pickers
        .iter(world)
        .flat_map(|picker| &picker.items)
        .map(|item| item.label.as_str())
        .collect();
    let short = short.unwrap_or_default();
    assert_eq!(
        labels,
        [format!(
            "main {short} · no model · idle · current · 2 queued · $0.10"
        )]
    );
}

/// The lines of an open `/agents` follow the agents: which is current,
/// which a fork came from, and whether one works.
#[test]
fn agents_marks_the_current_agent_and_forks_and_follows_their_state() {
    let mut app = App::new();
    app.add_plugins((AgentPlugin, AgentPickerPlugin));
    let first = AgentId("00000000-0000-4000-8000-000000000000".to_owned());
    let source = spawn_root(app.world_mut(), (Name::new("agent"), first)).id();
    let id = |app: &App, agent: Entity| {
        let id = app.world().get::<AgentId>(agent);
        id.map(|id| (id.0.clone(), id.short().to_owned()))
            .unwrap_or_default()
    };
    let (source_id, source_short) = id(&app, source);
    let (agent, since) = (source_id, MessageId::default());
    let from = ForkedFrom(vec![Ancestor { agent, since }]);
    let fork = spawn_root(app.world_mut(), from).id();
    let (_, fork_short) = id(&app, fork);
    app.update();
    app.world_mut()
        .trigger(RunCommand::new(source, "agents".to_owned()));
    let labels = |app: &mut App| {
        app.update();
        let world = app.world_mut();
        let mut pickers = world.query::<&Picker>();
        let mut labels: Vec<String> = (pickers.iter(world))
            .flat_map(|picker| &picker.items)
            .map(|item| item.label.clone())
            .collect();
        labels.sort();
        labels
    };
    let line =
        |short: &str, more: &str, state: &str| format!("agent {short} · no model · {state}{more}");
    let lines = |more: [&str; 2], state: &str| {
        let fork_of = format!("{} · fork of {source_short}", more[1]);
        let mut lines = vec![
            line(&source_short, more[0], "idle"),
            line(&fork_short, &fork_of, state),
        ];
        lines.sort();
        lines
    };
    assert_eq!(labels(&mut app), lines([" · current", ""], "idle"));

    app.world_mut().resource_mut::<ShownAgent>().0 = Some(fork);
    let turn = app.world_mut().spawn(TurnOf(fork)).id();
    assert_eq!(labels(&mut app), lines(["", " · current"], "working"));
    app.world_mut().despawn(turn);
    assert_eq!(labels(&mut app), lines(["", " · current"], "idle"));
}

/// When a line above the selected one leaves an open `/agents`, such as a
/// helper's that retired, Enter shows the agent that is selected on screen.
#[test]
fn enter_in_agents_shows_the_agent_selected_after_a_line_above_left() {
    let model = MockCompletionModel::from_stream_turns([reply("unused")]);
    let (mut app, _wakes) = on(&MemoryStore::default(), &model);
    app.insert_resource(TuiTestScreen::new(100, 30));
    tui_view(&mut app);
    app.add_plugins((PickPlugin, AgentPickerPlugin));
    let ids = [1, 2, 3, 4].map(|n| AgentId(format!("0000000{n}-0000-4000-8000-000000000000")));
    let agents = ids.map(|id| spawn_root(app.world_mut(), (Name::new("agent"), id)).id());
    app.finish();
    app.cleanup();
    app.update();
    let [first, gone, selected, _] = agents;
    app.world_mut().trigger(ShowAgent { entity: first });
    app.world_mut()
        .trigger(RunCommand::new(first, "agents".to_owned()));
    app.update();
    press(&mut app, "down down");
    app.world_mut().despawn(gone);
    app.update();
    press(&mut app, "enter");
    assert_eq!(shown(&app), Some(selected));
}

/// The agent the user looks at.
fn shown(app: &App) -> Option<Entity> {
    app.world().resource::<ShownAgent>().0
}

/// An app on `store` with the agents plugin, keeping how turns end.
fn on(store: &MemoryStore, model: &MockCompletionModel) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(Settings::default())
        .insert_resource(scripted_models([("test/model", model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    (app, wakes)
}
