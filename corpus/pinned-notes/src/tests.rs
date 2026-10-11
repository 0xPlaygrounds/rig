use std::sync::mpsc::Receiver;

use bevy_ecs::message::Messages;
use rig_cassette::journal::MemoryStore;
use rig_compaction::CompactionPlugin;
use rig_core::catalog::{Catalog, Connector};
use rig_core::test_utils::MockCompletionModel;
use rig_subagents::SubagentsPlugin;

use super::*;
use rig_harness_test_support::{Ended, app_on, connect, log_ended, reply, run_until, start_call};

/// The kernel, compaction, subagents and pinned notes on `store`, after
/// the first frame restored the session, with its loop's wakes and its
/// agent, answered by `model` when given.
fn start(
    store: &MemoryStore,
    model: Option<&MockCompletionModel>,
) -> Option<(App, Receiver<()>, Entity)> {
    let (mut app, wakes) = app_on(store);
    app.set_error_handler(bevy_ecs::error::warn)
        .insert_resource(Models(Connector::new(Catalog::default())))
        .add_plugins((
            AgentPlugin,
            JournalPlugin,
            CompactionPlugin,
            SubagentsPlugin,
            PinnedNotesPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    if let Some(connection) = model.and_then(connect) {
        let choice = ModelChoice(connection.spec.reference());
        app.world_mut().spawn((Agent, connection, choice));
    }
    app.finish();
    app.update();
    let world = app.world_mut();
    let mut roots = world.query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>();
    let agent = roots.iter(world).next()?;
    Some((app, wakes, agent))
}

/// Runs `/line` for `agent`.
fn run(app: &mut App, agent: Entity, line: &str) {
    let line = line.to_owned();
    app.world_mut().trigger(RunCommand {
        entity: agent,
        line,
    });
}

/// Asks `agent` something and runs frames until its turn ended.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
}

#[test]
fn pins_go_with_every_request_survive_compaction_and_come_back() {
    let model = MockCompletionModel::from_stream_turns([
        reply("First answer."),
        reply("## Goal\nAnswer questions."),
        reply("Second answer."),
    ]);
    let store = MemoryStore::default();
    let started = start(&store, Some(&model));
    assert!(started.is_some(), "the session has one agent");
    let Some((mut app, wakes, agent)) = started else {
        return;
    };

    run(&mut app, agent, "pin Use approach B");
    ask(&mut app, &wakes, agent, "Which approach?");
    run(&mut app, agent, "compact");
    run_until(&mut app, &wakes, |world| {
        world.get::<Condensed>(agent).is_some()
    });
    ask(&mut app, &wakes, agent, "And now?");

    let sent: Vec<String> = model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect();
    assert_eq!(
        sent.len(),
        3,
        "a request, a summary and a request: {sent:?}"
    );
    let [first, summary, after] = sent.as_slice() else {
        return;
    };
    assert!(
        first.contains("Pinned notes") && first.contains("1. Use approach B"),
        "{first}"
    );
    assert!(
        summary.contains("do not repeat them in the summary:\\n1. Use approach B"),
        "{summary}"
    );
    assert!(after.contains("## Goal"), "after compaction: {after}");
    assert!(after.contains("1. Use approach B"), "{after}");
    drop(app);

    // A restart restores the agent with its pins.
    let restarted = start(&store, None);
    assert!(restarted.is_some(), "the agent was restored");
    let Some((mut app, _wakes, agent)) = restarted else {
        return;
    };
    run(&mut app, agent, "pins");
    app.update();
    let notices = app.world().resource::<Messages<Notice>>();
    let mut cursor = notices.get_cursor();
    let said: Vec<&str> = cursor
        .read(notices)
        .map(|notice| notice.text.as_str())
        .collect();
    assert!(
        said.contains(&"Pinned notes (1):\n1. Use approach B"),
        "{said:?}"
    );
}

#[test]
fn only_a_task_with_inherit_pins_passes_the_pins_on() {
    let model = MockCompletionModel::from_stream_turns([reply("Looked.")]);
    let started = start(&MemoryStore::default(), Some(&model));
    assert!(started.is_some(), "the session has one agent");
    let Some((mut app, _wakes, parent)) = started else {
        return;
    };
    run(&mut app, parent, "pin Keep the API stable");
    let task = |inherit: bool| serde_json::json!({ "description": "Look", "prompt": "Look around.", "inherit_pins": inherit });
    let world = app.world_mut();
    start_call(world, parent, (TASK, "t1"), task(true));
    start_call(world, parent, (TASK, "t2"), task(false));
    for _ in 0..8 {
        app.world_mut().run_schedule(AgentPass);
    }

    let world = app.world();
    let children: Vec<Entity> = world
        .get::<Spawned>(parent)
        .map(|spawned| spawned.iter().collect())
        .unwrap_or_default();
    let pins: Vec<Option<&Pins>> = children
        .iter()
        .map(|&child| world.get::<Pins>(child))
        .collect();
    let kept = Pins(vec!["Keep the API stable".to_owned()]);
    assert_eq!(
        pins,
        [Some(&kept), Some(&Pins::default())],
        "the first task inherits, the second does not"
    );
}
