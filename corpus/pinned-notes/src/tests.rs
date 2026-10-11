use std::sync::mpsc::Receiver;

use bevy_ecs::message::Messages;
use rig_cassette::journal::MemoryStore;
use rig_compaction::CompactionPlugin;
use rig_core::test_utils::MockCompletionModel;

use super::*;
use rig_harness_test_support::{app_on, connect, reply, run_until};

/// How many turns ended.
#[derive(Resource, Default)]
struct Ended(usize);

/// The kernel, compaction and pinned notes on `store`, after the first
/// frame restored the session, with its loop's wakes and its agent.
fn start(store: &MemoryStore) -> Option<(App, Receiver<()>, Entity)> {
    let (mut app, wakes) = app_on(store);
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((
            AgentPlugin,
            JournalPlugin,
            CompactionPlugin,
            PinnedNotesPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(|_: On<TurnEnded>, mut ended: ResMut<Ended>| ended.0 += 1);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .single(world)
        .ok()?;
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
    let ended = app.world().resource::<Ended>().0;
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
    run_until(app, wakes, |world| world.resource::<Ended>().0 > ended);
}

#[test]
fn pins_go_with_every_request_survive_compaction_and_come_back() {
    let store = MemoryStore::default();
    let started = start(&store);
    assert!(started.is_some(), "the session has one agent");
    let Some((mut app, wakes, agent)) = started else {
        return;
    };
    let model = MockCompletionModel::from_stream_turns([
        reply("First answer."),
        reply("## Goal\nAnswer questions."),
        reply("Second answer."),
    ]);
    let connection = connect(&model);
    assert!(connection.is_some(), "the catalog lists the model");
    let Some(connection) = connection else {
        return;
    };
    app.world_mut().entity_mut(agent).insert(connection);

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
    assert!(
        after.contains("## Goal"),
        "the request after compaction: {after}"
    );
    assert!(after.contains("1. Use approach B"), "{after}");
    drop(app);

    // A restart restores the agent with its pins.
    let restarted = start(&store);
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
