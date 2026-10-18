use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use serde_json::json;

use super::*;
use rig_harness_test_support::{
    Ended, app_on, calls, connect, first_agent, log_ended, reply, run_until,
};

/// The kernel and the todo plugin on `store`, after the first frame
/// restored the session, with its first agent.
fn start(store: &MemoryStore) -> (App, Receiver<()>, Option<Entity>) {
    let (mut app, wakes) = app_on(store);
    app.add_plugins((AgentPlugin, SessionLogPlugin, TodoPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut());
    (app, wakes, agent)
}

#[test]
fn an_answer_with_open_items_is_nudged_once_and_the_list_is_restored() {
    let store = MemoryStore::default();
    let (mut app, wakes, agent) = start(&store);
    let model = MockCompletionModel::from_stream_turns([
        calls(
            "1",
            TODO_TOOL,
            json!({ "action": "set", "items": ["a", "b"] }),
        ),
        calls("2", TODO_TOOL, json!({ "action": "check", "id": 1 })),
        reply("done"),
        calls("3", TODO_TOOL, json!({ "action": "check", "id": 2 })),
        reply("all done"),
    ]);
    let pair = agent.zip(connect(&model));
    assert!(pair.is_some(), "an agent and the catalog's model");
    let Some((agent, connection)) = pair else {
        return;
    };
    let world = app.world_mut();
    world.entity_mut(agent).insert(connection);
    world.trigger(Deliver::new(agent, "Do a, then b.", DeliveryMode::Steer));
    let idle = |world: &World| {
        !world.resource::<Ended>().0.is_empty() && world.get::<ActiveTurn>(agent).is_none()
    };
    run_until(&mut app, &wakes, idle);
    // A few more frames: nothing else is sent.
    for _ in 0..5 {
        app.update();
    }

    let sent: Vec<String> = model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect();
    assert_eq!(sent.len(), 5, "{sent:#?}");
    let nudge = "You have 1 open todo items: 2. b; continue, or explain why you stopped.";
    assert!(
        sent.get(3).is_some_and(|fourth| fourth.contains(nudge)),
        "{sent:#?}"
    );
    assert!(
        sent.get(4)
            .is_some_and(|last| last.matches("open todo items").count() == 1),
        "one nudge: {sent:#?}"
    );
    let answers: Vec<Option<String>> = app
        .world()
        .resource::<Ended>()
        .0
        .iter()
        .map(|outcome| match outcome {
            TurnOutcome::Answered(message) => message.final_answer(),
            _ => None,
        })
        .collect();
    // The nudge carried the turn on past "done": one turn, one answer.
    assert_eq!(answers, [Some("all done".to_owned())]);
    drop(app);

    let (mut restarted, _wakes, agent) = start(&store);
    let world = restarted.world_mut();
    let todos = agent.and_then(|agent| world.get::<Todos>(agent));
    assert_eq!(
        todos.map(Todos::render).as_deref(),
        Some("[x] 1. a\n[x] 2. b")
    );
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(TodoPlugin);
}
