use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use serde_json::json;

use super::*;
use rig_harness_test_support::{
    MemoryStore, ToolOutputs, app_on, calls, log_tool_outputs, reply, run_until, scripted_models,
    start_call,
};

/// The kernel and the board on `store`, with the scripted models.
fn app(store: &MemoryStore, models: Models) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(models)
        .add_plugins((AgentPlugin, SessionLogPlugin, BlackboardPlugin))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    (app, wakes)
}

/// Sends `text` to `agent` and runs until its model was asked `requests`
/// times in all and the agent is idle.
fn turn(
    app: &mut App,
    wakes: &Receiver<()>,
    (agent, model): (Entity, &MockCompletionModel),
    text: &str,
    requests: usize,
) {
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        model.request_count() == requests && world.get::<ActiveTurn>(agent).is_none()
    });
}

fn requests(model: &MockCompletionModel) -> Vec<String> {
    let requests = model.requests();
    let sent = requests.iter().map(|request| format!("{request:?}"));
    sent.collect()
}

fn output(app: &App, id: &str) -> String {
    let outputs = &app.world().resource::<ToolOutputs>().0;
    outputs.get(id).cloned().unwrap_or_default()
}

#[test]
fn claims_are_refused_posts_reach_the_other_subscribers_and_the_board_is_restored() {
    let a = MockCompletionModel::from_stream_turns([
        calls(
            "a1",
            BOARD_POST,
            json!({ "topic": "bugs", "text": "null deref in x.rs", "claim": "x.rs" }),
        ),
        reply("posted"),
        reply("seen"),
    ]);
    let b = MockCompletionModel::from_stream_turns([
        calls("b1", BOARD_READ, json!({ "topic": "bugs" })),
        reply("nothing yet"),
        calls(
            "b2",
            BOARD_POST,
            json!({ "topic": "bugs", "text": "me too", "claim": "x.rs" }),
        ),
        reply("left it to A"),
    ]);
    let models = || scripted_models([("board-a/m", &a), ("board-b/m", &b)]);
    let store = MemoryStore::default();
    let (mut app, wakes) = app(&store, models());
    let world = app.world_mut();
    let agent_a = world
        .spawn((Agent, Name::new("A"), ModelChoice("board-a/m".to_owned())))
        .id();
    let agent_b = world
        .spawn((Agent, Name::new("B"), ModelChoice("board-b/m".to_owned())))
        .id();
    app.finish();
    app.update();

    turn(&mut app, &wakes, (agent_b, &b), "Read the bugs.", 2);
    assert!(output(&app, "b1").contains("No posts in `bugs` yet."));
    turn(&mut app, &wakes, (agent_a, &a), "Post your bug.", 2);
    assert!(output(&app, "a1").contains("Claimed `x.rs`."));
    for _ in 0..5 {
        app.update();
    }
    assert_eq!(b.request_count(), 2, "a note wakes no idle agent");

    turn(&mut app, &wakes, (agent_b, &b), "Post yours.", 4);
    let refused = output(&app, "b2");
    assert!(
        refused.contains("Claim of `x.rs` refused: A (agent"),
        "{refused}"
    );
    let b_sent = requests(&b);
    assert!(
        b_sent
            .get(2)
            .is_some_and(|next| next.contains("New board post: #1 [bugs] A (agent")
                && next.contains("null deref in x.rs")),
        "{b_sent:#?}"
    );

    turn(&mut app, &wakes, (agent_a, &a), "Anything new?", 3);
    let a_sent = requests(&a);
    assert!(
        a_sent
            .iter()
            .all(|sent| !sent.contains("New board post: #1")),
        "A never gets its own post: {a_sent:#?}"
    );
    assert!(
        a_sent
            .get(2)
            .is_some_and(|last| last.contains("New board post: #2 [bugs] B (agent")),
        "{a_sent:#?}"
    );
    drop(app);

    let (mut restored, wakes) = self::app(&store, models());
    restored.finish();
    restored.update();
    let world = restored.world_mut();
    let mut agents = world.query_filtered::<Entity, With<Agent>>();
    let agent = agents.iter(world).next().unwrap_or(Entity::PLACEHOLDER);
    let read = start_call(world, agent, (BOARD_READ, "r1"), json!({}));
    assert!(read.is_some());
    run_until(&mut restored, &wakes, |world| {
        world.resource::<ToolOutputs>().0.contains_key("r1")
    });
    let listed = output(&restored, "r1");
    assert!(listed.contains("#1 [bugs] A (agent"), "{listed}");
    assert!(
        listed.contains("null deref in x.rs (claims `x.rs`)"),
        "{listed}"
    );
    assert!(listed.contains("#2 [bugs] B (agent"), "{listed}");
    assert!(listed.contains("me too"), "{listed}");
    assert!(listed.contains("- `x.rs`: A (agent"), "{listed}");
}
