use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use serde_json::json;

use super::*;
use rig_harness_test_support::{
    Collected, Ended, ToolOutputs, app_on, calls, collect, connect, first_agent, log_ended,
    log_tool_outputs, reply, run_until,
};

/// What the test `read` tool answers.
const CONTENTS: &str = "the contents of a.txt";

/// A `read` tool counting its runs.
struct Read(Arc<AtomicUsize>);

impl Tool for Read {
    const NAME: &'static str = "read";
    type Args = Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Read a file.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {"path": {"type": "string"}}})
    }

    async fn call(&self, _: &mut ToolContext, _args: Value) -> Result<String, ToolExecutionError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(CONTENTS.to_owned())
    }
}

/// The kernel, the journal and the loop breaker on `store`, with a `read`
/// tool counting into `runs`, after the first frame restored the session,
/// with its loop's wakes.
fn start(store: &MemoryStore, runs: &Arc<AtomicUsize>) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, SessionLogPlugin, LoopBreakerPlugin))
        .add_tool(Read(runs.clone()))
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    (app, wakes)
}

#[test]
fn repeated_calls_are_refused_then_stop_the_turn() {
    let store = MemoryStore::default();
    let runs = Arc::new(AtomicUsize::new(0));
    let (mut app, wakes) = start(&store, &runs);
    // Five identical calls, each in its own reply; the answer is never
    // asked for.
    let read = |id| calls(id, "read", json!({"path": "a.txt"}));
    let model = MockCompletionModel::from_stream_turns([
        read("1"),
        read("2"),
        read("3"),
        read("4"),
        read("5"),
        reply("done"),
    ]);
    let world = app.world_mut();
    let agent = first_agent(world);
    let connection = connect(&model);
    assert!(
        agent.is_some() && connection.is_some(),
        "an agent and the catalog's model"
    );
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    world.entity_mut(agent).insert(connection);
    world.trigger(Deliver::new(agent, "Read a.txt.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    app.update();

    let sent: Vec<String> = model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect();
    assert_eq!(sent.len(), 5, "five requests, no sixth: {sent:#?}");
    let [_, second, third, fourth, fifth] = sent.as_slice() else {
        return;
    };
    // Calls 1 and 2 ran, and no other.
    assert_eq!(runs.load(Ordering::SeqCst), 2);
    assert!(second.contains(CONTENTS), "{second}");
    assert_eq!(third.matches(CONTENTS).count(), 2, "{third}");
    // Call 3's result is the refusal, and request 4 carries it.
    let refused = "You already called `read` with these arguments twice with the same result; \
                   change approach.";
    let outputs = &app.world().resource::<ToolOutputs>().0;
    assert_eq!(
        outputs.get("3").map(String::as_str),
        Some(refused),
        "{outputs:?}"
    );
    assert!(fourth.contains(refused), "{fourth}");
    assert!(fifth.contains("these arguments 3 times"), "{fifth}");

    let ended = &app.world().resource::<Ended>().0;
    let why = "the agent was repeating `read a.txt`";
    assert!(
        matches!(ended.as_slice(), [TurnOutcome::Stopped(stop)]
            if matches!(stop.cause(), StopCause::Plugin { why: stopped, .. } if stopped == why)),
        "{ended:?}"
    );
    let notices = &app.world().resource::<Collected<Notice>>().0;
    assert!(
        notices
            .iter()
            .any(|notice| notice.text == format!("stopped: {why}")),
        "{notices:?}"
    );
    drop(app);

    // A restart leaves the stopped turn stopped: nothing is sent again.
    let (mut app, _) = start(&store, &runs);
    for _ in 0..5 {
        app.update();
    }
    let world = app.world_mut();
    let busy = world.query::<&TurnOf>().iter(world).count();
    let ended = &world.resource::<Ended>().0;
    assert!(busy == 0 && ended.is_empty(), "{busy} turns, {ended:?}");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(LoopBreakerPlugin);
}
