use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::MockCompletionModel;
use serde_json::json;

use super::*;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};

/// What the test `read` tool answers.
const CONTENTS: &str = "the contents of a.txt";

/// A `read` tool counting its runs.
struct Read(Arc<AtomicUsize>);

impl PortableTool for Read {
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

    async fn call(&self, _args: Value) -> Result<String, ToolExecutionError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(CONTENTS.to_owned())
    }
}

/// How each turn ended, and every notice.
#[derive(Resource, Default)]
struct Seen {
    ended: Vec<TurnOutcome>,
    notices: Vec<String>,
}

fn record_notices(mut notices: MessageReader<Notice>, mut seen: ResMut<Seen>) {
    seen.notices
        .extend(notices.read().map(|notice| notice.text.clone()));
}

#[test]
fn repeated_calls_are_refused_then_end_the_turn() {
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    let runs = Arc::new(AtomicUsize::new(0));
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, JournalPlugin, LoopBreakerPlugin))
        .add_tool(Read(runs.clone()))
        .init_resource::<Seen>()
        .add_systems(Update, record_notices)
        .add_observer(|ended: On<TurnEnded>, mut seen: ResMut<Seen>| {
            seen.ended.push(ended.outcome.clone());
        });
    app.finish();
    app.update();
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
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
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
        !world.resource::<Seen>().ended.is_empty()
    });
    // The notice is read in the frame after the turn ended.
    app.update();

    let sent: Vec<String> = model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect();
    assert_eq!(sent.len(), 5, "no sixth request: {sent:#?}");
    let refused = "You already called `read` with these arguments twice with the same result";
    let [_, second, third, fourth, fifth] = sent.as_slice() else {
        return;
    };
    // Calls 1 and 2 ran: their results are the tool's.
    assert!(second.contains(CONTENTS), "call 1 ran: {second}");
    assert_eq!(third.matches(CONTENTS).count(), 2, "call 2 ran: {third}");
    assert!(!third.contains("You already called"), "{third}");
    // The kernel starts a call's task before any plugin sees it: the refused
    // calls ran too, only their results were replaced.
    assert!(runs.load(Ordering::SeqCst) >= 2);
    // Call 3's result is the refusal, and request 4 carries it.
    assert!(fourth.contains(refused), "{fourth}");
    assert_eq!(fourth.matches(CONTENTS).count(), 2, "{fourth}");
    assert!(fifth.contains("these arguments 3 times"), "{fifth}");

    let seen = app.world().resource::<Seen>();
    assert!(
        matches!(seen.ended.as_slice(), [TurnOutcome::Stopped]),
        "{:?}",
        seen.ended
    );
    assert!(
        seen.notices
            .iter()
            .any(|notice| notice == "stopped: the agent was repeating `read a.txt`"),
        "{:?}",
        seen.notices
    );
    // The fifth call has a result, and nothing waits for the model.
    let conversation = app.world().get::<Conversation>(agent);
    let last = conversation.and_then(|conversation| conversation.messages().last());
    let last = serde_json::to_string(&last).unwrap_or_default();
    assert!(last.contains("stopped: the agent was repeating"), "{last}");
}
