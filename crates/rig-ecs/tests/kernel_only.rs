//! The kernel on its own: a headless app with Bevy's task pools, the agent
//! runtime and its session journal, and nothing else. A scripted model asks
//! for one tool call, then answers; a second app on the same store gets the
//! conversation back. Only rig-ecs's public API is used, as a third-party
//! plugin would.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use rig_core::completion::Message;
use rig_core::operation::Completion;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_ecs::agent::answer_text;
use rig_ecs::effects::Handler;
use rig_ecs::models::ModelConnector;
use rig_ecs::prelude::*;
use rig_ecs::store::{MemoryStore, SessionStore};
use serde::Deserialize;

/// How long a turn may take before the test gives up.
const DEADLINE: Duration = Duration::from_secs(10);

/// Adds two numbers, counting its calls.
struct Add(Arc<AtomicU32>);

#[derive(Deserialize)]
struct AddArgs {
    a: i64,
    b: i64,
}

impl PortableTool for Add {
    const NAME: &'static str = "add";
    type Args = AddArgs;
    type Output = i64;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Adds two integers.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "a": { "type": "integer" }, "b": { "type": "integer" } },
            "required": ["a", "b"],
        })
    }

    async fn call(&self, args: AddArgs) -> Result<i64, ToolExecutionError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(args.a + args.b)
    }
}

/// How each turn ended, in order.
#[derive(Resource, Default)]
struct Ended(Vec<TurnOutcome>);

/// A headless kernel app on `store`, with its loop's wakes.
fn kernel(store: &MemoryStore) -> (App, Receiver<()>) {
    let (sender, wakes) = channel();
    let mut app = App::new();
    app.add_plugins(TaskPoolPlugin::default())
        .insert_resource(Wake::new(move || {
            sender.send(()).ok();
        }))
        .insert_resource(SessionStore::new(store.clone()))
        .add_plugins((AgentPlugin, JournalPlugin))
        .init_resource::<Ended>()
        .add_observer(|ended: On<TurnEnded>, mut log: ResMut<Ended>| {
            log.0.push(ended.outcome.clone());
        });
    (app, wakes)
}

/// Runs frames, sleeping until a wake between them, as a windowless loop
/// does, until a turn ended or the deadline passed.
fn run_until_a_turn_ends(app: &mut App, wakes: &Receiver<()>) {
    let started = Instant::now();
    while started.elapsed() < DEADLINE {
        app.update();
        if !app.world().resource::<Ended>().0.is_empty() {
            return;
        }
        wakes.recv_timeout(Duration::from_millis(100)).ok();
    }
}

fn messages(app: &App, agent: Entity) -> Vec<Message> {
    app.world()
        .get::<Conversation>(agent)
        .map(|conversation| conversation.messages().to_vec())
        .unwrap_or_default()
}

/// The agents of `app`, with their ids.
fn agents(app: &mut App) -> Vec<(Entity, AgentId)> {
    let mut agents = app
        .world_mut()
        .query_filtered::<(Entity, &AgentId), With<Agent>>();
    agents
        .iter(app.world())
        .map(|(entity, id)| (entity, id.clone()))
        .collect()
}

/// A kernel app on `store` whose agent, on a scripted model, was asked a
/// question; it ran the frames until the turn ended. Returns the app, the
/// agent, the model and how often the tool ran.
fn one_turn(store: &MemoryStore) -> Option<(App, Entity, MockCompletionModel, u32)> {
    let (mut app, wakes) = kernel(store);
    let calls = Arc::new(AtomicU32::new(0));
    app.add_tool(Add(calls.clone()));
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("call-1", "add", serde_json::json!({ "a": 2, "b": 3 })),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::text("5"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let spec = ModelConnector::default().resolve("deepseek/deepseek-flash")?;
    let handler = ErasedHandler::new(ModelAdapter::<Completion>::new(
        spec.reference(),
        model.clone(),
    ));
    let connection = Connection {
        spec,
        handler: Handler(handler),
    };
    let agent = app.world_mut().spawn((Agent, connection)).id();
    // The first frame restores the (empty) session and starts the journal.
    app.update();
    app.world_mut().trigger(Deliver::user(
        agent,
        "What is 2 + 3?",
        DeliveryMode::Steer,
        Vec::new(),
    ));
    run_until_a_turn_ends(&mut app, &wakes);
    let ran = calls.load(Ordering::Relaxed);
    Some((app, agent, model, ran))
}

#[test]
fn a_turn_ends_answered_after_one_tool_round() {
    let turn = one_turn(&MemoryStore::default());
    assert!(turn.is_some(), "the built-in catalog lists the model");
    let Some((app, agent, model, ran)) = turn else {
        return;
    };
    let ended = &app.world().resource::<Ended>().0;
    assert_eq!(ended.len(), 1, "{ended:?}");
    let answer = match ended.first() {
        Some(TurnOutcome::Answered(message)) => answer_text(message),
        _ => None,
    };
    assert_eq!(answer.as_deref(), Some("5"), "{ended:?}");
    assert_eq!(ran, 1, "the tool ran once");
    assert_eq!(model.request_count(), 2, "one tool round, then the answer");
    // The question, the tool call, its result and the answer.
    let said = messages(&app, agent);
    assert_eq!(said.len(), 4, "{said:?}");
}

#[test]
fn a_new_app_on_the_store_restores_the_conversation() {
    let store = MemoryStore::default();
    let turn = one_turn(&store);
    assert!(turn.is_some(), "the built-in catalog lists the model");
    let Some((mut app, agent, _, _)) = turn else {
        return;
    };
    let said = messages(&app, agent);
    assert_eq!(said.len(), 4, "{said:?}");
    let before = agents(&mut app);
    drop(app);

    let (mut restored, _wakes) = kernel(&store);
    restored.update();
    let after = agents(&mut restored);
    let ids = |agents: &[(Entity, AgentId)]| -> Vec<AgentId> {
        agents.iter().map(|(_, id)| id.clone()).collect()
    };
    assert_eq!(ids(&after), ids(&before), "the same one agent");
    let Some((again, _)) = after.first() else {
        return;
    };
    assert_eq!(messages(&restored, *again), said);
}
