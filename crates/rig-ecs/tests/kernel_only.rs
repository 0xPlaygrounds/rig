//! The kernel on its own: a headless app with Bevy's task pools, the agent
//! runtime and its session journal, and nothing else. A scripted model asks
//! for one tool call, then answers; a second app on the same store gets the
//! conversation back; a failed call is sent again once its backoff passed
//! on Bevy's clock; a model connects the same whichever of its settings
//! comes first. Only rig-ecs's public API is used, as a third-party plugin
//! would.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use bevy_time::TimeUpdateStrategy;
use rig_cassette::journal::MemoryStore;
use rig_core::ProviderResponseError;
use rig_core::completion::{Message, Reasoning};
use rig_core::message::UserContent;
use rig_core::operation::Completion;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::{MockCompletionModel, MockError, MockStreamEvent};
use rig_core::transcript::final_answer;
use rig_ecs::effects::Handler;
use rig_ecs::journal::SessionStore;
use rig_ecs::models::ModelConnector;
use rig_ecs::prelude::*;
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
/// does, until `done` or the deadline passed.
fn run_until(app: &mut App, wakes: &Receiver<()>, done: impl Fn(&mut World) -> bool) {
    let started = Instant::now();
    while started.elapsed() < DEADLINE {
        app.update();
        if done(app.world_mut()) {
            return;
        }
        wakes.recv_timeout(Duration::from_millis(100)).ok();
    }
}

fn a_turn_ended(world: &mut World) -> bool {
    !world.resource::<Ended>().0.is_empty()
}

/// An agent of `app` on `model`, as the built-in catalog's DeepSeek model.
fn connected(app: &mut App, model: &MockCompletionModel) -> Option<Entity> {
    let spec = ModelConnector::default().resolve("deepseek/deepseek-flash")?;
    let handler = ErasedHandler::new(ModelAdapter::<Completion>::new(
        spec.reference(),
        model.clone(),
    ));
    let connection = Connection {
        spec,
        handler: Handler(handler),
    };
    Some(app.world_mut().spawn((Agent, connection)).id())
}

fn ask(app: &mut App, agent: Entity) {
    // The first frame restores the (empty) session and starts the journal.
    app.update();
    app.world_mut().trigger(Deliver::user(
        agent,
        "What is 2 + 3?",
        DeliveryMode::Steer,
        Vec::new(),
    ));
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
/// question; it ran the frames until the turn ended. The model's first
/// reply also calls a tool that does not exist. Returns the app, the agent,
/// the model and how often the tool ran.
fn one_turn(store: &MemoryStore) -> Option<(App, Entity, MockCompletionModel, u32)> {
    let (mut app, wakes) = kernel(store);
    let calls = Arc::new(AtomicU32::new(0));
    app.add_tool(Add(calls.clone()));
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("call-1", "add", serde_json::json!({ "a": 2, "b": 3 })),
            MockStreamEvent::tool_call("call-2", "sub", serde_json::json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::text("5"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let agent = connected(&mut app, &model)?;
    ask(&mut app, agent);
    run_until(&mut app, &wakes, a_turn_ended);
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
        Some(TurnOutcome::Answered(message)) => final_answer(message),
        _ => None,
    };
    assert_eq!(answer.as_deref(), Some("5"), "{ended:?}");
    assert_eq!(ran, 1, "the tool ran once");
    assert_eq!(model.request_count(), 2, "one tool round, then the answer");
    // The question, the tool calls, their results in call order and the answer.
    let said = messages(&app, agent);
    assert_eq!(said.len(), 4, "{said:?}");
    let results: Vec<(bool, String)> = match said.get(2) {
        Some(Message::User { content }) => content
            .iter()
            .filter_map(|item| match item {
                UserContent::ToolResult(result) => Some((result.is_error, format!("{result:?}"))),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    let refused = |text: &String| text.contains("no tool named `sub` is available");
    assert!(
        matches!(results.as_slice(), [(false, _), (true, sub)] if refused(sub)),
        "{results:?}"
    );
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

/// A kernel app on a model whose first call fails with an error worth
/// retrying and whose second answers, on a clock that moves only when the
/// test moves it. It ran until the failed call's [`Backoff`] began.
fn backing_off() -> Option<(App, Receiver<()>, Entity, MockCompletionModel)> {
    let (mut app, wakes) = kernel(&MemoryStore::default());
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO));
    let overloaded = ProviderResponseError::without_status("overloaded").with_transient(Some(true));
    let model = MockCompletionModel::from_stream_turns([
        vec![MockStreamEvent::Error(MockError::ProviderResponse(
            overloaded,
        ))],
        vec![
            MockStreamEvent::text("5"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let agent = connected(&mut app, &model)?;
    ask(&mut app, agent);
    run_until(&mut app, &wakes, |world| {
        world.query::<&Backoff>().iter(world).next().is_some() || a_turn_ended(world)
    });
    Some((app, wakes, agent, model))
}

/// Runs one frame `by` later on the app's clock, then a few with the
/// clock still.
fn advance(app: &mut App, by: Duration) {
    app.insert_resource(TimeUpdateStrategy::ManualDuration(by));
    app.update();
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO));
    for _ in 0..3 {
        app.update();
    }
}

#[test]
fn a_failed_call_is_sent_again_once_its_backoff_passed_on_the_clock() {
    let started = backing_off();
    assert!(started.is_some(), "the built-in catalog lists the model");
    let Some((mut app, wakes, _, model)) = started else {
        return;
    };
    assert_eq!(
        model.request_count(),
        1,
        "{:?}",
        app.world().resource::<Ended>().0
    );
    // The first backoff is 1.5 to 2 seconds; no time passes on its own.
    advance(&mut app, Duration::from_secs(1));
    assert_eq!(model.request_count(), 1, "still waiting");
    advance(&mut app, Duration::from_secs(1));
    run_until(&mut app, &wakes, a_turn_ended);
    let ended = &app.world().resource::<Ended>().0;
    let answer = match ended.as_slice() {
        [TurnOutcome::Answered(message)] => final_answer(message),
        _ => None,
    };
    assert_eq!(answer.as_deref(), Some("5"), "{ended:?}");
    assert_eq!(model.request_count(), 2);
}

#[test]
fn an_interrupt_during_a_backoff_cancels_the_retry() {
    let started = backing_off();
    assert!(started.is_some(), "the built-in catalog lists the model");
    let Some((mut app, _wakes, agent, model)) = started else {
        return;
    };
    app.world_mut().trigger(Interrupt { entity: agent });
    advance(&mut app, Duration::from_secs(5));
    assert!(
        matches!(
            app.world().resource::<Ended>().0.as_slice(),
            [TurnOutcome::Stopped]
        ),
        "{:?}",
        app.world().resource::<Ended>().0
    );
    assert_eq!(model.request_count(), 1, "the call was not sent again");
}

#[test]
fn a_model_connects_the_same_whichever_setting_a_restore_inserts_first() {
    let (mut app, _wakes) = kernel(&MemoryStore::default());
    let model = MockCompletionModel::from_turns([]);
    let agents = [connected(&mut app, &model), connected(&mut app, &model)];
    assert!(agents.iter().all(Option::is_some));
    let [Some(effort_first), Some(model_first)] = agents else {
        return;
    };
    let world = app.world_mut();
    let reference =
        |world: &World, agent| world.get::<Connection>(agent).map(|c| c.spec.reference());
    let choice = ModelChoice(reference(world, effort_first).unwrap_or_default());
    // The model takes effort levels, not a reasoning budget.
    let budget = Effort(Some(Reasoning::Budget { tokens: 1000 }));
    let mut agent = world.entity_mut(effort_first);
    agent.insert(budget).insert(choice.clone());
    world
        .entity_mut(model_first)
        .insert(choice.clone())
        .insert(budget);
    world.flush();
    let settings = |agent| (reference(world, agent), world.get::<Effort>(agent).copied());
    let expected = (Some(choice.0), Some(Effort(None)));
    assert_eq!(settings(effort_first), expected);
    assert_eq!(settings(model_first), expected);
}
