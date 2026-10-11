//! What the rig agent's plugin tests build: an app on a session store, a
//! model answering as the catalog's DeepSeek model, its replies, and a
//! front that answers questions from a script ([`AnswerAsks`]).

use std::collections::{HashSet, VecDeque};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use rig_cassette::journal::MemoryStore;
use rig_core::catalog::Catalog;
use rig_core::effect::EffectId;
use rig_core::message::{ToolFunction, ToolName};
use rig_core::operation::Completion;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_ecs::effects::Handler;
use rig_ecs::journal::SessionStore;
use rig_ecs::prelude::*;
use rig_harness::ask::{Answer, Answerer, Ask};
use rig_harness::settings::Settings;

/// How long a test waits for its agents.
const DEADLINE: Duration = Duration::from_secs(10);

/// An app with a task pool on the session in `store`, no user settings,
/// and its loop's wakes.
pub fn app_on(store: &MemoryStore) -> (App, Receiver<()>) {
    let (sender, wakes) = channel();
    let mut app = App::new();
    app.add_plugins(TaskPoolPlugin::default())
        .insert_resource(Wake::new(move || {
            sender.send(()).ok();
        }))
        .insert_resource(SessionStore::new(store.clone()))
        .insert_resource(Settings::default());
    (app, wakes)
}

/// How each turn ended, in the order they ended, kept by [`log_ended`].
#[derive(Resource, Default)]
pub struct Ended(pub Vec<TurnOutcome>);

/// Keeps how each turn ended in [`Ended`]: `app.init_resource::<Ended>()`
/// and `.add_observer(log_ended)`.
pub fn log_ended(settled: On<Add<Settled>>, turns: Query<&TurnOutcome>, mut log: ResMut<Ended>) {
    log.0.extend(turns.get(settled.entity).cloned());
}

/// `model` as the catalog's DeepSeek model.
pub fn connect(model: &MockCompletionModel) -> Option<Connection> {
    let spec = Catalog::builtin()
        .resolve("deepseek/deepseek-flash")
        .ok()?
        .shared();
    let handler = ModelAdapter::<Completion>::new(spec.reference(), model.clone());
    Some(Connection {
        spec,
        handler: Handler(ErasedHandler::new(handler)),
    })
}

/// Runs frames of `app`, sleeping until one of its `wakes` between them,
/// until `done` or the deadline passed.
pub fn run_until(app: &mut App, wakes: &Receiver<()>, done: impl Fn(&World) -> bool) {
    let started = Instant::now();
    while started.elapsed() < DEADLINE && !done(app.world()) {
        app.update();
        wakes.recv_timeout(Duration::from_millis(100)).ok();
    }
}

/// A model reply of `text`, then the end of the stream.
pub fn reply(text: &str) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::text(text),
        MockStreamEvent::final_response_with_default_usage(),
    ]
}

/// A model reply that calls `tool` with `args` in call `id`.
pub fn calls(id: &str, tool: &str, args: serde_json::Value) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::tool_call(id, tool, args),
        MockStreamEvent::final_response_with_default_usage(),
    ]
}

/// Spawns `agent`'s call `id` of `tool` with `args`, as its model call 1
/// asked, and returns the call's entity, which gets the call's `Outcome`.
/// The next frame starts it.
pub fn start_call(
    world: &mut World,
    agent: Entity,
    (tool, id): (&str, &str),
    args: serde_json::Value,
) -> Option<Entity> {
    let name = ToolName::new(tool).ok()?;
    let call = ToolCall::from_wire(id, ToolFunction::new(name, args));
    let parent = Effect::child_of(Some(EffectId::from_raw(1)));
    Some(
        world
            .spawn((ToolRequest { call }, parent, ForAgent(agent)))
            .id(),
    )
}

/// The answers [`AnswerAsks`] gives, in order, and the title of every
/// question it saw, in the order they were asked.
#[derive(Resource, Default, Debug)]
pub struct Answers {
    /// What the next questions are answered; a question asked when none
    /// is left stays open.
    pub script: VecDeque<Answer>,
    /// Each question's title.
    pub asked: Vec<String>,
}

impl Answers {
    /// Answers the next questions with `script`, in order.
    pub fn new(script: impl IntoIterator<Item = Answer>) -> Self {
        Self {
            script: script.into_iter().collect(),
            asked: Vec::new(),
        }
    }
}

/// A front someone sits at who answers every question at once from the
/// [`Answers`] script: an [`Answerer`] for every agent, answering in Gate.
pub struct AnswerAsks;

impl Plugin for AnswerAsks {
    fn build(&self, app: &mut App) {
        app.world_mut().spawn(Answerer::default());
        app.init_resource::<Answers>()
            .add_systems(AgentPass, answer_asks.in_set(Stage::Gate));
    }
}

fn answer_asks(
    asks: Query<(Entity, &Ask), (Without<Outcome>, Without<Settled>)>,
    mut seen: Local<HashSet<Entity>>,
    mut answers: ResMut<Answers>,
    mut commands: Commands,
) {
    for (entity, ask) in &asks {
        if !seen.insert(entity) {
            continue;
        }
        answers.asked.push(ask.title.clone());
        if let Some(answer) = answers.script.pop_front() {
            commands.entity(entity).insert_if_new(answer.outcome());
        }
    }
}
