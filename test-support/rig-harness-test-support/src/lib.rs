//! What the rig agent's plugin tests build: an app on a session store, a
//! model answering as the catalog's DeepSeek model, and its replies.

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
use rig_ecs::turn::ToolStarter;

/// How long a test waits for its agents.
const DEADLINE: Duration = Duration::from_secs(10);

/// An app with a task pool on the session in `store`, and its loop's wakes.
pub fn app_on(store: &MemoryStore) -> (App, Receiver<()>) {
    let (sender, wakes) = channel();
    let mut app = App::new();
    app.add_plugins(TaskPoolPlugin::default())
        .insert_resource(Wake::new(move || {
            sender.send(()).ok();
        }))
        .insert_resource(SessionStore::new(store.clone()));
    (app, wakes)
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

/// Starts `agent`'s call `id` of `tool` with `args`, as its model call 1
/// asked, and returns the call's entity, which gets the call's
/// `ToolOutput`.
pub fn start_call(
    world: &mut World,
    agent: Entity,
    (tool, id): (&str, &str),
    args: serde_json::Value,
) -> Option<Entity> {
    let name = ToolName::new(tool).ok()?;
    let call = ToolCall::from_wire(id, ToolFunction::new(name, args));
    let start = move |starter: ToolStarter, mut commands: Commands| {
        let run = starter.run(call.clone(), Some(EffectId::from_raw(1)));
        let entity = commands.spawn(run.clone()).id();
        starter.start(&mut commands, entity, agent, &run);
        entity
    };
    let start = world.register_system(start);
    world.run_system(start).ok()
}
