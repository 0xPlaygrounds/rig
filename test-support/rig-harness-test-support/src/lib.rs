//! What the rig agent's plugin tests build: an app on a session store, a
//! model answering as the catalog's DeepSeek model, and its replies.

use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use rig_cassette::journal::MemoryStore;
use rig_core::catalog::Catalog;
use rig_core::operation::Completion;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_ecs::effects::Handler;
use rig_ecs::journal::SessionStore;
use rig_ecs::prelude::*;

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
