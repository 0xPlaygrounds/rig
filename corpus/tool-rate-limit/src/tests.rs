use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use bevy_time::TimeUpdateStrategy;
use futures::FutureExt;
use futures::channel::oneshot;
use futures::future::Shared;
use rig_cassette::journal::MemoryStore;
use rig_core::completion::{CompletionRequest, Message};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{app_on, connect, reply, run_until};
use schemars::JsonSchema;
use serde_json::json;

use super::*;

#[derive(Deserialize, JsonSchema)]
struct Args {
    id: String,
}

/// Answers its call's id, counting its calls.
struct EchoTool(Arc<AtomicUsize>);

impl PortableTool for EchoTool {
    const NAME: &'static str = "echo_tool";
    type Args = Args;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Echo.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<Args>()
    }

    async fn call(&self, args: Args) -> Result<String, ToolExecutionError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(args.id)
    }
}

/// Runs until the test opens its gate, logging when each call starts and
/// ends.
struct GateTool {
    log: Arc<Mutex<Vec<String>>>,
    open: Shared<oneshot::Receiver<()>>,
}

impl PortableTool for GateTool {
    const NAME: &'static str = "gate";
    type Args = Args;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Wait for the gate.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<Args>()
    }

    async fn call(&self, args: Args) -> Result<String, ToolExecutionError> {
        let note = |what: &str| {
            if let Ok(mut log) = self.log.lock() {
                log.push(format!("{what} {}", args.id));
            }
        };
        note("start");
        self.open.clone().await.ok();
        note("end");
        Ok(args.id.clone())
    }
}

/// How many turns ended.
#[derive(Resource, Default)]
struct Ended(usize);

/// The kernel with this plugin and `limits`, its agent answered by `model`
/// with the calls `tools` registers, and a message delivered.
fn start(
    limits: &str,
    model: &MockCompletionModel,
    tools: impl FnOnce(&mut App),
) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let limits: Limits = toml::from_str(limits).unwrap_or_default();
    app.insert_resource(limits)
        .add_plugins((AgentPlugin, JournalPlugin, ToolRateLimitPlugin))
        .init_resource::<Ended>()
        .add_observer(|_: On<TurnEnded>, mut ended: ResMut<Ended>| ended.0 += 1);
    tools(&mut app);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()
        .unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(model) {
        world.entity_mut(agent).insert(connection);
    }
    world.trigger(Deliver::new(agent, "go", DeliveryMode::Queue));
    (app, wakes)
}

/// A model that calls `tool` once for each of `ids`, in one reply, then
/// answers.
fn model(tool: &str, ids: &[&str]) -> MockCompletionModel {
    let mut calls: Vec<_> = ids
        .iter()
        .map(|id| MockStreamEvent::tool_call(*id, tool, json!({ "id": id })))
        .collect();
    calls.push(MockStreamEvent::final_response_with_default_usage());
    MockCompletionModel::from_stream_turns([calls, reply("Done.")])
}

/// The tool results `request` sends, as JSON text.
fn results(request: Option<&CompletionRequest>) -> Vec<String> {
    let messages = request.map(|request| request.chat_history.iter());
    messages
        .into_iter()
        .flatten()
        .filter_map(|message| match message {
            Message::User { content } => Some(content.iter()),
            _ => None,
        })
        .flatten()
        .filter_map(|content| match content {
            UserContent::ToolResult(result) => serde_json::to_string(result).ok(),
            _ => None,
        })
        .collect()
}

fn ended(world: &World) -> bool {
    world.resource::<Ended>().0 > 0
}

#[test]
fn a_call_over_the_turn_limit_is_refused_and_not_run() {
    let ran = Arc::new(AtomicUsize::new(0));
    let model = model("echo_tool", &["a", "b", "c"]);
    let echo = EchoTool(ran.clone());
    let (mut app, wakes) = start("[echo_tool]\nper_turn = 2", &model, |app| {
        app.add_limited_tool_with(echo, ToolOptions::default());
    });
    run_until(&mut app, &wakes, ended);

    assert_eq!(ran.load(Ordering::SeqCst), 2);
    let sent = results(model.requests().get(1));
    assert_eq!(sent.len(), 3, "{sent:?}");
    let refused: Vec<_> = sent
        .iter()
        .filter(|r| r.contains("limit reached"))
        .collect();
    assert_eq!(refused.len(), 1, "{sent:?}");
    assert!(
        sent.last().is_some_and(|r| r.contains("2 per turn")),
        "{sent:?}"
    );
}

#[test]
fn over_the_concurrent_limit_a_read_only_call_waits_for_the_running_one() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let (gate, open) = oneshot::channel();
    let tool = GateTool {
        log: log.clone(),
        open: open.shared(),
    };
    let model = model("gate", &["a", "b"]);
    let read_only = ToolOptions {
        rules: &[],
        footprint: Footprint::ReadOnly,
    };
    let (mut app, wakes) = start("[gate]\nconcurrent = 1", &model, |app| {
        app.add_limited_tool_with(tool, read_only);
    });
    let logged =
        |log: &Arc<Mutex<Vec<String>>>| log.lock().map(|log| log.clone()).unwrap_or_default();
    run_until(&mut app, &wakes, |_| !logged(&log).is_empty());
    for _ in 0..20 {
        app.update();
    }
    assert_eq!(logged(&log), ["start a"]);

    gate.send(()).ok();
    run_until(&mut app, &wakes, ended);
    assert_eq!(logged(&log), ["start a", "end a", "start b", "end b"]);
}

#[test]
fn over_the_minute_limit_a_call_starts_once_the_minute_passed() {
    let ran = Arc::new(AtomicUsize::new(0));
    let model = model("echo_tool", &["a", "b"]);
    let echo = EchoTool(ran.clone());
    let (mut app, wakes) = start("[echo_tool]\nper_minute = 1", &model, |app| {
        app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO))
            .add_limited_tool_with(echo, ToolOptions::default());
    });
    app.world_mut()
        .resource_mut::<Time<Virtual>>()
        .set_max_delta(Duration::from_secs(120));
    run_until(&mut app, &wakes, |_| ran.load(Ordering::SeqCst) > 0);
    for _ in 0..20 {
        app.update();
    }
    assert_eq!(ran.load(Ordering::SeqCst), 1);

    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs(61)));
    app.update();
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO));
    run_until(&mut app, &wakes, ended);
    assert_eq!(ran.load(Ordering::SeqCst), 2);
    let sent = results(model.requests().get(1));
    assert!(sent.iter().all(|r| !r.contains("limit")), "{sent:?}");
}
