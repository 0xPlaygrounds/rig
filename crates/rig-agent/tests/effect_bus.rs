//! The agent over the effect bus: ownership, ordering, record and replay.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use futures::StreamExt;
use rig_agent::{
    Agent, AgentBuilder,
    agent::{AgentHook, DispatchAction, DispatchEvent, HookContext, OutcomeAction, OutcomeEvent},
    tool::{Tool, ToolContext, ToolExecutionError},
};
use rig_core::{
    bus::{Bus, BusConfig, BusDriver, EffectLogReplayer, adapters::CompletionAdapter},
    effect::{EffectFamily, EffectKind, HandlerKey},
    error::ErrorKind,
    test_utils::{MockCompletionModel, MockTurn},
};
use serde::Deserialize;
use serde_json::json;

async fn within<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(5), future)
        .await
        .expect("a run over the bus never hangs")
}

#[derive(Deserialize)]
struct SlowArgs {
    #[serde(default)]
    delay_ms: u64,
    tag: String,
}

/// Records the order tool calls *complete* in, after a per-call delay.
#[derive(Clone, Default)]
struct Slow {
    completed: Arc<Mutex<Vec<String>>>,
}

impl Tool for Slow {
    const NAME: &'static str = "slow";
    type Args = SlowArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "sleeps then answers".into()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {"delay_ms": {"type": "integer"}, "tag": {"type": "string"}}})
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: SlowArgs,
    ) -> Result<String, Self::Error> {
        tokio::time::sleep(Duration::from_millis(args.delay_ms)).await;
        self.completed.lock().expect("lock").push(args.tag.clone());
        Ok(args.tag)
    }
}

fn two_tool_calls_then_done() -> MockCompletionModel {
    MockCompletionModel::from_turns([
        MockTurn::from_contents([
            rig_core::message::AssistantContent::ToolCall(rig_core::message::ToolCall::from_wire(
                "tc-1",
                rig_core::message::ToolFunction::new(
                    "slow".to_owned(),
                    json!({"delay_ms": 40, "tag": "first"}),
                ),
            )),
            rig_core::message::AssistantContent::ToolCall(rig_core::message::ToolCall::from_wire(
                "tc-2",
                rig_core::message::ToolFunction::new(
                    "slow".to_owned(),
                    json!({"delay_ms": 0, "tag": "second"}),
                ),
            )),
        ]),
        MockTurn::text("done"),
    ])
}

#[tokio::test]
async fn serial_per_handler_is_proven_under_the_agents_inline_driver() {
    // With serial serving, the second call to the same handler waits for the
    // first even though the runner dispatches both concurrently.
    let serial = Slow::default();
    let agent = AgentBuilder::with_bus_config(
        BusConfig {
            serial_per_handler: true,
            ..BusConfig::default()
        },
        "default",
        two_tool_calls_then_done(),
    )
    .tool(serial.clone())
    .build();
    let response = within(agent.prompt("go").max_turns(3).tool_concurrency(2).run())
        .await
        .expect("run");
    assert_eq!(response.output, "done");
    assert_eq!(
        *serial.completed.lock().expect("lock"),
        vec!["first".to_string(), "second".to_string()],
        "serial serving keeps arrival order"
    );

    // Concurrent serving lets the shorter call finish first.
    let concurrent = Slow::default();
    let agent = AgentBuilder::new(two_tool_calls_then_done())
        .tool(concurrent.clone())
        .build();
    let response = within(agent.prompt("go").max_turns(3).tool_concurrency(2).run())
        .await
        .expect("run");
    assert_eq!(response.output, "done");
    assert_eq!(
        *concurrent.completed.lock().expect("lock"),
        vec!["second".to_string(), "first".to_string()],
        "concurrent serving finishes by delay"
    );
}

#[tokio::test]
async fn into_parts_hands_over_the_driver_with_the_dispatcher() {
    let agent = AgentBuilder::new(MockCompletionModel::text("parts")).build();
    assert!(agent.owns_bus());
    let parts = match agent.into_parts() {
        Ok(parts) => parts,
        Err(_) => panic!("the only clone can take the bus apart"),
    };
    let rig_agent::agent::AgentParts {
        dispatcher,
        driver,
        agent,
    } = parts;
    assert!(!agent.owns_bus(), "the agent no longer drives");

    // Spawn the driver ourselves; the dispatcher clone and the agent both
    // resolve through it.
    let task = tokio::spawn(driver);
    let handle: rig_core::bus::ModelHandle = dispatcher
        .handle(&HandlerKey::from("model:default"))
        .expect("the model is registered");
    assert_eq!(handle.model_ref().as_str(), "default");
    let response = within(agent.prompt("hello").run())
        .await
        .expect("served by the spawned driver");
    assert_eq!(response.output, "parts");
    drop(agent);
    drop(dispatcher);
    drop(handle);
    within(task)
        .await
        .expect("driver ends when every dispatcher is gone");
}

#[tokio::test]
async fn into_parts_fails_while_a_clone_still_shares_the_driver() {
    let agent = AgentBuilder::new(MockCompletionModel::text("shared")).build();
    let clone = agent.clone();
    let agent = match agent.into_parts() {
        Ok(_) => panic!("a clone still shares the driver"),
        Err(agent) => agent,
    };
    let response = within(agent.prompt("still runs").run()).await.expect("run");
    assert_eq!(response.output, "shared");
    drop(clone);
}

#[tokio::test]
async fn a_run_over_a_dropped_host_bus_answers_bus_closed_not_a_hang() {
    let (dispatcher, mut driver) = Bus::channel();
    driver.register(
        "model:host",
        CompletionAdapter::new("host", MockCompletionModel::text("never")),
    );
    let agent = AgentBuilder::over_bus(dispatcher, HandlerKey::from("model:host")).build();
    drop(driver);
    let error = within(agent.prompt("hello").run())
        .await
        .expect_err("closed bus");
    let message = error.to_string();
    assert!(
        message.contains("bus driver is gone"),
        "expected BusClosed, got {message}"
    );
}

#[tokio::test]
async fn a_run_records_every_dispatch_and_replays_from_the_log() {
    let recorded = AgentBuilder::new(two_tool_calls_then_done())
        .tool(Slow::default())
        .record_effects()
        .build();
    let response = within(recorded.prompt("go").max_turns(3).run())
        .await
        .expect("recorded run");
    assert_eq!(response.output, "done");
    let log = recorded.take_effect_log().expect("recording was enabled");
    assert_eq!(log.len(), 4, "two completions and two tool calls");
    assert_eq!(log[0].kind.family(), EffectFamily::Completion);
    assert_eq!(log[1].kind.family(), EffectFamily::Tool);
    assert_eq!(log[2].kind.family(), EffectFamily::Tool);
    assert_eq!(log[3].kind.family(), EffectFamily::Completion);
    assert!(log.iter().all(|record| record.outcome.is_ok()));

    // The examples-doc flow: save the session, restore it, replay it with
    // no model and no tool behind the keys.
    let saved = serde_json::to_string(&log).expect("log serializes");
    let restored: rig_core::effect::EffectLog = serde_json::from_str(&saved).expect("restores");
    let (dispatcher, mut driver) = Bus::channel();
    EffectLogReplayer::register_all(&restored, &mut driver);
    let replay_task = tokio::spawn(driver);

    // The replayed agent advertises the same tool (its handler is the
    // replayer under the recorded key) and dispatches to the recorded keys.
    let model_key = log[0].key.clone();
    let tool_key = log[1].key.clone();
    let tool_replayer =
        EffectLogReplayer::for_key(&restored, &tool_key).expect("the log has the tool's records");
    let replayed = AgentBuilder::over_bus(dispatcher.clone(), model_key)
        .tool_server_handle({
            let server = rig_agent::tool::server::ToolServer::new().run();
            // The registration's handler *is* the replayer, under the recorded
            // key: the catalog advertises the tool, the bus answers from the
            // log, and no `Slow` exists on this side at all.
            server.add_registered_tool(
                rig_core::tool::RegisteredTool::from_handler(tool_replayer)
                    .expect("a tool-family replayer"),
            );
            server
        })
        .build();
    let response = within(replayed.prompt("go").max_turns(3).run())
        .await
        .expect("replayed run");
    assert_eq!(
        response.output, "done",
        "the same PromptResponse from the record"
    );
    drop(replayed);
    drop(dispatcher);
    within(replay_task).await.expect("replay driver ends");
}

/// A hook at the dispatch boundary sees every effect with its id, can patch
/// a tool call's arguments, and can deny one.
#[derive(Clone, Default)]
struct Boundary {
    seen: Arc<Mutex<Vec<(u64, EffectFamily)>>>,
    outcomes: Arc<Mutex<Vec<(u64, bool)>>>,
}

impl AgentHook for Boundary {
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        self.seen
            .lock()
            .expect("lock")
            .push((event.id.as_u64(), event.kind.family()));
        match event.kind {
            EffectKind::ToolCall { name, context, .. } if name == "slow" => {
                DispatchAction::patch(EffectKind::ToolCall {
                    name: name.clone(),
                    args: json!({"delay_ms": 0, "tag": "patched"}).to_string(),
                    context: context.clone(),
                })
            }
            _ => DispatchAction::proceed(),
        }
    }

    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        self.outcomes
            .lock()
            .expect("lock")
            .push((event.id.as_u64(), event.outcome.is_ok()));
        OutcomeAction::proceed()
    }
}

#[tokio::test]
async fn dispatch_boundary_hooks_see_ids_and_patch_effects() {
    let tool = Slow::default();
    let boundary = Boundary::default();
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::tool_call("tc-1", "slow", json!({"delay_ms": 30, "tag": "original"})),
        MockTurn::text("done"),
    ]))
    .tool(tool.clone())
    .add_hook(boundary.clone())
    .build();
    let response = within(agent.prompt("go").max_turns(3).run())
        .await
        .expect("run");
    assert_eq!(response.output, "done");
    assert_eq!(
        *tool.completed.lock().expect("lock"),
        vec!["patched".to_string()],
        "the patched arguments reached the tool"
    );
    let seen = boundary.seen.lock().expect("lock").clone();
    assert_eq!(
        seen.iter().map(|(_, family)| *family).collect::<Vec<_>>(),
        vec![
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion
        ]
    );
    let outcomes = boundary.outcomes.lock().expect("lock").clone();
    assert_eq!(
        outcomes.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        seen.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        "every outcome carries the id its dispatch was seen with"
    );
    assert!(outcomes.iter().all(|(_, ok)| *ok));
}

struct DenyTools;

impl AgentHook for DenyTools {
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        match event.kind {
            EffectKind::ToolCall { .. } => DispatchAction::skip("policy says no"),
            _ => DispatchAction::proceed(),
        }
    }
}

#[tokio::test]
async fn a_denied_tool_dispatch_is_the_skipped_result_the_model_sees() {
    let tool = Slow::default();
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::tool_call("tc-1", "slow", json!({"tag": "denied"})),
        MockTurn::text("done"),
    ]))
    .tool(tool.clone())
    .add_hook(DenyTools)
    .build();
    let response = within(agent.prompt("go").max_turns(3).run())
        .await
        .expect("run");
    assert_eq!(response.output, "done");
    assert!(tool.completed.lock().expect("lock").is_empty(), "never ran");
    let history = response.messages.expect("history");
    let saw_skip = history.iter().any(|message| {
        serde_json::to_string(message)
            .expect("serializes")
            .contains("policy says no")
    });
    assert!(saw_skip, "the model saw the skip reason as the tool result");
}

struct CancelCompletion;

impl AgentHook for CancelCompletion {
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        match event.kind {
            EffectKind::Completion { .. } => DispatchAction::stop("halt before the model"),
            _ => DispatchAction::proceed(),
        }
    }
}

#[tokio::test]
async fn a_cancelled_completion_dispatch_cancels_the_run_before_the_model() {
    let model = MockCompletionModel::text("never");
    let agent = AgentBuilder::new(model.clone())
        .add_hook(CancelCompletion)
        .build();
    let error = within(agent.prompt("go").run())
        .await
        .expect_err("cancelled");
    assert!(
        error.to_string().contains("halt before the model"),
        "{error}"
    );
    assert_eq!(model.request_count(), 0, "the model never saw the request");
}

#[tokio::test]
async fn selecting_an_unregistered_model_label_fails_at_bind_time() {
    struct SelectMissing;
    impl AgentHook for SelectMissing {
        fn on_model_select(
            &self,
            _ctx: &HookContext,
            _event: rig_agent::agent::ModelSelection<'_>,
        ) -> rig_agent::agent::ModelSelectionAction {
            rig_agent::agent::ModelSelectionAction::select("nope")
        }
    }
    let model = MockCompletionModel::text("never");
    let agent = AgentBuilder::new(model.clone())
        .add_hook(SelectMissing)
        .build();
    let error = within(agent.prompt("go").run()).await.expect_err("unbound");
    let rig_agent::completion::PromptError::CompletionError(
        rig_core::completion::CompletionError::Report(report),
    ) = error
    else {
        panic!("expected a report, got {error}");
    };
    assert_eq!(report.kind, ErrorKind::HandlerUnavailable);
    assert!(report.message.contains("model:nope"), "{}", report.message);
    assert_eq!(model.request_count(), 0);
}

#[tokio::test]
async fn streaming_runs_drive_the_bus_too() {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([vec![
        rig_core::test_utils::MockStreamEvent::text("streamed "),
        rig_core::test_utils::MockStreamEvent::text("over the bus"),
        rig_core::test_utils::MockStreamEvent::final_response_with_total_tokens(3),
    ]]))
    .build();
    let mut stream = agent.stream_prompt("go").stream().await;
    let mut output = None;
    while let Some(item) = within(stream.next()).await {
        if let rig_agent::agent::MultiTurnStreamItem::FinalResponse(response) = item.expect("item")
        {
            output = Some(response.output().to_owned());
        }
    }
    assert_eq!(output.as_deref(), Some("streamed over the bus"));
}

fn _assertions(agent: Agent, driver: BusDriver) {
    fn assert_send<T: Send>(_: &T) {}
    assert_send(&agent);
    assert_send(&driver);
}
