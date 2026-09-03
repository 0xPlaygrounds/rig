//! The agent over the effect bus: ownership, ordering, record and replay.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

use std::{
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use futures::StreamExt;
use rig_agent::{
    Agent, AgentBuilder,
    agent::{
        AgentHook, CompletionCallAction, CompletionCallEvent, DispatchAction, DispatchEvent,
        HookContext, InvalidToolCallAction, InvalidToolCallContext, ModelSelection,
        ModelSelectionAction, ModelTurnAction, ModelTurnFinished, ObservationAction, OutcomeAction,
        OutcomeEvent, ReasoningDelta, RunSettled, RunStart, RunStartAction, StepEventKind,
        TextDelta, ToolCallDelta,
    },
    tool::{Tool, ToolContext, ToolExecutionError, ToolSet},
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
    driver
        .register(
            "model:host",
            CompletionAdapter::new("host", MockCompletionModel::text("never")),
        )
        .expect("register");
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
    EffectLogReplayer::register_all(&restored, &mut driver).expect("fresh keys");
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
    let rig_agent::completion::PromptError::Report(report) = error else {
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

fn streamed_text(text: &str) -> Vec<rig_core::test_utils::MockStreamEvent> {
    vec![
        rig_core::test_utils::MockStreamEvent::text(text),
        rig_core::test_utils::MockStreamEvent::final_response_with_total_tokens(1),
    ]
}

async fn drain(stream: &mut rig_agent::agent::StreamingResult) -> Option<String> {
    let mut output = None;
    while let Some(item) = within(stream.next()).await {
        if let rig_agent::agent::MultiTurnStreamItem::FinalResponse(response) = item.expect("item")
        {
            output = Some(response.output().to_owned());
        }
    }
    output
}

// Whoever holds the driver drives — and a run that has *finished* holds
// nothing. A finished stream its owner keeps in scope must not block the
// next run on the agent.
#[tokio::test]
async fn a_finished_stream_kept_in_scope_does_not_block_the_next_run() {
    let agent = AgentBuilder::named_model(
        "stream",
        MockCompletionModel::from_stream_turns([streamed_text("first")]),
    )
    .model_route("unary", MockCompletionModel::text("second"))
    .build();
    let mut stream = agent.stream_prompt("go").stream().await;
    assert_eq!(drain(&mut stream).await.as_deref(), Some("first"));
    // `stream` is still alive here.
    let response = within(agent.prompt("again").using_model("unary").run())
        .await
        .expect("the finished stream released the driver");
    assert_eq!(response.output, "second");
    drop(stream);
}

#[tokio::test]
async fn two_streams_polled_alternately_both_complete() {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        streamed_text("one"),
        streamed_text("two"),
    ]))
    .build();
    let mut first = agent.stream_prompt("a").stream().await;
    let mut second = agent.stream_prompt("b").stream().await;
    let (mut out_first, mut out_second) = (None, None);
    let (mut done_first, mut done_second) = (false, false);
    while !(done_first && done_second) {
        if !done_first {
            match within(first.next()).await {
                Some(item) => {
                    if let rig_agent::agent::MultiTurnStreamItem::FinalResponse(r) =
                        item.expect("item")
                    {
                        out_first = Some(r.output().to_owned());
                    }
                }
                None => done_first = true,
            }
        }
        if !done_second {
            match within(second.next()).await {
                Some(item) => {
                    if let rig_agent::agent::MultiTurnStreamItem::FinalResponse(r) =
                        item.expect("item")
                    {
                        out_second = Some(r.output().to_owned());
                    }
                }
                None => done_second = true,
            }
        }
    }
    let mut outputs = [out_first.expect("first"), out_second.expect("second")];
    outputs.sort();
    assert_eq!(outputs, ["one".to_string(), "two".to_string()]);
}

#[tokio::test]
async fn a_prompt_awaited_inside_a_stream_loop_on_a_clone_resolves() {
    let agent = AgentBuilder::named_model(
        "stream",
        MockCompletionModel::from_stream_turns([streamed_text("outer")]),
    )
    .model_route("unary", MockCompletionModel::text("inner"))
    .build();
    let clone = agent.clone();
    let mut stream = agent.stream_prompt("go").stream().await;
    let mut inner = None;
    let mut outer = None;
    while let Some(item) = within(stream.next()).await {
        if let rig_agent::agent::MultiTurnStreamItem::FinalResponse(r) = item.expect("item") {
            outer = Some(r.output().to_owned());
        }
        if inner.is_none() {
            // The outer run holds the driver; the clone's run queues on it
            // and is served by the outer run's polling.
            let response = within(clone.prompt("nested").using_model("unary").run())
                .await
                .expect("the nested run is served by the driving run");
            inner = Some(response.output);
        }
    }
    assert_eq!(outer.as_deref(), Some("outer"));
    assert_eq!(inner.as_deref(), Some("inner"));
}

/// A tool that never answers; its drop is the observable cancellation.
#[derive(Clone, Default)]
struct Hanging {
    dropped: Arc<AtomicBool>,
}

struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

impl Tool for Hanging {
    const NAME: &'static str = "hanging";
    type Args = serde_json::Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "never answers".into()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({"type": "object"})
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: serde_json::Value,
    ) -> Result<String, Self::Error> {
        let _flag = DropFlag(self.dropped.clone());
        tokio::time::sleep(Duration::from_secs(30)).await;
        Ok("never".into())
    }
}

fn one_tool_call(name: &str) -> MockCompletionModel {
    MockCompletionModel::from_turns([
        MockTurn::from_contents([rig_core::message::AssistantContent::ToolCall(
            rig_core::message::ToolCall::from_wire(
                "tc-1",
                rig_core::message::ToolFunction::new(name.to_owned(), json!({})),
            ),
        )]),
        MockTurn::text("done"),
    ])
}

#[tokio::test]
async fn a_run_dropped_mid_flight_cancels_the_tool_immediately() {
    let tool = Hanging::default();
    let agent = AgentBuilder::new(one_tool_call("hanging"))
        .tool(tool.clone())
        .build();
    let run = agent.prompt("go").max_turns(2).run();
    // Dropping the timed-out future drops the run mid-tool.
    let timed_out = tokio::time::timeout(Duration::from_millis(50), run)
        .await
        .is_err();
    assert!(timed_out, "the tool never answers");
    assert!(
        tool.dropped.load(Ordering::SeqCst),
        "dropping the run gave the driver its last poll and the tool future was dropped with it"
    );
    // The agent is usable afterwards: nothing holds the driver.
    let again = AgentBuilder::new(MockCompletionModel::text("fresh")).build();
    let _ = again;
    let response = within(agent.prompt("again").max_turns(1).run()).await;
    // The scripted model has one turn left ("done") for this run.
    assert_eq!(response.expect("run after a dropped one").output, "done");
}

/// A tool that runs a nested prompt on a clone of the agent it belongs to.
#[derive(Clone, Default)]
struct Nested {
    agent: Arc<OnceLock<Agent>>,
}

impl Tool for Nested {
    const NAME: &'static str = "nested";
    type Args = serde_json::Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "asks the agent again".into()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({"type": "object"})
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: serde_json::Value,
    ) -> Result<String, Self::Error> {
        let agent = self.agent.get().expect("set after build").clone();
        let response = agent
            .prompt("nested")
            .max_turns(1)
            .run()
            .await
            .map_err(|err| {
                ToolExecutionError::new(rig_agent::tool::ToolErrorKind::Other, err.to_string())
            })?;
        Ok(response.output)
    }
}

#[tokio::test]
async fn a_nested_agent_call_from_a_tool_is_served_by_the_driving_run() {
    // Script order: the outer turn calls the tool, the nested run takes the
    // next turn, the outer run's second turn ends it.
    let tool = Nested::default();
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::from_contents([rig_core::message::AssistantContent::ToolCall(
            rig_core::message::ToolCall::from_wire(
                "tc-1",
                rig_core::message::ToolFunction::new("nested".to_owned(), json!({})),
            ),
        )]),
        MockTurn::text("inner-done"),
        MockTurn::text("done"),
    ]))
    .tool(tool.clone())
    .build();
    tool.agent.set(agent.clone()).ok().expect("unset");
    let response = within(agent.prompt("go").max_turns(3).run())
        .await
        .expect("the nested run is served while the outer run drives");
    assert_eq!(response.output, "done");
}

/// A tool that, from inside its own execution, runs a nested prompt whose
/// model calls this same tool again.
#[derive(Clone, Default)]
struct NestedSameTool {
    agent: Arc<OnceLock<Agent>>,
    inner_outputs: Arc<Mutex<Vec<String>>>,
}

impl Tool for NestedSameTool {
    const NAME: &'static str = "same";
    type Args = serde_json::Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "asks the agent to call me again".into()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({"type": "object"})
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: serde_json::Value,
    ) -> Result<String, Self::Error> {
        let agent = self.agent.get().expect("set after build").clone();
        let response = agent
            .prompt("nested")
            .max_turns(2)
            .run()
            .await
            .map_err(|err| {
                ToolExecutionError::new(rig_agent::tool::ToolErrorKind::Other, err.to_string())
            })?;
        self.inner_outputs
            .lock()
            .expect("lock")
            .push(response.output.clone());
        Ok(response.output)
    }
}

#[tokio::test]
async fn a_nested_call_to_the_in_flight_tool_under_serial_serving_fails_fast() {
    // Outer turn: call `same`. Inside it the nested run's model calls `same`
    // again — under serial serving that would queue behind the outer call
    // that waits on it, so the bus refuses it and the nested model sees a
    // skipped tool result, answers, and the outer run completes.
    let tool = NestedSameTool::default();
    let call_same = || {
        MockTurn::from_contents([rig_core::message::AssistantContent::ToolCall(
            rig_core::message::ToolCall::from_wire(
                "tc",
                rig_core::message::ToolFunction::new("same".to_owned(), json!({})),
            ),
        )])
    };
    let agent = AgentBuilder::with_bus_config(
        BusConfig {
            serial_per_handler: true,
            ..BusConfig::default()
        },
        "default",
        MockCompletionModel::from_turns([
            call_same(),
            call_same(),
            MockTurn::text("inner-done"),
            MockTurn::text("done"),
        ]),
    )
    .tool(tool.clone())
    .build();
    tool.agent.set(agent.clone()).ok().expect("unset");
    let response = within(agent.prompt("go").max_turns(3).run())
        .await
        .expect("the outer run completes: the re-entrant call was refused, not queued");
    assert_eq!(response.output, "done");
    assert_eq!(
        *tool.inner_outputs.lock().expect("lock"),
        vec!["inner-done".to_string()]
    );
}

// ---------------------------------------------------------------------------
// The hook invocation sequence: every hook, every run shape, pinned exactly.
// Before the collapse a model turn read `on_completion_call → on_model_select
// → on_dispatch(completion) → on_outcome(completion) → on_completion_response
// → on_model_turn_finished` and a tool call `on_tool_call →
// on_dispatch(tool_call) → on_outcome(tool_call) → on_tool_result`; the
// vectors below are those sequences with the collapsed entries removed.
// ---------------------------------------------------------------------------

/// Records every hook invocation, in order, and opts into every family.
#[derive(Clone, Default)]
struct Sequence(Arc<Mutex<Vec<String>>>);

impl Sequence {
    fn push(&self, entry: impl Into<String>) {
        self.0.lock().expect("lock").push(entry.into());
    }

    fn take(&self) -> Vec<String> {
        std::mem::take(&mut *self.0.lock().expect("lock"))
    }
}

impl AgentHook for Sequence {
    async fn on_run_start(&self, _ctx: &HookContext, _event: RunStart<'_>) -> RunStartAction {
        self.push("on_run_start");
        RunStartAction::Continue
    }

    async fn on_run_settled(&self, _ctx: &HookContext, _event: RunSettled<'_>) {
        self.push("on_run_settled");
    }

    fn on_model_select(
        &self,
        _ctx: &HookContext,
        _event: ModelSelection<'_>,
    ) -> ModelSelectionAction {
        self.push("on_model_select");
        ModelSelectionAction::Continue
    }

    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        self.push("on_completion_call");
        CompletionCallAction::Continue
    }

    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        _event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        self.push("on_model_turn_finished");
        ModelTurnAction::Continue
    }

    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        _event: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        self.push("on_invalid_tool_call");
        None
    }

    async fn on_text_delta(&self, _ctx: &HookContext, _event: TextDelta<'_>) -> ObservationAction {
        self.push("on_text_delta");
        ObservationAction::Continue
    }

    async fn on_reasoning_delta(
        &self,
        _ctx: &HookContext,
        _event: ReasoningDelta<'_>,
    ) -> ObservationAction {
        self.push("on_reasoning_delta");
        ObservationAction::Continue
    }

    async fn on_tool_call_delta(
        &self,
        _ctx: &HookContext,
        _event: ToolCallDelta<'_>,
    ) -> ObservationAction {
        self.push("on_tool_call_delta");
        ObservationAction::Continue
    }

    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        self.push(format!("on_dispatch({})", event.kind.name()));
        DispatchAction::Proceed
    }

    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        self.push(format!("on_outcome({})", event.kind.name()));
        OutcomeAction::Proceed
    }

    fn observes(&self, _kind: StepEventKind) -> bool {
        true
    }
}

fn strings(entries: &[&str]) -> Vec<String> {
    entries.iter().map(|entry| (*entry).to_owned()).collect()
}

#[tokio::test]
async fn hook_sequence_unary_turn_without_tools() {
    let sequence = Sequence::default();
    let agent = AgentBuilder::new(MockCompletionModel::text("done"))
        .add_hook(sequence.clone())
        .build();
    let response = within(agent.prompt("go").run()).await.expect("run");
    assert_eq!(response.output, "done");
    assert_eq!(
        sequence.take(),
        strings(&[
            "on_run_start",
            "on_completion_call",
            "on_model_select",
            "on_dispatch(completion)",
            "on_outcome(completion)",
            "on_model_turn_finished",
            "on_run_settled",
        ])
    );
}

#[tokio::test]
async fn hook_sequence_unary_turn_with_one_tool_call() {
    let sequence = Sequence::default();
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::tool_call("tc-1", "slow", json!({"delay_ms": 0, "tag": "t"})),
        MockTurn::text("done"),
    ]))
    .tool(Slow::default())
    .add_hook(sequence.clone())
    .build();
    let response = within(agent.prompt("go").max_turns(3).run())
        .await
        .expect("run");
    assert_eq!(response.output, "done");
    assert_eq!(
        sequence.take(),
        strings(&[
            "on_run_start",
            "on_completion_call",
            "on_model_select",
            "on_dispatch(completion)",
            "on_outcome(completion)",
            "on_model_turn_finished",
            "on_dispatch(tool_call)",
            "on_outcome(tool_call)",
            "on_completion_call",
            "on_model_select",
            "on_dispatch(completion)",
            "on_outcome(completion)",
            "on_model_turn_finished",
            "on_run_settled",
        ])
    );
}

#[tokio::test]
async fn hook_sequence_streaming_turn_with_one_tool_call() {
    let sequence = Sequence::default();
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        vec![
            rig_core::test_utils::MockStreamEvent::tool_call(
                "tc-1",
                "slow",
                json!({"delay_ms": 0, "tag": "t"}),
            ),
            rig_core::test_utils::MockStreamEvent::final_response_with_total_tokens(1),
        ],
        vec![
            rig_core::test_utils::MockStreamEvent::text("do"),
            rig_core::test_utils::MockStreamEvent::text("ne"),
            rig_core::test_utils::MockStreamEvent::final_response_with_total_tokens(1),
        ],
    ]))
    .tool(Slow::default())
    .add_hook(sequence.clone())
    .build();
    let mut stream = agent.stream_prompt("go").max_turns(3).stream().await;
    assert_eq!(drain(&mut stream).await.as_deref(), Some("done"));
    let recorded = sequence.take();
    // Deltas are provisional observations between a completion's dispatch
    // and its folded outcome; the lifecycle sequence is asserted without
    // them, and their placement separately.
    let lifecycle: Vec<String> = recorded
        .iter()
        .filter(|entry| !entry.ends_with("_delta"))
        .cloned()
        .collect();
    assert_eq!(
        lifecycle,
        strings(&[
            "on_run_start",
            "on_completion_call",
            "on_model_select",
            "on_dispatch(completion)",
            "on_outcome(completion)",
            "on_model_turn_finished",
            "on_dispatch(tool_call)",
            "on_outcome(tool_call)",
            "on_completion_call",
            "on_model_select",
            "on_dispatch(completion)",
            "on_outcome(completion)",
            "on_model_turn_finished",
            "on_run_settled",
        ])
    );
    let text_deltas: Vec<usize> = recorded
        .iter()
        .enumerate()
        .filter(|(_, entry)| *entry == "on_text_delta")
        .map(|(index, _)| index)
        .collect();
    assert_eq!(text_deltas.len(), 2, "{recorded:?}");
    let second_dispatch = recorded
        .iter()
        .rposition(|entry| entry == "on_dispatch(completion)")
        .expect("second completion dispatch");
    let second_outcome = recorded
        .iter()
        .rposition(|entry| entry == "on_outcome(completion)")
        .expect("second completion outcome");
    assert!(
        text_deltas
            .iter()
            .all(|index| *index > second_dispatch && *index < second_outcome),
        "deltas fire between the dispatch and its folded outcome: {recorded:?}"
    );
}

/// A retrieval index that always names the `slow` tool.
struct AlwaysSlow;

impl rig_core::vector_store::VectorStoreIndex for AlwaysSlow {
    type Filter = rig_core::vector_store::request::Filter<serde_json::Value>;

    async fn top_n<T: serde::de::DeserializeOwned + Send>(
        &self,
        _req: rig_core::vector_store::request::VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, rig_core::vector_store::VectorStoreError> {
        Ok(Vec::new())
    }

    async fn top_n_ids(
        &self,
        _req: rig_core::vector_store::request::VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, rig_core::vector_store::VectorStoreError> {
        Ok(vec![(1.0, "slow".to_owned())])
    }
}

#[tokio::test]
async fn hook_sequence_with_memory_and_tool_retrieval_when_a_hook_opts_in() {
    let sequence = Sequence::default();
    let agent = AgentBuilder::new(MockCompletionModel::text("done"))
        .memory(rig_core::memory::InMemoryConversationMemory::new())
        .conversation("c-1")
        .retrieved_tools(1, AlwaysSlow, ToolSet::from_tools(vec![Slow::default()]))
        .add_hook(sequence.clone())
        .build();
    let response = within(agent.prompt("go").run()).await.expect("run");
    assert_eq!(response.output, "done");
    assert_eq!(
        sequence.take(),
        strings(&[
            "on_dispatch(memory)",
            "on_outcome(memory)",
            "on_run_start",
            "on_completion_call",
            "on_model_select",
            "on_dispatch(retrieve)",
            "on_outcome(retrieve)",
            "on_dispatch(completion)",
            "on_outcome(completion)",
            "on_model_turn_finished",
            "on_dispatch(memory)",
            "on_outcome(memory)",
            "on_run_settled",
        ]),
        "the memory load precedes the run, tool retrieval precedes the request, the append precedes settlement"
    );
}

/// Opts into nothing extra: the internal families stay invisible.
#[derive(Clone, Default)]
struct DefaultObserver(Sequence);

impl AgentHook for DefaultObserver {
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        self.0.push(format!("on_dispatch({})", event.kind.name()));
        DispatchAction::Proceed
    }

    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        self.0.push(format!("on_outcome({})", event.kind.name()));
        OutcomeAction::Proceed
    }
}

#[tokio::test]
async fn internal_families_are_observe_only_unless_a_hook_opts_in() {
    let observer = DefaultObserver::default();
    let agent = AgentBuilder::new(MockCompletionModel::text("done"))
        .memory(rig_core::memory::InMemoryConversationMemory::new())
        .conversation("c-1")
        .add_hook(observer.clone())
        .record_effects()
        .build();
    let response = within(agent.prompt("go").run()).await.expect("run");
    assert_eq!(response.output, "done");
    assert_eq!(
        observer.0.take(),
        strings(&["on_dispatch(completion)", "on_outcome(completion)"]),
        "memory dispatches did not reach a hook that did not opt in"
    );
    let log = agent.effect_log().expect("recording");
    assert_eq!(
        log.iter()
            .map(|record| record.kind.name().to_owned())
            .collect::<Vec<_>>(),
        strings(&["memory", "completion", "memory"]),
        "…but the recorder saw every one of them"
    );
}

/// Stops the run from the completion's outcome.
struct StopOnAnswer;

impl AgentHook for StopOnAnswer {
    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        if event.completion().is_some() {
            OutcomeAction::stop("seen enough")
        } else {
            OutcomeAction::proceed()
        }
    }
}

#[tokio::test]
async fn a_cancelling_replacement_on_a_completion_outcome_stops_the_run_on_both_media() {
    let unary = AgentBuilder::new(MockCompletionModel::text("done"))
        .add_hook(StopOnAnswer)
        .build();
    let err = within(unary.prompt("go").run()).await.expect_err("stopped");
    assert!(
        matches!(
            err,
            rig_agent::run::response::PromptError::PromptCancelled { .. }
        ),
        "{err:?}"
    );

    let streaming = AgentBuilder::new(MockCompletionModel::from_stream_turns([streamed_text(
        "done",
    )]))
    .add_hook(StopOnAnswer)
    .build();
    let mut stream = streaming.stream_prompt("go").stream().await;
    let mut cancelled = false;
    while let Some(item) = within(stream.next()).await {
        if let Err(rig_agent::agent::StreamingError::Prompt(err)) = item
            && matches!(
                *err,
                rig_agent::run::response::PromptError::PromptCancelled { .. }
            )
        {
            cancelled = true;
        }
    }
    assert!(
        cancelled,
        "the streamed completion's outcome stopped the run"
    );
}

/// Replaces the streamed completion's content.
struct ReplaceAnswer;

impl AgentHook for ReplaceAnswer {
    async fn on_outcome(&self, ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        let Some(response) = event.completion() else {
            return OutcomeAction::proceed();
        };
        assert!(ctx.is_streaming(), "this test streams");
        let mut replaced = response.clone();
        replaced.choice = vec![rig_core::message::AssistantContent::text("replaced")];
        OutcomeAction::replace(Ok(rig_core::effect::Outcome::Completion(replaced)))
    }
}

#[tokio::test]
async fn a_replacement_on_a_streamed_completion_is_what_the_run_keeps() {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([streamed_text(
        "streamed",
    )]))
    .add_hook(ReplaceAnswer)
    .build();
    let mut stream = agent.stream_prompt("go").stream().await;
    assert_eq!(drain(&mut stream).await.as_deref(), Some("replaced"));
}

fn _assertions(agent: Agent, driver: BusDriver) {
    fn assert_send<T: Send>(_: &T) {}
    assert_send(&agent);
    assert_send(&driver);
}
