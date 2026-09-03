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
use rig_bus::{Bus, BusConfig, BusDriver};
use rig_core::serve::adapters::CompletionAdapter;
use rig_core::{
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
        registrar: _,
        driver,
        agent,
    } = parts;
    assert!(!agent.owns_bus(), "the agent no longer drives");

    // Spawn the driver ourselves; the dispatcher clone and the agent both
    // resolve through it.
    let task = tokio::spawn(driver);
    let handle: rig_bus::ModelHandle = dispatcher
        .bind(agent.model_key())
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
    let (dispatcher, registrar, mut driver) = Bus::channel();
    driver
        .register(
            "model:host",
            CompletionAdapter::new("host", MockCompletionModel::text("never")),
        )
        .expect("register");
    let agent = AgentBuilder::over_bus(
        dispatcher,
        registrar,
        "guest",
        HandlerKey::from("model:host"),
    )
    .build();
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

/// Records which agent's registration a call reached.
#[derive(Clone)]
struct Tag {
    owner: &'static str,
    calls: Arc<Mutex<Vec<&'static str>>>,
}

impl Tool for Tag {
    const NAME: &'static str = "slow";
    type Args = SlowArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "answers with its owner".into()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {"delay_ms": {"type": "integer"}, "tag": {"type": "string"}}})
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: SlowArgs,
    ) -> Result<String, ToolExecutionError> {
        self.calls.lock().expect("lock").push(self.owner);
        Ok(self.owner.to_owned())
    }
}

#[tokio::test]
async fn two_agents_on_one_host_bus_keep_their_own_keys() {
    let (dispatcher, registrar, mut driver) = Bus::channel();
    let recorder = rig_bus::EffectLogRecorder::new();
    driver.record_to(recorder.clone());
    for label in ["left-host", "right-host"] {
        driver
            .register(
                format!("model:{label}"),
                CompletionAdapter::new(label, two_tool_calls_then_done()),
            )
            .expect("register");
    }
    let calls = Arc::new(Mutex::new(Vec::new()));
    let left = AgentBuilder::over_bus(
        dispatcher.clone(),
        registrar.clone(),
        "left",
        HandlerKey::from("model:left-host"),
    )
    .tool(Tag {
        owner: "left",
        calls: calls.clone(),
    })
    .build();
    let right = AgentBuilder::over_bus(
        dispatcher.clone(),
        registrar.clone(),
        "right",
        HandlerKey::from("model:right-host"),
    )
    .tool(Tag {
        owner: "right",
        calls: calls.clone(),
    })
    .build();
    assert_eq!(left.owner(), "left");
    assert_eq!(right.owner(), "right");
    let task = tokio::spawn(driver);

    let (left_response, right_response) = within(futures::future::join(
        left.prompt("go").max_turns(3).run(),
        right.prompt("go").max_turns(3).run(),
    ))
    .await;
    assert_eq!(left_response.expect("left").output, "done");
    assert_eq!(right_response.expect("right").output, "done");
    let mut reached = calls.lock().expect("lock").clone();
    reached.sort_unstable();
    assert_eq!(
        reached,
        ["left", "left", "right", "right"],
        "each agent's two calls reached its own registration"
    );

    // Two registries minted two owners for the same tool name; neither
    // retirement touched the other's key.
    let tool_keys: Vec<HandlerKey> = dispatcher
        .keys()
        .into_iter()
        .filter(|key| key.as_str().contains("/tool:slow#"))
        .collect();
    assert_eq!(
        tool_keys.len(),
        2,
        "one live key per registry: {tool_keys:?}"
    );
    let owners: std::collections::BTreeSet<&str> = tool_keys
        .iter()
        .map(|key| key.as_str().split('/').next().expect("owner segment"))
        .collect();
    assert_eq!(owners.len(), 2, "distinct owners: {tool_keys:?}");

    // The host's log names every key; each agent's tool records are under
    // that agent's registry only.
    let log = recorder.take();
    let recorded_tool_keys: std::collections::BTreeSet<HandlerKey> = log
        .iter()
        .filter(|record| record.kind.family() == EffectFamily::Tool)
        .map(|record| record.key.clone())
        .collect();
    assert_eq!(recorded_tool_keys.len(), 2, "{recorded_tool_keys:?}");

    drop((left, right, dispatcher));
    within(task).await.expect("driver task");
}

#[tokio::test]
async fn a_retired_tool_generation_leaves_the_bus_when_its_last_snapshot_drops() {
    let (dispatcher, registrar, driver) = Bus::channel();
    let server = rig_agent::tool::server::ToolServer::new()
        .tool(Slow::default())
        .run();
    server.attach(&registrar);
    let first_key = dispatcher
        .keys()
        .into_iter()
        .find(|key| key.as_str().contains("/tool:slow#"))
        .expect("the tool is published");

    let snapshot = server.snapshot();
    server.add_tool(Slow::default());
    assert!(
        dispatcher.keys().contains(&first_key),
        "the retired generation is served while a snapshot pins it"
    );
    drop(snapshot);
    assert!(
        !dispatcher.keys().contains(&first_key),
        "the last lease dropping deregisters the generation without a registry read"
    );
    assert_eq!(
        dispatcher
            .keys()
            .iter()
            .filter(|key| key.as_str().contains("/tool:slow#"))
            .count(),
        1
    );
    drop(driver);
}

#[tokio::test]
async fn registering_an_explicit_key_that_is_live_keeps_it_served() {
    let (dispatcher, registrar, driver) = Bus::channel();
    let server = rig_agent::tool::server::ToolServer::new().run();
    server.attach(&registrar);
    let registration = || {
        rig_agent::tool::RegisteredTool::from_tool(Slow::default()).with_key(
            rig_core::effect::Key::new_unchecked(HandlerKey::from("host/tool:slow")),
        )
    };
    server.add_registered_tool(registration());
    let snapshot = server.snapshot();
    server.add_registered_tool(registration());
    drop(snapshot);
    let key = HandlerKey::from("host/tool:slow");
    assert!(
        dispatcher.keys().contains(&key),
        "replacing a registration under its own key never deregisters the key"
    );
    assert!(dispatcher.descriptor(&key).is_some());
    drop(driver);
}

#[tokio::test]
async fn anonymous_models_are_scoped_to_the_values_that_selected_them() {
    let agent = AgentBuilder::new(MockCompletionModel::text("default")).build();
    let parts = agent
        .into_parts()
        .unwrap_or_else(|_| panic!("the only clone"));
    let rig_agent::agent::AgentParts {
        dispatcher,
        registrar: _,
        driver,
        agent,
    } = parts;
    let task = tokio::spawn(driver);
    let anonymous = |dispatcher: &rig_bus::Dispatcher| {
        dispatcher
            .keys()
            .into_iter()
            .filter(|key| key.as_str().contains("/model:anonymous#"))
            .count()
    };

    for _ in 0..1_000 {
        let runner = agent
            .runner("hello")
            .using_model_value(MockCompletionModel::text("anonymous"));
        assert_eq!(
            anonymous(&dispatcher),
            1,
            "one live registration while a runner selects it"
        );
        let response = within(runner.run()).await.expect("run");
        assert_eq!(response.output, "anonymous");
    }
    assert_eq!(
        anonymous(&dispatcher),
        0,
        "a finished run's registration is gone"
    );

    let mut swapped = agent.clone();
    swapped.set_model(MockCompletionModel::text("swapped"));
    let clone = swapped.clone();
    assert_eq!(anonymous(&dispatcher), 1);
    drop(swapped);
    assert_eq!(anonymous(&dispatcher), 1, "a clone still selects it");
    let response = within(clone.prompt("hello").run()).await.expect("run");
    assert_eq!(response.output, "swapped");
    drop(clone);
    assert_eq!(
        anonymous(&dispatcher),
        0,
        "the last value dropping deregisters"
    );

    let response = within(agent.prompt("hello").run()).await.expect("run");
    assert_eq!(
        response.output, "default",
        "the agent's own default is untouched"
    );
    drop((agent, dispatcher));
    within(task).await.expect("driver task");
}

#[tokio::test]
async fn recording_survives_into_parts() {
    let agent = AgentBuilder::new(MockCompletionModel::text("recorded"))
        .record_effects()
        .build();
    let parts = agent
        .into_parts()
        .unwrap_or_else(|_| panic!("the only clone"));
    let rig_agent::agent::AgentParts {
        dispatcher,
        registrar: _,
        driver,
        agent,
    } = parts;
    let task = tokio::spawn(driver);
    let response = within(agent.prompt("hello").run()).await.expect("run");
    assert_eq!(response.output, "recorded");
    let log = agent.effect_log().expect("the agent still records");
    assert_eq!(
        log.len(),
        1,
        "the moved driver records into the agent's log"
    );
    assert_eq!(log[0].kind.family(), EffectFamily::Completion);
    drop((agent, dispatcher));
    within(task).await.expect("driver task");
}

#[tokio::test]
async fn a_bus_failure_on_a_tool_dispatch_is_a_run_error_not_a_tool_result() {
    let (dispatcher, registrar, mut driver) = Bus::channel();
    driver
        .register(
            "model:host",
            CompletionAdapter::new("host", two_tool_calls_then_done()),
        )
        .expect("register");
    let agent = AgentBuilder::over_bus(
        dispatcher.clone(),
        registrar.clone(),
        "guest",
        HandlerKey::from("model:host"),
    )
    .tool(Slow::default())
    .build();
    let task = tokio::spawn(driver);
    // The host pulls the tool's handler out from under the registry: the
    // catalog still advertises it, the bus cannot serve it.
    let tool_key = dispatcher
        .keys()
        .into_iter()
        .find(|key| key.as_str().contains("/tool:slow#"))
        .expect("the tool is published");
    assert!(registrar.deregister(&tool_key));

    let error = within(agent.prompt("go").max_turns(3).run())
        .await
        .expect_err("the run fails");
    match error {
        rig_agent::completion::PromptError::Report(report) => {
            assert_eq!(report.kind, ErrorKind::HandlerUnavailable, "{report}");
        }
        other => panic!("expected the bus report, got {other:?}"),
    }
    drop((agent, dispatcher));
    within(task).await.expect("driver task");
}

#[tokio::test]
async fn a_streamed_completion_names_its_provider_like_a_unary_one() {
    let agent = AgentBuilder::new(MockCompletionModel::text("hello"))
        .model_route(
            "streamer",
            MockCompletionModel::from_stream_turns([[
                rig_core::test_utils::MockStreamEvent::text("hello"),
                rig_core::test_utils::MockStreamEvent::final_response_with_default_usage(),
            ]]),
        )
        .build();
    let parts = agent
        .into_parts()
        .unwrap_or_else(|_| panic!("the only clone"));
    let rig_agent::agent::AgentParts {
        dispatcher,
        registrar: _,
        driver,
        agent,
    } = parts;
    let task = tokio::spawn(driver);
    let model: rig_bus::ModelHandle = dispatcher
        .bind(agent.model_key())
        .expect("the model is registered");
    let streamer: rig_bus::ModelHandle = dispatcher
        .handle(&HandlerKey::from(format!(
            "{}/model:streamer",
            agent.owner()
        )))
        .expect("the route is registered under the agent's owner");
    let request = |text: &str| rig_core::completion::CompletionRequest {
        model: None,
        chat_history: vec![rig_core::message::Message::user(text)],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let unary = within(model.complete(request("hi"))).await.expect("unary");
    let mut stream = streamer.stream(request("hi"));
    assert_eq!(
        stream.provider(),
        "streamer",
        "before the terminal record the stream carries the handler's label"
    );
    while let Some(item) = within(stream.next()).await {
        item.expect("a clean stream");
    }
    let streamed = stream.finish();
    assert_eq!(
        streamed.provider, unary.provider,
        "the terminal record names the provider"
    );
    assert_eq!(streamed.choice, unary.choice);
    drop((agent, dispatcher, model, streamer));
    within(task).await.expect("driver task");
}

#[tokio::test]
async fn a_model_registered_through_the_parts_registrar_serves_the_next_run() {
    let agent = AgentBuilder::new(two_tool_calls_then_done())
        .tool(Slow::default())
        .build();
    let parts = agent
        .into_parts()
        .unwrap_or_else(|_| panic!("the only clone"));
    let rig_agent::agent::AgentParts {
        dispatcher,
        registrar,
        driver,
        agent,
    } = parts;
    let task = tokio::spawn(driver);

    // A run is in flight on the moved-out driver (its tool sleeps); the host
    // registers another model through the registrar meanwhile.
    let in_flight = tokio::spawn({
        let agent = agent.clone();
        async move { agent.prompt("go").max_turns(3).run().await }
    });
    tokio::time::sleep(Duration::from_millis(10)).await;
    let key = HandlerKey::from(format!("{}/model:late", agent.owner()));
    registrar
        .register(
            key.clone(),
            CompletionAdapter::new("late", MockCompletionModel::text("late")),
        )
        .expect("a fresh key");
    assert!(
        dispatcher.descriptor(&key).is_some(),
        "the descriptor is visible at once"
    );
    let response = within(agent.runner("hello").using_model("late").run())
        .await
        .expect("the next run selects it");
    assert_eq!(response.output, "late");
    let first = within(in_flight)
        .await
        .expect("join")
        .expect("the in-flight run");
    assert_eq!(first.output, "done");
    drop((agent, dispatcher, registrar));
    within(task).await.expect("driver task");
}

/// A `MakeWriter` that keeps what the subscriber wrote.
#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Captured {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("lock").extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Captured {
    type Writer = Captured;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

#[test]
fn a_host_key_that_serves_another_family_is_reported_at_the_hosts_line() {
    let (dispatcher, registrar, mut driver) = Bus::channel();
    driver
        .register(
            "not-a-model",
            rig_core::serve::adapters::ToolAdapter::new(Slow::default()),
        )
        .expect("register");
    let captured = Captured::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(captured.clone())
        .with_ansi(false)
        .with_max_level(tracing::Level::ERROR)
        .finish();
    let expected_line = line!() + 2;
    let _agent = tracing::subscriber::with_default(subscriber, || {
        AgentBuilder::over_bus(
            dispatcher,
            registrar,
            "guest",
            HandlerKey::from("not-a-model"),
        )
        .build()
    });
    drop(driver);
    let logged = String::from_utf8(captured.0.lock().expect("lock").clone()).expect("utf8");
    assert!(
        logged.contains("does not serve a completion model"),
        "the build reported the key: {logged}"
    );
    assert!(
        logged.contains(&format!("effect_bus.rs:{expected_line}")),
        "the report names the host's line, not the builder's: {logged}"
    );
}

/// Binds a run-scoped view inside the hook body and dispatches through it
/// (the `dynamic_context` shape): the view lives for the hook call only.
struct AsksTheModel {
    key: rig_core::effect::Key<rig_core::effect::family::Completion>,
    seen: Arc<Mutex<Vec<String>>>,
}

impl AgentHook for AsksTheModel {
    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let model = ctx.bind(&self.key).expect("bound for this run");
        let request = rig_core::completion::CompletionRequest {
            model: None,
            chat_history: vec![rig_core::message::Message::user("side question")],
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };
        let answer = model
            .complete(request)
            .await
            .expect("the side model answers");
        self.seen.lock().expect("lock").push(
            answer
                .choice
                .iter()
                .filter_map(|content| match content {
                    rig_core::message::AssistantContent::Text(text) => Some(text.text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        );
        CompletionCallAction::continue_run()
    }
}

#[tokio::test]
async fn a_hook_binds_a_run_scoped_view_and_dispatches_through_it() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let agent = AgentBuilder::new(MockCompletionModel::text("main answer"))
        .model_route("side", MockCompletionModel::text("side answer"))
        .build();
    let key = rig_core::effect::Key::new_unchecked(HandlerKey::from(format!(
        "{}/model:side",
        agent.owner()
    )));
    let response = within(
        agent
            .runner("hello")
            .add_hook(AsksTheModel {
                key,
                seen: seen.clone(),
            })
            .run(),
    )
    .await
    .expect("run");
    assert_eq!(response.output, "main answer");
    assert_eq!(*seen.lock().expect("lock"), vec!["side answer".to_string()]);
}
