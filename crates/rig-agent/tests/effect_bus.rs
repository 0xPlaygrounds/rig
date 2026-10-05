//! The agent over the effect bus: ownership, ordering, record and replay.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

use std::{
    sync::{Arc, Mutex, OnceLock, atomic::Ordering},
    time::Duration,
};

use futures::StreamExt;
use rig_agent::bus::{Bus, BusDriver};
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
use rig_core::serve::ServingPolicy;
use rig_core::serve::adapters::ModelAdapter;
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
                    rig_core::message::ToolName::new("slow".to_owned()).expect("tool name"),
                    json!({"delay_ms": 40, "tag": "first"}),
                ),
            )),
            rig_core::message::AssistantContent::ToolCall(rig_core::message::ToolCall::from_wire(
                "tc-2",
                rig_core::message::ToolFunction::new(
                    rig_core::message::ToolName::new("slow".to_owned()).expect("tool name"),
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
    let agent = AgentBuilder::named_model("default", two_tool_calls_then_done())
        .configure_bus(ServingPolicy {
            serial_per_handler: true,
            ..ServingPolicy::default()
        })
        .tool(serial.clone())
        .build();
    let response = within(agent.prompt("go").max_turns(3).tool_concurrency(2).run())
        .await
        .expect("run");
    assert_eq!(response.output(), "done");
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
    assert_eq!(response.output(), "done");
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
    let handle: rig_agent::bus::ModelHandle = dispatcher
        .bind(agent.model_key())
        .expect("the model is registered");
    assert_eq!(handle.label().as_str(), "default");
    let response = within(agent.prompt("hello").run())
        .await
        .expect("served by the spawned driver");
    assert_eq!(response.output(), "parts");
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
    assert_eq!(response.output(), "shared");
    drop(clone);
}

#[tokio::test]
async fn a_run_over_a_dropped_host_bus_answers_bus_closed_not_a_hang() {
    let (dispatcher, registrar, mut driver) = Bus::channel();
    driver
        .register(
            "model:host",
            ModelAdapter::new("host", MockCompletionModel::text("never")),
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
            EffectKind::ToolCall { name, .. } if name == "slow" => {
                DispatchAction::patch(EffectKind::ToolCall {
                    name: name.clone(),
                    args: json!({"delay_ms": 0, "tag": "patched"}).to_string(),
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
    assert_eq!(response.output(), "done");
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

#[tokio::test]
async fn two_streams_polled_alternately_both_complete() {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        streamed_text("one"),
        streamed_text("two"),
    ]))
    .build();
    let mut first = agent.prompt("a").stream();
    let mut second = agent.prompt("b").stream();
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
        Ok(response.output())
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
                rig_core::message::ToolFunction::new(
                    rig_core::message::ToolName::new("nested".to_owned()).expect("tool name"),
                    json!({}),
                ),
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
    assert_eq!(response.output(), "done");
}

/// A tool that, from inside its own execution, runs a nested prompt whose
/// model calls this same tool again. The nested agent is built over the
/// call's scope (`ToolContext::scope`, a dispatcher parented by this
/// call), so its dispatches descend from the outer call: causality as
/// data, which is what the serial re-entrancy rule reads.
#[derive(Clone, Default)]
struct NestedSameTool {
    host: Arc<
        OnceLock<(
            rig_agent::bus::Registrar,
            HandlerKey,
            rig_agent::tool::server::ToolServerHandle,
        )>,
    >,
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
        context: &mut ToolContext,
        _args: serde_json::Value,
    ) -> Result<String, Self::Error> {
        let (registrar, model_key, tools) = self.host.get().expect("set after build").clone();
        let scoped = context
            .scope::<rig_agent::bus::Dispatcher>()
            .expect("served over a bus: the call has a scope");
        assert!(
            scoped.parent().is_some(),
            "the scope is parented by this call"
        );
        let nested = AgentBuilder::over_bus((*scoped).clone(), registrar, "golden", model_key)
            .name("golden")
            .tool_server_handle(tools)
            .build();
        let response = nested
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
            .push(response.output());
        Ok(response.output())
    }
}

/// A tool that, from inside its own execution, prompts the *same* agent
/// again on the agent's own bus: nothing ties that run to the call.
#[derive(Clone, Default)]
struct NestedOnItself {
    agent: Arc<OnceLock<Agent>>,
}

impl Tool for NestedOnItself {
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
        Ok(response.output())
    }
}

#[tokio::test]
async fn a_nested_call_to_the_in_flight_tool_under_serial_serving_fails_fast() {
    // The shape the thread-id rule used to refuse: a tool prompting the
    // same own-bus agent from inside its own call, under serial serving.
    // Re-entrancy is a chain now, and this run carries no parent — nothing
    // ties it to the call — so the nested call to `same` queues behind the
    // outer call that waits on it: the run does not complete. Pinned as the
    // documented behaviour; the test below is the shape that fails fast.
    let tool = NestedOnItself::default();
    let call_same = || {
        MockTurn::from_contents([rig_core::message::AssistantContent::ToolCall(
            rig_core::message::ToolCall::from_wire(
                "tc",
                rig_core::message::ToolFunction::new(
                    rig_core::message::ToolName::new("same".to_owned()).expect("tool name"),
                    json!({}),
                ),
            ),
        )])
    };
    let agent = AgentBuilder::named_model(
        "default",
        MockCompletionModel::from_turns([
            call_same(),
            call_same(),
            MockTurn::text("inner-done"),
            MockTurn::text("done"),
        ]),
    )
    .configure_bus(ServingPolicy {
        serial_per_handler: true,
        ..ServingPolicy::default()
    })
    .tool(tool.clone())
    .build();
    tool.agent.set(agent.clone()).ok().expect("unset");
    let waited = tokio::time::timeout(
        Duration::from_millis(300),
        agent.prompt("go").max_turns(3).run(),
    )
    .await;
    assert!(
        waited.is_err(),
        "a nested run on the same own-bus agent under serial serving waits on itself: {waited:?}"
    );
}

#[tokio::test]
async fn a_nested_call_over_the_calls_scope_under_serial_serving_fails_fast() {
    // Outer turn: call `same`. Inside it the nested run's model calls `same`
    // again — under serial serving that would queue behind the outer call
    // that waits on it, so the bus refuses it (the nested run descends from
    // the outer call), the nested model sees a skipped tool result,
    // answers, and the outer run completes.
    let tool = NestedSameTool::default();
    let call_same = || {
        MockTurn::from_contents([rig_core::message::AssistantContent::ToolCall(
            rig_core::message::ToolCall::from_wire(
                "tc",
                rig_core::message::ToolFunction::new(
                    rig_core::message::ToolName::new("same".to_owned()).expect("tool name"),
                    json!({}),
                ),
            ),
        )])
    };
    let (dispatcher, registrar, mut driver) = Bus::channel_with(ServingPolicy {
        serial_per_handler: true,
        ..ServingPolicy::default()
    });
    let model_key = HandlerKey::from("golden/model:default");
    driver
        .register_erased(
            model_key.clone(),
            rig_core::serve::ErasedHandler::new(ModelAdapter::new(
                "default",
                MockCompletionModel::from_turns([
                    call_same(),
                    call_same(),
                    MockTurn::text("inner-done"),
                    MockTurn::text("done"),
                ]),
            )),
        )
        .expect("a fresh key");
    let driving = tokio::spawn(driver);
    let agent = AgentBuilder::over_bus(dispatcher, registrar.clone(), "golden", model_key.clone())
        .name("golden")
        .tool(tool.clone())
        .build();
    tool.host
        .set((registrar, model_key, agent.tool_server_handle().clone()))
        .ok()
        .expect("unset");
    let response = within(agent.prompt("go").max_turns(3).run())
        .await
        .expect("the outer run completes: the re-entrant call was refused, not queued");
    assert_eq!(response.output(), "done");
    assert_eq!(
        *tool.inner_outputs.lock().expect("lock"),
        vec!["inner-done".to_string()]
    );
    drop((agent, tool));
    within(driving).await.expect("the driver ends");
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

/// A retrieval index that always names the `slow` tool.
struct AlwaysSlow;

impl rig_core::vector_store::VectorStoreIndex for AlwaysSlow {
    type Filter = rig_core::vector_store::request::Filter<serde_json::Value>;

    async fn top_n<T: serde::de::DeserializeOwned + Send>(
        &self,
        _req: rig_core::vector_store::request::VectorSearchRequest<Self::Filter>,
    ) -> Result<
        Vec<rig_core::vector_store::VectorSearchResult<T>>,
        rig_core::vector_store::VectorStoreError,
    > {
        Ok(Vec::new())
    }

    async fn top_n_ids(
        &self,
        _req: rig_core::vector_store::request::VectorSearchRequest<Self::Filter>,
    ) -> Result<
        Vec<rig_core::vector_store::VectorSearchIdResult>,
        rig_core::vector_store::VectorStoreError,
    > {
        Ok(vec![rig_core::vector_store::VectorSearchIdResult {
            score: 1.0,
            id: "slow".to_owned(),
        }])
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
    assert_eq!(response.output(), "done");
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
    let mut stream = agent.prompt("go").stream();
    assert_eq!(drain(&mut stream).await.as_deref(), Some("replaced"));
}

fn _assertions(agent: Agent, driver: BusDriver) {
    fn assert_send<T: Send>(_: &T) {}
    assert_send(&agent);
    assert_send(&driver);
}

#[test]
fn a_reused_explicit_key_waits_for_its_latest_retired_snapshot() {
    for old_first in [true, false] {
        let (dispatcher, registrar, driver) = Bus::channel();
        let server = rig_agent::tool::server::ToolServer::new().run();
        server.attach(&registrar);
        let key = HandlerKey::from("host/tool:slow");
        let registration = || {
            rig_agent::tool::RegisteredTool::from_tool(Slow::default())
                .with_key(rig_core::effect::Key::new_unchecked(key.clone()))
        };
        server.add_registered_tool(registration());
        let old = server.snapshot();
        server.remove_tool("slow");
        server.add_registered_tool(registration());
        let replacement = server.snapshot();
        server.remove_tool("slow");
        if old_first {
            drop(old);
            assert!(
                dispatcher.descriptor(&key).is_some(),
                "the latest retired lease still pins this key"
            );
            drop(replacement);
        } else {
            drop(replacement);
            assert!(
                dispatcher.descriptor(&key).is_none(),
                "an obsolete lease cannot extend the replacement's lifetime"
            );
            drop(old);
        }
        assert!(
            dispatcher.descriptor(&key).is_none(),
            "the last current lease removes its key"
        );
        drop(driver);
    }
}

#[test]
fn attaching_a_bus_preserves_the_latest_explicit_key_owner() {
    let (dispatcher, registrar, driver) = Bus::channel();
    let server = rig_agent::tool::server::ToolServer::new().run();
    let key = HandlerKey::from("host/tool:shared");
    for name in ["first", "second", "first"] {
        let tool = rig_core::tool::DynamicTool::new(
            rig_core::message::ToolName::new(name).expect("tool name"),
            "test binding",
            json!({"type": "object"}),
            |_| Box::pin(async { Ok(rig_core::tool::ToolOutput::text("answer")) }),
        );
        server.add_registered_tool(
            rig_agent::tool::RegisteredTool::from_dynamic(tool)
                .with_key(rig_core::effect::Key::new_unchecked(key.clone())),
        );
    }
    server.attach(&registrar);
    let descriptor = dispatcher.descriptor(&key).expect("latest binding");
    let rig_core::effect::FamilyDescriptor::Tool { name, .. } = descriptor.family else {
        panic!("expected tool descriptor");
    };
    assert_eq!(
        name, "first",
        "publication order, not name insertion order, owns the binding"
    );
    drop(driver);
}

#[test]
fn a_host_key_serving_a_tool_fails_at_build_at_the_hosts_line() {
    let (dispatcher, registrar, mut driver) = Bus::channel();
    driver
        .register(
            "not-a-model",
            rig_core::serve::adapters::ToolAdapter::new(Slow::default()),
        )
        .expect("register");
    let expected_line = line!() + 2;
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        AgentBuilder::over_bus(
            dispatcher,
            registrar,
            "guest",
            HandlerKey::from("not-a-model"),
        )
        .build()
    }));
    drop(driver);
    let payload = match built {
        Err(payload) => payload,
        Ok(_) => panic!("a host key of another family must fail at build"),
    };
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .expect("a message");
    assert!(
        message.contains("serves the tool_call family, not a completion model"),
        "the build named the key's family: {message}"
    );
    assert!(
        message.contains(&format!("effect_bus.rs:{expected_line}")),
        "the failure names the host's line, not the builder's: {message}"
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
        let request = rig_core::completion::CompletionRequest::new("side question");
        let answer = model.call(request).await.expect("the side model answers");
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
            .prompt("hello")
            .add_hook(AsksTheModel {
                key,
                seen: seen.clone(),
            })
            .run(),
    )
    .await
    .expect("run");
    assert_eq!(response.output(), "main answer");
    assert_eq!(*seen.lock().expect("lock"), vec!["side answer".to_string()]);
}

/// `RunSettled` fires exactly once per run, before the run's error reaches
/// the consumer, on every surface and for every error ending: a provider
/// refusal and a memory load that fails before the engine starts, through
/// `run()`, `run_channel()`, and a stream a consumer drops at its first
/// `Err` (the idiomatic `let item = item?;` loop, which this crate's own
/// CLI chatbot uses).
#[tokio::test]
async fn run_settled_fires_once_before_every_error_ending_on_every_surface() {
    use std::sync::atomic::AtomicUsize;

    #[derive(Clone, Default)]
    struct Settles(Arc<AtomicUsize>, Arc<Mutex<Vec<String>>>);
    impl AgentHook for Settles {
        async fn on_run_settled(&self, _ctx: &HookContext, event: RunSettled<'_>) {
            self.0.fetch_add(1, Ordering::SeqCst);
            if let rig_agent::agent::SettledOutcome::Error(reason) = event.outcome {
                self.1.lock().expect("lock").push(reason.to_owned());
            }
        }
    }

    enum Ending {
        Provider,
        MemoryLoad,
    }
    let agent = |ending: &Ending, hook: Settles| {
        let model = MockCompletionModel::from_turns([MockTurn::error("boom")]);
        let builder = AgentBuilder::new(model).add_hook(hook);
        match ending {
            Ending::Provider => builder.build(),
            Ending::MemoryLoad => builder
                .memory(rig_core::test_utils::FailingMemory::new("load boom"))
                .conversation("settled-once")
                .build(),
        }
    };
    let expected = |ending: &Ending| match ending {
        Ending::Provider => "boom",
        Ending::MemoryLoad => "load boom",
    };

    for ending in [Ending::Provider, Ending::MemoryLoad] {
        // Blocking.
        let hook = Settles::default();
        let error = within(agent(&ending, hook.clone()).prompt("go").run())
            .await
            .expect_err("the run fails");
        assert!(error.to_string().contains(expected(&ending)), "{error}");
        assert_eq!(hook.0.load(Ordering::SeqCst), 1, "run(): settled once");
        assert!(hook.1.lock().expect("lock")[0].contains(expected(&ending)));

        // Channelled: the future is the run; the events are dropped.
        let hook = Settles::default();
        let (future, events) = agent(&ending, hook.clone()).prompt("go").run_channel();
        drop(events);
        within(future).await.expect_err("the run fails");
        assert_eq!(
            hook.0.load(Ordering::SeqCst),
            1,
            "run_channel(): settled once"
        );

        // Streamed, dropped at the first `Err`.
        let hook = Settles::default();
        let mut stream = agent(&ending, hook.clone()).prompt("go").stream();
        let first = within(stream.next()).await.expect("an item");
        assert!(first.is_err(), "the first item is the ending");
        drop(stream);
        assert_eq!(
            hook.0.load(Ordering::SeqCst),
            1,
            "stream(): settled before the error was yielded"
        );
    }
}
