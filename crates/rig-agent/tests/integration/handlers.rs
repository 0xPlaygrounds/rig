//! The agent over its handler table: dispatch boundaries, nesting, ordering.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

use std::{
    sync::{Arc, Mutex, OnceLock, atomic::Ordering},
    time::Duration,
};

use futures::StreamExt;
use rig_agent::handlers::Handlers;
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
    effect::{EffectFamily, EffectKind, HandlerKey},
    error::ErrorKind,
    test_utils::{MockCompletionModel, MockTurn},
};
use serde::Deserialize;
use serde_json::json;

async fn within<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(5), future)
        .await
        .expect("a run never hangs")
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
async fn concurrent_tool_calls_finish_by_delay() {
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
async fn hook_sequence_with_tool_retrieval_when_a_hook_opts_in() {
    let sequence = Sequence::default();
    let agent = AgentBuilder::new(MockCompletionModel::text("done"))
        .retrieved_tools(1, AlwaysSlow, ToolSet::from_tools(vec![Slow::default()]))
        .add_hook(sequence.clone())
        .build();
    let response = within(agent.prompt("go").run()).await.expect("run");
    assert_eq!(response.output(), "done");
    assert_eq!(
        sequence.take(),
        strings(&[
            "on_run_start",
            "on_completion_call",
            "on_model_select",
            "on_dispatch(retrieve)",
            "on_outcome(retrieve)",
            "on_dispatch(completion)",
            "on_outcome(completion)",
            "on_model_turn_finished",
            "on_run_settled",
        ]),
        "tool retrieval precedes the request"
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
        OutcomeAction::replace(Ok(rig_core::effect::Answered::Completion(replaced.into())))
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

fn _assertions(agent: Agent, handlers: Handlers) {
    fn assert_send<T: Send>(_: &T) {}
    assert_send(&agent);
    assert_send(&handlers);
}

#[test]
fn attaching_a_table_preserves_the_latest_explicit_key_owner() {
    let handlers = Handlers::default();
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
    server.attach(&handlers);
    let descriptor = handlers.descriptor(&key).expect("latest binding");
    let rig_core::effect::FamilyDescriptor::Tool { name, .. } = descriptor.family else {
        panic!("expected tool descriptor");
    };
    assert_eq!(
        name, "first",
        "publication order, not name insertion order, owns the binding"
    );
}

#[tokio::test]
async fn a_host_key_serving_a_tool_fails_the_run_before_any_call() {
    let handlers = Handlers::default();
    handlers
        .register(
            "not-a-model",
            rig_core::serve::adapters::ToolAdapter::new(Slow::default()),
        )
        .expect("register");
    let agent =
        AgentBuilder::over_handlers(handlers, "guest", HandlerKey::from("not-a-model")).build();
    let error = within(agent.prompt("hello").run())
        .await
        .expect_err("a tool key is no model");
    let rig_agent::completion::PromptError::Report(report) = error else {
        panic!("expected a report, got {error}");
    };
    assert_eq!(report.kind, ErrorKind::HandlerUnavailable);
    assert!(report.message.contains("tool_call"), "{}", report.message);
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
/// the consumer, on every surface: a provider refusal through `run()`,
/// `run_channel()`, and a stream a consumer drops at its first `Err` (the
/// idiomatic `let item = item?;` loop, which this crate's own CLI chatbot
/// uses).
#[tokio::test]
async fn run_settled_fires_once_before_the_error_on_every_surface() {
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

    let agent = |hook: Settles| {
        let model = MockCompletionModel::from_turns([MockTurn::error("boom")]);
        AgentBuilder::new(model).add_hook(hook).build()
    };
    // Blocking.
    let hook = Settles::default();
    let error = within(agent(hook.clone()).prompt("go").run())
        .await
        .expect_err("the run fails");
    assert!(error.to_string().contains("boom"), "{error}");
    assert_eq!(hook.0.load(Ordering::SeqCst), 1, "run(): settled once");
    assert!(hook.1.lock().expect("lock")[0].contains("boom"));

    // Channelled: the future is the run; the events are dropped.
    let hook = Settles::default();
    let (future, events) = agent(hook.clone()).prompt("go").run_channel();
    drop(events);
    within(future).await.expect_err("the run fails");
    assert_eq!(
        hook.0.load(Ordering::SeqCst),
        1,
        "run_channel(): settled once"
    );

    // Streamed, dropped at the first `Err`.
    let hook = Settles::default();
    let mut stream = agent(hook.clone()).prompt("go").stream();
    let first = within(stream.next()).await.expect("an item");
    assert!(first.is_err(), "the first item is the ending");
    drop(stream);
    assert_eq!(
        hook.0.load(Ordering::SeqCst),
        1,
        "stream(): settled before the error was yielded"
    );
}
