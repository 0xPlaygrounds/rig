use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use super::*;
use serde_json::{Value, json};

fn ctx() -> HookContext {
    HookContext::new(false, Some("test-agent".to_string()))
}

fn model(label: &str) -> ModelHandle {
    ModelHandle::named(label, crate::test_utils::MockCompletionModel::default())
}

enum RouteDecision {
    Continue,
    Select(ModelHandle),
    Stop,
}

type RouteLog = Arc<Mutex<Vec<(&'static str, Option<String>)>>>;

struct RouteRecorder {
    label: &'static str,
    log: RouteLog,
    decision: RouteDecision,
}

impl AgentHook for RouteRecorder {
    fn on_model_select(
        &self,
        _ctx: &HookContext,
        event: ModelSelection<'_>,
    ) -> ModelSelectionAction {
        self.log
            .lock()
            .expect("route log")
            .push((self.label, event.selected_model.label().map(str::to_owned)));
        match &self.decision {
            RouteDecision::Continue => ModelSelectionAction::continue_run(),
            RouteDecision::Select(model) => ModelSelectionAction::select(model.clone()),
            RouteDecision::Stop => ModelSelectionAction::stop("routing stopped"),
        }
    }
}

fn model_selection<'a>(prompt: &'a Message, default_model: &'a ModelHandle) -> ModelSelection<'a> {
    ModelSelection {
        prompt,
        history: &[],
        request_patch: None,
        previous_model: None,
        default_model,
        selected_model: default_model,
    }
}

#[test]
fn model_selections_chain_in_registration_order_and_last_wins() {
    let default = model("default");
    let first = model("first");
    let last = model("last");
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut stack = HookStack::with(RouteRecorder {
        label: "continue",
        log: log.clone(),
        decision: RouteDecision::Continue,
    });
    stack.push(RouteRecorder {
        label: "first",
        log: log.clone(),
        decision: RouteDecision::Select(first),
    });
    stack.push(RouteRecorder {
        label: "last",
        log: log.clone(),
        decision: RouteDecision::Select(last),
    });
    let prompt = Message::user("route");

    let action = stack.on_model_select(&ctx(), model_selection(&prompt, &default));

    let ModelSelectionAction::Select(selected) = action else {
        panic!("stack should select the last candidate");
    };
    assert_eq!(selected.label(), Some("last"));
    assert_eq!(
        log.lock().expect("route log").as_slice(),
        &[
            ("continue", Some("default".to_owned())),
            ("first", Some("default".to_owned())),
            ("last", Some("first".to_owned())),
        ]
    );
}

#[test]
fn model_selection_stop_short_circuits_later_hooks() {
    let default = model("default");
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut stack = HookStack::with(RouteRecorder {
        label: "stop",
        log: log.clone(),
        decision: RouteDecision::Stop,
    });
    stack.push(RouteRecorder {
        label: "later",
        log: log.clone(),
        decision: RouteDecision::Select(model("later")),
    });
    let prompt = Message::user("route");

    assert!(matches!(
        stack.on_model_select(&ctx(), model_selection(&prompt, &default)),
        ModelSelectionAction::Stop(reason) if reason == "routing stopped"
    ));
    assert_eq!(
        log.lock().expect("route log").as_slice(),
        &[("stop", Some("default".to_owned()))]
    );
}

#[test]
fn nested_model_selection_stacks_preserve_candidate_chaining() {
    let default = model("default");
    let log = Arc::new(Mutex::new(Vec::new()));
    let inner = HookStack::with(RouteRecorder {
        label: "inner",
        log: log.clone(),
        decision: RouteDecision::Select(model("inner")),
    });
    let mut outer = HookStack::with(RouteRecorder {
        label: "outer-before",
        log: log.clone(),
        decision: RouteDecision::Select(model("outer")),
    });
    outer.push(inner);
    outer.push(RouteRecorder {
        label: "outer-after",
        log: log.clone(),
        decision: RouteDecision::Continue,
    });
    let prompt = Message::user("route");

    let action = outer.on_model_select(&ctx(), model_selection(&prompt, &default));

    let ModelSelectionAction::Select(selected) = action else {
        panic!("nested stack should preserve the inner selection");
    };
    assert_eq!(selected.label(), Some("inner"));
    assert_eq!(
        log.lock().expect("route log").as_slice(),
        &[
            ("outer-before", Some("default".to_owned())),
            ("inner", Some("outer".to_owned())),
            ("outer-after", Some("inner".to_owned())),
        ]
    );
}

#[test]
fn nested_model_selection_stack_without_a_selection_preserves_outer_candidate() {
    let default = model("default");
    let log = Arc::new(Mutex::new(Vec::new()));
    let inner = HookStack::with(RouteRecorder {
        label: "inner-continue",
        log: log.clone(),
        decision: RouteDecision::Continue,
    });
    let mut outer = HookStack::with(RouteRecorder {
        label: "outer-select",
        log: log.clone(),
        decision: RouteDecision::Select(model("outer")),
    });
    outer.push(inner);
    outer.push(RouteRecorder {
        label: "outer-after",
        log: log.clone(),
        decision: RouteDecision::Continue,
    });
    let prompt = Message::user("route");

    let action = outer.on_model_select(&ctx(), model_selection(&prompt, &default));

    let ModelSelectionAction::Select(selected) = action else {
        panic!("outer selection should survive a continuing nested stack");
    };
    assert_eq!(selected.label(), Some("outer"));
    assert_eq!(
        log.lock().expect("route log").as_slice(),
        &[
            ("outer-select", Some("default".to_owned())),
            ("inner-continue", Some("outer".to_owned())),
            ("outer-after", Some("outer".to_owned())),
        ]
    );
}

#[test]
fn nested_model_selection_stop_short_circuits_the_outer_stack() {
    let default = model("default");
    let log = Arc::new(Mutex::new(Vec::new()));
    let inner = HookStack::with(RouteRecorder {
        label: "inner-stop",
        log: log.clone(),
        decision: RouteDecision::Stop,
    });
    let mut outer = HookStack::with(RouteRecorder {
        label: "outer-before",
        log: log.clone(),
        decision: RouteDecision::Select(model("outer")),
    });
    outer.push(inner);
    outer.push(RouteRecorder {
        label: "outer-after",
        log: log.clone(),
        decision: RouteDecision::Select(model("unreachable")),
    });
    let prompt = Message::user("route");

    assert!(matches!(
        outer.on_model_select(&ctx(), model_selection(&prompt, &default)),
        ModelSelectionAction::Stop(reason) if reason == "routing stopped"
    ));
    assert_eq!(
        log.lock().expect("route log").as_slice(),
        &[
            ("outer-before", Some("default".to_owned())),
            ("inner-stop", Some("outer".to_owned())),
        ]
    );
}

struct ToolRecorder {
    label: u32,
    log: Arc<Mutex<Vec<u32>>>,
    stop: bool,
}
impl AgentHook for ToolRecorder {
    async fn on_tool_call(&self, _ctx: &HookContext, _event: ToolCall<'_>) -> ToolCallAction {
        self.log.lock().expect("log").push(self.label);
        if self.stop {
            ToolCallAction::stop("stop")
        } else {
            ToolCallAction::run()
        }
    }
}

struct ObservationRecorder {
    label: u32,
    log: Arc<Mutex<Vec<u32>>>,
    stop: bool,
}
impl AgentHook for ObservationRecorder {
    async fn on_text_delta(&self, _ctx: &HookContext, _event: TextDelta<'_>) -> ObservationAction {
        self.log.lock().expect("log").push(self.label);
        if self.stop {
            ObservationAction::stop("stop")
        } else {
            ObservationAction::continue_run()
        }
    }

    async fn on_reasoning_delta(
        &self,
        _ctx: &HookContext,
        _event: ReasoningDelta<'_>,
    ) -> ObservationAction {
        self.log.lock().expect("log").push(self.label);
        if self.stop {
            ObservationAction::stop("stop")
        } else {
            ObservationAction::continue_run()
        }
    }
}

struct ObservesOnly(StepEventKind);
impl AgentHook for ObservesOnly {
    fn observes(&self, kind: StepEventKind) -> bool {
        kind == self.0
    }
}

struct InvalidResponder {
    action: InvalidToolCallAction,
    calls: Arc<AtomicUsize>,
}
impl AgentHook for InvalidResponder {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        _event: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Some(self.action.clone())
    }
}

struct Patcher {
    label: u32,
    log: Arc<Mutex<Vec<u32>>>,
    patch: RequestPatch,
    stop: bool,
}
impl AgentHook for Patcher {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCall<'_>,
    ) -> CompletionCallAction {
        self.log.lock().expect("log").push(self.label);
        if self.stop {
            CompletionCallAction::stop("stop")
        } else {
            CompletionCallAction::patch(self.patch.clone())
        }
    }
}

fn tool_call_event() -> ToolCall<'static> {
    ToolCall {
        tool_name: "add",
        tool_call_id: Some("tc1"),
        internal_call_id: InternalCallId::new(),
        args: "{}",
    }
}
fn completion_call_event() -> CompletionCall<'static> {
    static PROMPT: std::sync::OnceLock<rig_core::message::Message> = std::sync::OnceLock::new();
    CompletionCall {
        prompt: PROMPT.get_or_init(|| rig_core::message::Message::user("hi")),
        history: &[],
        turn: 1,
    }
}

fn invalid_tool_call_context() -> InvalidToolCallContext {
    InvalidToolCallContext {
        tool_name: "unknown".into(),
        tool_call_id: Some("tc1".into()),
        internal_call_id: Some(InternalCallId::new()),
        args: Some("{}".into()),
        available_tools: vec!["add".into()],
        allowed_tools: vec!["add".into()],
        tool_choice: None,
        chat_history: vec![],
        is_streaming: false,
    }
}

#[tokio::test]
async fn runs_hooks_in_registration_order_and_consults_all_on_continue() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut stack = HookStack::with(ToolRecorder {
        label: 1,
        log: log.clone(),
        stop: false,
    });
    stack.push(ToolRecorder {
        label: 2,
        log: log.clone(),
        stop: false,
    });
    assert_eq!(
        stack.on_tool_call(&ctx(), tool_call_event()).await,
        ToolCallAction::run()
    );
    assert_eq!(*log.lock().unwrap(), vec![1, 2]);
}

#[tokio::test]
async fn first_stop_short_circuits_on_chained_tool_call() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut stack = HookStack::with(ToolRecorder {
        label: 1,
        log: log.clone(),
        stop: true,
    });
    stack.push(ToolRecorder {
        label: 2,
        log: log.clone(),
        stop: false,
    });
    assert!(matches!(
        stack.on_tool_call(&ctx(), tool_call_event()).await,
        ToolCallAction::Stop(_)
    ));
    assert_eq!(*log.lock().unwrap(), vec![1]);
}

#[tokio::test]
async fn first_stop_short_circuits_observation() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut stack = HookStack::with(ObservationRecorder {
        label: 1,
        log: log.clone(),
        stop: true,
    });
    stack.push(ObservationRecorder {
        label: 2,
        log: log.clone(),
        stop: false,
    });
    assert!(matches!(
        stack
            .on_text_delta(
                &ctx(),
                TextDelta {
                    delta: "hi",
                    aggregated: "hi"
                }
            )
            .await,
        ObservationAction::Stop(_)
    ));
    assert_eq!(*log.lock().unwrap(), vec![1]);
}

#[tokio::test]
async fn reasoning_delta_observation_preserves_nested_order_and_stop() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut inner = HookStack::with(ObservationRecorder {
        label: 1,
        log: log.clone(),
        stop: false,
    });
    inner.push(ObservationRecorder {
        label: 2,
        log: log.clone(),
        stop: true,
    });
    let mut outer = HookStack::with(inner);
    outer.push(ObservationRecorder {
        label: 3,
        log: log.clone(),
        stop: false,
    });

    assert!(matches!(
        outer
            .on_reasoning_delta(
                &ctx(),
                ReasoningDelta {
                    id: "corr_1",
                    provider_id: Some("rs_1"),
                    delta: "think",
                    aggregated: "think",
                },
            )
            .await,
        ObservationAction::Stop(_)
    ));
    assert_eq!(*log.lock().expect("log"), vec![1, 2]);
}

#[tokio::test]
async fn explicit_fail_short_circuits_later_invalid_tool_hooks() {
    let fail_calls = Arc::new(AtomicUsize::new(0));
    let retry_calls = Arc::new(AtomicUsize::new(0));
    let mut stack = HookStack::with(InvalidResponder {
        action: InvalidToolCallAction::fail(),
        calls: fail_calls.clone(),
    });
    stack.push(InvalidResponder {
        action: InvalidToolCallAction::retry("try another tool"),
        calls: retry_calls.clone(),
    });

    let action = stack
        .on_invalid_tool_call(&ctx(), &invalid_tool_call_context())
        .await;

    assert_eq!(action, Some(InvalidToolCallAction::fail()));
    assert_eq!(fail_calls.load(Ordering::Relaxed), 1);
    assert_eq!(retry_calls.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn no_invalid_tool_decision_defers_to_later_hooks() {
    let retry_calls = Arc::new(AtomicUsize::new(0));
    let mut stack = HookStack::with(());
    stack.push(InvalidResponder {
        action: InvalidToolCallAction::retry("try another tool"),
        calls: retry_calls.clone(),
    });

    let action = stack
        .on_invalid_tool_call(&ctx(), &invalid_tool_call_context())
        .await;

    assert_eq!(
        action,
        Some(InvalidToolCallAction::retry("try another tool"))
    );
    assert_eq!(retry_calls.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn completion_patches_accumulate_and_stop_discards_prior_patch() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut stack = HookStack::with(Patcher {
        label: 1,
        log: log.clone(),
        patch: RequestPatch::new().temperature(0.1),
        stop: false,
    });
    stack.push(Patcher {
        label: 2,
        log: log.clone(),
        patch: RequestPatch::new().max_tokens(256),
        stop: false,
    });
    match stack
        .on_completion_call(&ctx(), completion_call_event())
        .await
    {
        CompletionCallAction::Patch(p) => {
            assert_eq!(p.temperature, Some(0.1));
            assert_eq!(p.max_tokens, Some(256));
        }
        other => panic!("expected patch, got {other:?}"),
    }
    assert_eq!(*log.lock().unwrap(), vec![1, 2]);
    let mut stopped = HookStack::with(Patcher {
        label: 3,
        log: log.clone(),
        patch: RequestPatch::new(),
        stop: true,
    });
    stopped.push(Patcher {
        label: 4,
        log: log.clone(),
        patch: RequestPatch::new(),
        stop: false,
    });
    assert!(matches!(
        stopped
            .on_completion_call(&ctx(), completion_call_event())
            .await,
        CompletionCallAction::Stop(_)
    ));
    assert_eq!(*log.lock().unwrap(), vec![1, 2, 3]);
}

#[tokio::test]
async fn nested_stack_composes_patches() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut inner = HookStack::with(Patcher {
        label: 1,
        log: log.clone(),
        patch: RequestPatch::new().temperature(0.2),
        stop: false,
    });
    inner.push(Patcher {
        label: 2,
        log: log.clone(),
        patch: RequestPatch::new().max_tokens(128),
        stop: false,
    });
    let mut outer = HookStack::with(inner);
    outer.push(Patcher {
        label: 3,
        log: log.clone(),
        patch: RequestPatch::new().preamble("outer"),
        stop: false,
    });
    match outer
        .on_completion_call(&ctx(), completion_call_event())
        .await
    {
        CompletionCallAction::Patch(p) => {
            assert_eq!(p.temperature, Some(0.2));
            assert_eq!(p.max_tokens, Some(128));
            assert_eq!(p.preamble.as_deref(), Some("outer"));
        }
        other => panic!("expected patch, got {other:?}"),
    }
    assert_eq!(*log.lock().unwrap(), vec![1, 2, 3]);
}

#[test]
fn stack_observes_is_the_or_of_members() {
    let mut stack = HookStack::with(ObservesOnly(StepEventKind::ToolCall));
    stack.push(ObservesOnly(StepEventKind::ToolResult));
    assert!(<HookStack as AgentHook>::observes(
        &stack,
        StepEventKind::ToolCall
    ));
    assert!(<HookStack as AgentHook>::observes(
        &stack,
        StepEventKind::ToolResult
    ));
    assert!(!<HookStack as AgentHook>::observes(
        &stack,
        StepEventKind::TextDelta
    ));
}

#[test]
fn empty_stack_observes_nothing() {
    let empty = HookStack::new();
    assert!(empty.is_empty());
    assert!(!<HookStack as AgentHook>::observes(
        &empty,
        StepEventKind::ToolCall
    ));
}

#[test]
fn unit_hook_observes_no_event_kind() {
    for kind in [
        StepEventKind::CompletionCall,
        StepEventKind::CompletionResponse,
        StepEventKind::ModelTurnFinished,
        StepEventKind::InvalidToolCall,
        StepEventKind::ToolCall,
        StepEventKind::ToolResult,
        StepEventKind::TextDelta,
        StepEventKind::ReasoningDelta,
        StepEventKind::ToolCallDelta,
        StepEventKind::StreamResponseFinish,
    ] {
        assert!(!<() as AgentHook>::observes(&(), kind));
    }
}

fn doc(id: &str) -> crate::completion::Document {
    crate::completion::Document {
        id: id.into(),
        text: String::new(),
        additional_props: Default::default(),
    }
}

#[test]
fn merge_appends_extra_context_in_order() {
    let merged = RequestPatch::new()
        .context(doc("a"))
        .merge(RequestPatch::new().context(doc("b")));
    assert_eq!(
        merged
            .extra_context
            .iter()
            .map(|d| d.id.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b"]
    );
}

#[test]
fn merge_shallow_merges_additional_params_later_wins() {
    let merged = RequestPatch::new()
        .additional_params(json!({"x":1,"y":2}))
        .merge(RequestPatch::new().additional_params(json!({"y":3,"z":4})));
    assert_eq!(merged.additional_params, Some(json!({"x":1,"y":3,"z":4})));
}

#[test]
fn merge_scalar_last_writer_wins() {
    assert_eq!(
        RequestPatch::new()
            .temperature(0.1)
            .merge(RequestPatch::new().temperature(0.9))
            .temperature,
        Some(0.9)
    );
}

#[test]
fn merge_active_tools_intersects() {
    let merged = RequestPatch::new()
        .active_tools(["add", "sub"])
        .merge(RequestPatch::new().active_tools(["sub", "mul"]));
    assert_eq!(merged.active_tools, Some(vec!["sub".into()]));
}

#[test]
fn merge_active_tools_empty_intersection_yields_empty() {
    assert_eq!(
        RequestPatch::new()
            .active_tools(["a"])
            .merge(RequestPatch::new().active_tools(["b"]))
            .active_tools,
        Some(vec![])
    );
}

#[test]
fn scratchpad_insert_get_update_remove() {
    #[derive(Clone, Default, Debug, PartialEq)]
    struct Count(u32);
    let pad = Scratchpad::default();
    pad.update(|c: &mut Count| c.0 += 1);
    pad.update(|c: &mut Count| c.0 += 1);
    assert_eq!(pad.get::<Count>(), Some(Count(2)));
    assert_eq!(pad.remove::<Count>(), Some(Count(2)));
}

#[test]
fn scratchpad_is_shared_across_clones() {
    let pad = Scratchpad::default();
    let clone = pad.clone();
    pad.insert(7u32);
    assert_eq!(clone.get::<u32>(), Some(7));
}

#[test]
fn hook_context_reports_identity_and_turn() {
    let context = HookContext::new(true, Some("agent".into()));
    assert!(context.is_streaming());
    assert_eq!(context.agent_name(), Some("agent"));
    context.set_turn(3);
    assert_eq!(context.turn(), 3);
    assert!(context.run_id().to_raw() > 0);
}

struct RewriteHook(Value);
impl AgentHook for RewriteHook {
    async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
        ToolCallAction::rewrite(self.0.clone())
    }
}
struct SkipHook;
impl AgentHook for SkipHook {
    async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
        ToolCallAction::skip("denied")
    }
}
struct StopHook;
impl AgentHook for StopHook {
    async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
        ToolCallAction::stop("stop")
    }
}
#[derive(Clone, Default)]
struct ArgsSpy(Arc<Mutex<Vec<String>>>);
impl AgentHook for ArgsSpy {
    async fn on_tool_call(&self, _: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
        self.0.lock().unwrap().push(event.args.into());
        ToolCallAction::run()
    }
}

struct OnToolCallOnly(Arc<AtomicUsize>);
impl AgentHook for OnToolCallOnly {
    async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
        self.0.fetch_add(1, Ordering::Relaxed);
        ToolCallAction::skip("called")
    }
}

struct YieldingRewriteFromCallId;
impl AgentHook for YieldingRewriteFromCallId {
    async fn on_tool_call(&self, _: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
        tokio::task::yield_now().await;
        ToolCallAction::rewrite(json!({"call_id": event.internal_call_id}))
    }
}

struct YieldingSkip;
impl AgentHook for YieldingSkip {
    async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
        tokio::task::yield_now().await;
        ToolCallAction::skip("denied")
    }
}

async fn resolve(stack: &HookStack) -> (ToolCallAction, Option<Value>) {
    stack.resolve_tool_call(&ctx(), tool_call_event()).await
}

#[tokio::test]
async fn erased_dispatch_uses_the_public_on_tool_call_method() {
    let calls = Arc::new(AtomicUsize::new(0));
    let stack = HookStack::with(OnToolCallOnly(calls.clone()));

    let (action, salvaged) = resolve(&stack).await;

    assert_eq!(action, ToolCallAction::skip("called"));
    assert_eq!(salvaged, None);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn string_rewrite_is_json_encoded_for_later_hook_in_same_stack() {
    let spy = ArgsSpy::default();
    let replacement = Value::String("sanitized".into());
    let mut stack = HookStack::new();
    stack.push(RewriteHook(replacement.clone()));
    stack.push(spy.clone());

    let (action, salvaged) = resolve(&stack).await;

    assert_eq!(action, ToolCallAction::rewrite(replacement.clone()));
    assert_eq!(salvaged, None);
    assert_eq!(
        spy.0.lock().unwrap().as_slice(),
        [serde_json::to_string(&replacement).unwrap()]
    );
}

#[tokio::test]
async fn string_rewrite_is_json_encoded_for_hook_in_nested_stack() {
    let spy = ArgsSpy::default();
    let replacement = Value::String("sanitized".into());
    let inner = HookStack::with(spy.clone());
    let mut outer = HookStack::new();
    outer.push(RewriteHook(replacement.clone()));
    outer.push(inner);

    let (action, salvaged) = resolve(&outer).await;

    assert_eq!(action, ToolCallAction::rewrite(replacement.clone()));
    assert_eq!(salvaged, None);
    assert_eq!(
        spy.0.lock().unwrap().as_slice(),
        [serde_json::to_string(&replacement).unwrap()]
    );
}

#[tokio::test]
async fn nested_rewrite_then_skip_preserves_rewrite() {
    let mut inner = HookStack::new();
    inner.push(RewriteHook(json!({"x":41})));
    inner.push(SkipHook);
    let mut outer = HookStack::new();
    outer.push(inner);
    let (action, salvaged) = resolve(&outer).await;
    assert!(matches!(action, ToolCallAction::Skip(_)));
    assert_eq!(salvaged, Some(json!({"x":41})));
}

#[tokio::test]
async fn nested_rewrite_then_stop_preserves_rewrite() {
    let mut inner = HookStack::new();
    inner.push(RewriteHook(json!({"x":41})));
    inner.push(StopHook);
    let mut outer = HookStack::new();
    outer.push(inner);
    let (action, salvaged) = resolve(&outer).await;
    assert!(matches!(action, ToolCallAction::Stop(_)));
    assert_eq!(salvaged, Some(json!({"x":41})));
}

#[tokio::test]
async fn deeply_nested_terminal_action_preserves_the_last_rewrite() {
    let mut inner = HookStack::new();
    inner.push(RewriteHook(json!({"x":3})));
    inner.push(SkipHook);

    let mut middle = HookStack::new();
    middle.push(RewriteHook(json!({"x":2})));
    middle.push(inner);

    let mut outer = HookStack::new();
    outer.push(RewriteHook(json!({"x":1})));
    outer.push(middle);

    let (action, salvaged) = resolve(&outer).await;

    assert_eq!(action, ToolCallAction::skip("denied"));
    assert_eq!(salvaged, Some(json!({"x":3})));
}

#[tokio::test]
async fn concurrent_nested_resolutions_keep_rewrites_isolated_by_call() {
    let mut inner = HookStack::new();
    inner.push(YieldingRewriteFromCallId);
    inner.push(YieldingSkip);
    let outer = HookStack::with(inner);
    let context = ctx();

    let first_id = InternalCallId::new();
    let second_id = InternalCallId::new();
    let first = outer.resolve_tool_call(
        &context,
        ToolCall {
            internal_call_id: first_id,
            ..tool_call_event()
        },
    );
    let second = outer.resolve_tool_call(
        &context,
        ToolCall {
            internal_call_id: second_id,
            ..tool_call_event()
        },
    );
    let ((first_action, first_rewrite), (second_action, second_rewrite)) =
        tokio::join!(first, second);

    assert_eq!(first_action, ToolCallAction::skip("denied"));
    assert_eq!(first_rewrite, Some(json!({"call_id": first_id})));
    assert_eq!(second_action, ToolCallAction::skip("denied"));
    assert_eq!(second_rewrite, Some(json!({"call_id": second_id})));
}

#[tokio::test]
async fn outer_rewrite_threads_into_nested_stack() {
    let spy = ArgsSpy::default();
    let mut inner = HookStack::new();
    inner.push(spy.clone());
    inner.push(SkipHook);
    let mut outer = HookStack::new();
    outer.push(RewriteHook(json!({"x":1})));
    outer.push(inner);
    let (action, salvaged) = resolve(&outer).await;
    assert!(matches!(action, ToolCallAction::Skip(_)));
    assert_eq!(salvaged, Some(json!({"x":1})));
    assert_eq!(
        spy.0.lock().unwrap().as_slice(),
        [serde_json::to_string(&json!({"x":1})).unwrap()]
    );
}

#[tokio::test]
async fn nested_proceeding_rewrite_surfaces_as_rewrite_action() {
    let mut proceed = HookStack::new();
    proceed.push(RewriteHook(json!({"x":5})));
    let (action, salvaged) = resolve(&proceed).await;
    assert_eq!(action, ToolCallAction::rewrite(json!({"x":5})));
    assert_eq!(salvaged, None);
}

#[test]
fn action_types_are_event_specific() {
    fn model_selection(_: ModelSelectionAction) {}
    fn completion(_: CompletionCallAction) {}
    fn model_turn(_: ModelTurnAction) {}
    fn retry_request(_: RetryRequest) {}
    fn call(_: ToolCallAction) {}
    fn result(_: ToolResultAction) {}
    fn invalid(_: InvalidToolCallAction) {}
    fn observation(_: ObservationAction) {}
    model_selection(ModelSelectionAction::continue_run());
    completion(CompletionCallAction::continue_run());
    model_turn(ModelTurnAction::retry_with_feedback("try again"));
    retry_request(RetryRequest::Repeat);
    call(ToolCallAction::run());
    result(ToolResultAction::keep());
    invalid(InvalidToolCallAction::fail());
    observation(ObservationAction::continue_run());
    let calls = AtomicUsize::new(0);
    calls.fetch_add(1, Ordering::Relaxed);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}
