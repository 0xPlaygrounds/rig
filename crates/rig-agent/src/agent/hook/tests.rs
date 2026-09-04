use super::*;

static BLOCK: BlockId = BlockId::Wire(String::new());
use crate::tool::{ToolErrorKind, ToolExecutionError};

/// Rewrites the run-start prompt by appending its tag; used to observe
/// rewrite chaining across a stack.
struct StartRewriter(&'static str);
impl AgentHook for StartRewriter {
    async fn on_run_start(&self, _ctx: &HookContext, event: RunStart<'_>) -> RunStartAction {
        let current = match event.prompt {
            Message::User { .. } => event.prompt.rag_text().expect("test prompts carry text"),
            _ => panic!("run-start prompts are user messages"),
        };
        RunStartAction::rewrite(Message::user(format!("{current}{}", self.0)))
    }
}

struct StartStopper;
impl AgentHook for StartStopper {
    async fn on_run_start(&self, _ctx: &HookContext, _event: RunStart<'_>) -> RunStartAction {
        RunStartAction::stop("blocked at start")
    }
}

#[tokio::test]
async fn run_start_rewrites_chain_in_registration_order() {
    let mut stack = HookStack::with(StartRewriter("-a"));
    stack.push(StartRewriter("-b"));
    let ctx = HookContext::new(false, None, None);
    let prompt = Message::user("p");
    let action = stack
        .on_run_start(
            &ctx,
            RunStart {
                prompt: &prompt,
                history: &[],
            },
        )
        .await;
    match action {
        RunStartAction::Rewrite(message) => {
            // The second hook saw the first hook's rewrite.
            assert_eq!(message.rag_text().expect("text"), "p-a-b");
        }
        other => panic!("expected a chained rewrite, got {other:?}"),
    }
}

#[tokio::test]
async fn run_start_first_stop_wins_and_short_circuits() {
    let mut stack = HookStack::with(StartRewriter("-a"));
    stack.push(StartStopper);
    // This rewriter must never run.
    struct Panicker;
    impl AgentHook for Panicker {
        async fn on_run_start(&self, _ctx: &HookContext, _event: RunStart<'_>) -> RunStartAction {
            panic!("a stop must short-circuit later hooks");
        }
    }
    stack.push(Panicker);
    let ctx = HookContext::new(false, None, None);
    let prompt = Message::user("p");
    let action = stack
        .on_run_start(
            &ctx,
            RunStart {
                prompt: &prompt,
                history: &[],
            },
        )
        .await;
    assert_eq!(action, RunStartAction::Stop("blocked at start".into()));
}

#[test]
fn append_entry_stamps_the_current_turn_and_reads_replay_in_order() {
    let ctx = HookContext::new(false, None, None);
    // A resumed run's carried entries come first.
    ctx.seed_entries(&[RunEntry {
        kind: "counter".into(),
        turn: 2,
        value: serde_json::json!(2),
    }]);

    ctx.set_turn(3);
    ctx.append_entry("counter", &3u64).expect("serializable");
    ctx.append_entry("other", &()).expect("null marker");

    let counters = ctx.entries("counter");
    assert_eq!(
        counters.iter().map(|e| e.turn).collect::<Vec<_>>(),
        [2, 3],
        "seeded entries precede this run's appends"
    );
    // Last-wins snapshot read.
    let last = ctx.last_entry("counter").expect("appended");
    assert_eq!(last.turn, 3);
    assert_eq!(last.value, serde_json::json!(3));
    assert!(ctx.last_entry("absent").is_none());

    // Only this run's appends are pending for the driver to flush —
    // seeded entries are already in the run.
    let pending = ctx.drain_pending_entries();
    assert_eq!(
        pending.iter().map(|e| e.kind.as_str()).collect::<Vec<_>>(),
        ["counter", "other"]
    );
    // Draining does not affect reads, and is not repeatable.
    assert_eq!(ctx.entries("counter").len(), 2);
    assert!(ctx.drain_pending_entries().is_empty());
}

struct TemperaturePatcher(f64);
impl AgentHook for TemperaturePatcher {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCall<'_>,
    ) -> CompletionCallAction {
        CompletionCallAction::patch(RequestPatch::new().temperature(self.0))
    }
}

#[tokio::test]
async fn nested_completion_patches_compose() {
    let inner = HookStack::with(TemperaturePatcher(0.1));
    let mut outer = HookStack::with(inner);
    outer.push(TemperaturePatcher(0.2));
    let prompt = Message::user("hi");
    let action = outer
        .on_completion_call(
            &HookContext::new(false, None, None),
            CompletionCall {
                prompt: &prompt,
                history: &[],
                turn: 1,
            },
        )
        .await;
    assert!(matches!(
        action,
        CompletionCallAction::Patch(RequestPatch {
            temperature: Some(0.2),
            ..
        })
    ));
}

/// Builds a dispatch event for `kind` answering the shared test block.
fn dispatch_event(kind: &EffectKind) -> DispatchEvent<'_> {
    DispatchEvent {
        id: EffectId::from_raw(1),
        kind,
        turn: 1,
        block_id: Some(&BLOCK),
    }
}

/// Builds an outcome event for `kind` that resolved to `outcome`.
fn outcome_event<'a>(
    kind: &'a EffectKind,
    outcome: &'a Result<Outcome, ErrorReport>,
) -> OutcomeEvent<'a> {
    OutcomeEvent {
        id: EffectId::from_raw(1),
        kind,
        outcome,
        turn: 1,
        block_id: Some(&BLOCK),
    }
}

/// The arguments a `Patch` carries, parsed; `None` for anything else.
fn patched_args(action: &DispatchAction) -> Option<Value> {
    match action {
        DispatchAction::Patch(EffectKind::ToolCall { args, .. }) => {
            Some(serde_json::from_str(args).expect("patched args are JSON"))
        }
        _ => None,
    }
}

/// Asserts `action` skips the tool call with `reason`.
fn assert_skipped(action: &DispatchAction, reason: &str) {
    match action {
        DispatchAction::Deny(report) => {
            assert_eq!(report.kind, ErrorKind::Other);
            assert_eq!(report.message, reason);
        }
        other => panic!("expected a skip, got {other:?}"),
    }
}

/// The tool result a `Replace` carries; `None` for anything else.
fn replaced_result(action: &OutcomeAction) -> Option<&ToolResult> {
    match action {
        OutcomeAction::Replace(Ok(Outcome::ToolResult { result, .. })) => Some(result),
        _ => None,
    }
}

#[derive(Clone)]
struct CallRewriter {
    seen: Arc<std::sync::Mutex<Vec<String>>>,
    replacement: serde_json::Value,
}

impl AgentHook for CallRewriter {
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        let Some(args) = event.tool_args() else {
            return DispatchAction::proceed();
        };
        self.seen.lock().unwrap().push(args.to_string());
        DispatchAction::rewrite_tool_args(event.kind, self.replacement.clone())
    }
}

#[tokio::test]
async fn tool_call_rewrites_chain_in_registration_order() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut stack = HookStack::with(CallRewriter {
        seen: seen.clone(),
        replacement: serde_json::json!({"step": 1}),
    });
    stack.push(CallRewriter {
        seen: seen.clone(),
        replacement: serde_json::json!({"step": 2}),
    });

    let kind = EffectKind::ToolCall {
        name: "tool".into(),
        args: r#"{"step":0}"#.into(),
        context: ToolContext::new(),
    };
    let action = stack
        .on_dispatch(&HookContext::new(false, None, None), dispatch_event(&kind))
        .await;

    assert_eq!(
        *seen.lock().unwrap(),
        vec![r#"{"step":0}"#.to_string(), r#"{"step":1}"#.to_string()]
    );
    assert_eq!(patched_args(&action), Some(serde_json::json!({"step": 2})));
}

#[derive(Clone)]
struct ResultRewriter {
    seen: Arc<std::sync::Mutex<Vec<(String, ToolErrorKind, String)>>>,
    replacement: String,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq)]
struct RequestMetadata(String);

impl rig_core::tool::ContextValue for RequestMetadata {
    const KEY: &'static str = "test.request_metadata";
}

impl AgentHook for ResultRewriter {
    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        let Some(result) = event.tool_result() else {
            return OutcomeAction::proceed();
        };
        self.seen.lock().unwrap().push((
            result.output().render(),
            result.error().unwrap().kind(),
            event
                .tool_context()
                .unwrap()
                .result::<RequestMetadata>()
                .unwrap()
                .unwrap()
                .0,
        ));
        OutcomeAction::rewrite_tool_result(&event, self.replacement.clone())
    }
}

#[tokio::test]
async fn result_rewrites_chain_without_mutating_raw_result_or_context() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut stack = HookStack::with(ResultRewriter {
        seen: seen.clone(),
        replacement: "redacted".into(),
    });
    stack.push(ResultRewriter {
        seen: seen.clone(),
        replacement: "truncated".into(),
    });
    let raw = ToolResult::failed(ToolExecutionError::timeout("raw failure"));
    let mut context = ToolContext::new();
    context
        .insert_result(RequestMetadata("request-metadata".to_string()))
        .unwrap();

    let kind = tool_call_kind();
    let outcome = Ok(Outcome::ToolResult {
        result: raw.clone(),
        context: context.clone(),
    });
    let action = stack
        .on_outcome(
            &HookContext::new(false, None, None),
            outcome_event(&kind, &outcome),
        )
        .await;

    let replaced = replaced_result(&action).expect("a rewritten tool result");
    assert_eq!(replaced.output().as_text(), Some("truncated"));
    // The rewrite keeps the result's status.
    assert_eq!(replaced.error().unwrap().kind(), ToolErrorKind::Timeout);
    assert_eq!(
        *seen.lock().unwrap(),
        vec![
            (
                "raw failure".into(),
                ToolErrorKind::Timeout,
                "request-metadata".into()
            ),
            (
                "redacted".into(),
                ToolErrorKind::Timeout,
                "request-metadata".into()
            ),
        ]
    );
    assert_eq!(raw.output().as_text(), Some("raw failure"));
    assert_eq!(
        context
            .result::<RequestMetadata>()
            .unwrap()
            .map(|m| m.0)
            .as_deref(),
        Some("request-metadata")
    );
}

struct StopThenCount {
    stop: bool,
    calls: Arc<AtomicUsize>,
    seen_cancelled: Arc<AtomicUsize>,
}

impl AgentHook for StopThenCount {
    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if matches!(
            event.outcome,
            Err(ErrorReport {
                kind: ErrorKind::Cancelled,
                ..
            })
        ) {
            self.seen_cancelled.fetch_add(1, Ordering::Relaxed);
        }
        if self.stop {
            OutcomeAction::stop("terminal")
        } else {
            OutcomeAction::proceed()
        }
    }
}

#[tokio::test]
async fn stop_outcome_threads_to_later_hooks_and_surfaces_as_cancelled() {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen_cancelled = Arc::new(AtomicUsize::new(0));
    let mut stack = HookStack::with(StopThenCount {
        stop: true,
        calls: calls.clone(),
        seen_cancelled: seen_cancelled.clone(),
    });
    stack.push(StopThenCount {
        stop: false,
        calls: calls.clone(),
        seen_cancelled: seen_cancelled.clone(),
    });
    let kind = tool_call_kind();
    let outcome = Ok(Outcome::ToolResult {
        result: ToolResult::success(ToolOutput::text("ok")),
        context: ToolContext::new(),
    });
    let action = stack
        .on_outcome(
            &HookContext::new(false, None, None),
            outcome_event(&kind, &outcome),
        )
        .await;

    assert!(matches!(
        action,
        OutcomeAction::Replace(Err(ErrorReport {
            kind: ErrorKind::Cancelled,
            ref message,
            ..
        })) if message == "terminal"
    ));
    // A stop is a replacement like any other: the later hook sees it as the
    // outcome instead of being short-circuited.
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(seen_cancelled.load(Ordering::Relaxed), 1);
}

// ---- hook stack composition and model-selection routing ----

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use serde_json::{Value, json};

fn ctx() -> HookContext {
    HookContext::new(false, Some("test-agent".to_string()), None)
}

fn model(label: &str) -> ModelRef {
    ModelRef::new(label)
}

enum RouteDecision {
    Continue,
    Select(ModelRef),
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
            .push((self.label, Some(event.selected_model.to_string())));
        match &self.decision {
            RouteDecision::Continue => ModelSelectionAction::continue_run(),
            RouteDecision::Select(model) => ModelSelectionAction::select(model.clone()),
            RouteDecision::Stop => ModelSelectionAction::stop("routing stopped"),
        }
    }
}

fn model_selection<'a>(prompt: &'a Message, default_model: &'a ModelRef) -> ModelSelection<'a> {
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
    assert_eq!(selected.as_str(), "last");
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
    assert_eq!(selected.as_str(), "inner");
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
    assert_eq!(selected.as_str(), "outer");
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
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if event.tool_name().is_none() {
            return DispatchAction::proceed();
        }
        self.log.lock().expect("log").push(self.label);
        if self.stop {
            DispatchAction::stop("stop")
        } else {
            DispatchAction::proceed()
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

fn tool_call_kind() -> EffectKind {
    EffectKind::ToolCall {
        name: "add".into(),
        args: "{}".into(),
        context: ToolContext::new(),
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
        block_id: Some(BlockId::wire("tc1")),
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
    let kind = tool_call_kind();
    assert!(matches!(
        stack.on_dispatch(&ctx(), dispatch_event(&kind)).await,
        DispatchAction::Proceed
    ));
    assert_eq!(*log.lock().unwrap(), vec![1, 2]);
}

#[tokio::test]
async fn first_stop_short_circuits_on_chained_tool_dispatch() {
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
    let kind = tool_call_kind();
    assert!(matches!(
        stack.on_dispatch(&ctx(), dispatch_event(&kind)).await,
        DispatchAction::Deny(ErrorReport {
            kind: ErrorKind::Cancelled,
            ..
        })
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
                    id: &BlockId::wire("corr_1"),
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
    let mut stack = HookStack::with(ObservesOnly(StepEventKind::ToolDispatch));
    stack.push(ObservesOnly(StepEventKind::CompletionDispatch));
    assert!(<HookStack as AgentHook>::observes(
        &stack,
        StepEventKind::ToolDispatch
    ));
    assert!(<HookStack as AgentHook>::observes(
        &stack,
        StepEventKind::CompletionDispatch
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
        StepEventKind::ToolDispatch
    ));
}

#[test]
fn unit_hook_observes_no_event_kind() {
    for kind in [
        StepEventKind::RunStart,
        StepEventKind::RunSettled,
        StepEventKind::CompletionCall,
        StepEventKind::ModelTurnFinished,
        StepEventKind::InvalidToolCall,
        StepEventKind::TextDelta,
        StepEventKind::ReasoningDelta,
        StepEventKind::ToolCallDelta,
        StepEventKind::CompletionDispatch,
        StepEventKind::ToolDispatch,
        StepEventKind::EmbedDispatch,
        StepEventKind::RerankDispatch,
        StepEventKind::MemoryDispatch,
        StepEventKind::RetrieveDispatch,
        StepEventKind::CustomDispatch,
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
    let context = HookContext::new(true, Some("agent".into()), None);
    assert!(context.is_streaming());
    assert_eq!(context.agent_name(), Some("agent"));
    context.set_turn(3);
    assert_eq!(context.turn(), 3);
    assert!(context.run_id().to_raw() > 0);
}

struct RewriteHook(Value);
impl AgentHook for RewriteHook {
    async fn on_dispatch(&self, _: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if event.tool_name().is_none() {
            return DispatchAction::proceed();
        }
        DispatchAction::rewrite_tool_args(event.kind, self.0.clone())
    }
}
struct SkipHook;
impl AgentHook for SkipHook {
    async fn on_dispatch(&self, _: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if event.tool_name().is_none() {
            return DispatchAction::proceed();
        }
        DispatchAction::skip("denied")
    }
}
#[derive(Clone, Default)]
struct ArgsSpy(Arc<Mutex<Vec<String>>>);
impl AgentHook for ArgsSpy {
    async fn on_dispatch(&self, _: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        let Some(args) = event.tool_args() else {
            return DispatchAction::proceed();
        };
        self.0.lock().unwrap().push(args.into());
        DispatchAction::proceed()
    }
}

struct OnDispatchOnly(Arc<AtomicUsize>);
impl AgentHook for OnDispatchOnly {
    async fn on_dispatch(&self, _: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if event.tool_name().is_none() {
            return DispatchAction::proceed();
        }
        self.0.fetch_add(1, Ordering::Relaxed);
        DispatchAction::skip("called")
    }
}

async fn resolve(stack: &HookStack) -> DispatchAction {
    let kind = tool_call_kind();
    stack.on_dispatch(&ctx(), dispatch_event(&kind)).await
}

#[tokio::test]
async fn erased_dispatch_uses_the_public_on_dispatch_method() {
    let calls = Arc::new(AtomicUsize::new(0));
    let stack = HookStack::with(OnDispatchOnly(calls.clone()));

    let action = resolve(&stack).await;

    assert_skipped(&action, "called");
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn string_rewrite_is_json_encoded_for_later_hook_in_same_stack() {
    let spy = ArgsSpy::default();
    let replacement = Value::String("sanitized".into());
    let mut stack = HookStack::new();
    stack.push(RewriteHook(replacement.clone()));
    stack.push(spy.clone());

    let action = resolve(&stack).await;

    assert_eq!(patched_args(&action), Some(replacement.clone()));
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

    let action = resolve(&outer).await;

    assert_eq!(patched_args(&action), Some(replacement.clone()));
    assert_eq!(
        spy.0.lock().unwrap().as_slice(),
        [serde_json::to_string(&replacement).unwrap()]
    );
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
    let action = resolve(&outer).await;
    assert_skipped(&action, "denied");
    assert_eq!(
        spy.0.lock().unwrap().as_slice(),
        [serde_json::to_string(&json!({"x":1})).unwrap()]
    );
}

#[tokio::test]
async fn nested_proceeding_rewrite_surfaces_as_patch_action() {
    let mut proceed = HookStack::new();
    proceed.push(RewriteHook(json!({"x":5})));
    let action = resolve(&proceed).await;
    assert_eq!(patched_args(&action), Some(json!({"x":5})));
}

struct StopHook;
impl AgentHook for StopHook {
    async fn on_dispatch(&self, _: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if event.tool_name().is_none() {
            return DispatchAction::proceed();
        }
        DispatchAction::stop("halt")
    }
}

/// The patch a stack had accumulated when it denied: what the engine reads
/// for the skipped result's arguments (`HookContext::take_salvaged_patch`).
fn salvaged_args(context: &HookContext, id: EffectId) -> Option<Value> {
    context.take_salvaged_patch(id).and_then(|kind| match kind {
        EffectKind::ToolCall { args, .. } => serde_json::from_str(&args).ok(),
        _ => None,
    })
}

#[tokio::test]
async fn nested_rewrite_then_skip_preserves_rewrite() {
    let mut inner = HookStack::new();
    inner.push(RewriteHook(json!({"x":41})));
    inner.push(SkipHook);
    let mut outer = HookStack::new();
    outer.push(inner);
    let context = ctx();
    let kind = tool_call_kind();
    let event = dispatch_event(&kind);
    let action = outer.on_dispatch(&context, event).await;
    assert_skipped(&action, "denied");
    assert_eq!(salvaged_args(&context, event.id), Some(json!({"x":41})));
}

#[tokio::test]
async fn nested_rewrite_then_stop_preserves_rewrite() {
    let mut inner = HookStack::new();
    inner.push(RewriteHook(json!({"x":41})));
    inner.push(StopHook);
    let mut outer = HookStack::new();
    outer.push(inner);
    let context = ctx();
    let kind = tool_call_kind();
    let event = dispatch_event(&kind);
    let action = outer.on_dispatch(&context, event).await;
    assert!(
        matches!(&action, DispatchAction::Deny(report) if report.kind == ErrorKind::Cancelled),
        "{action:?}"
    );
    assert_eq!(salvaged_args(&context, event.id), Some(json!({"x":41})));
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

    let context = ctx();
    let kind = tool_call_kind();
    let event = dispatch_event(&kind);
    let action = outer.on_dispatch(&context, event).await;
    assert_skipped(&action, "denied");
    assert_eq!(salvaged_args(&context, event.id), Some(json!({"x":3})));
}

/// Rewrites the arguments to the dispatch's own id, yielding first so two
/// concurrent resolutions interleave.
struct YieldingRewriteFromId;
impl AgentHook for YieldingRewriteFromId {
    async fn on_dispatch(&self, _: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        tokio::task::yield_now().await;
        DispatchAction::rewrite_tool_args(event.kind, json!({"id": event.id.as_u64()}))
    }
}

struct YieldingSkip;
impl AgentHook for YieldingSkip {
    async fn on_dispatch(&self, _: &HookContext, _: DispatchEvent<'_>) -> DispatchAction {
        tokio::task::yield_now().await;
        DispatchAction::skip("denied")
    }
}

#[tokio::test]
async fn concurrent_nested_resolutions_keep_rewrites_isolated_by_dispatch() {
    let mut inner = HookStack::new();
    inner.push(YieldingRewriteFromId);
    inner.push(YieldingSkip);
    let outer = HookStack::with(inner);
    let context = ctx();
    let kind = tool_call_kind();
    let first = DispatchEvent {
        id: EffectId::from_raw(7),
        ..dispatch_event(&kind)
    };
    let second = DispatchEvent {
        id: EffectId::from_raw(8),
        ..dispatch_event(&kind)
    };
    let (first_action, second_action) = tokio::join!(
        outer.on_dispatch(&context, first),
        outer.on_dispatch(&context, second)
    );
    assert_skipped(&first_action, "denied");
    assert_skipped(&second_action, "denied");
    assert_eq!(salvaged_args(&context, first.id), Some(json!({"id": 7})));
    assert_eq!(salvaged_args(&context, second.id), Some(json!({"id": 8})));
}

#[test]
fn action_types_are_event_specific() {
    fn model_selection(_: ModelSelectionAction) {}
    fn completion(_: CompletionCallAction) {}
    fn model_turn(_: ModelTurnAction) {}
    fn retry_request(_: RetryRequest) {}
    fn dispatch(_: DispatchAction) {}
    fn outcome(_: OutcomeAction) {}
    fn invalid(_: InvalidToolCallAction) {}
    fn observation(_: ObservationAction) {}
    model_selection(ModelSelectionAction::continue_run());
    completion(CompletionCallAction::continue_run());
    model_turn(ModelTurnAction::retry_with_feedback("try again"));
    retry_request(RetryRequest::Repeat);
    dispatch(DispatchAction::proceed());
    outcome(OutcomeAction::proceed());
    invalid(InvalidToolCallAction::fail());
    observation(ObservationAction::continue_run());
    let calls = AtomicUsize::new(0);
    calls.fetch_add(1, Ordering::Relaxed);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

/// A hook that names itself is recorded under that name; one that does
/// not is recorded under its type name.
#[test]
fn a_hook_names_itself_or_is_named_by_its_type() {
    struct Plain;
    impl AgentHook for Plain {}
    struct Threshold(usize);
    impl AgentHook for Threshold {
        fn name(&self) -> Option<String> {
            Some(format!("Threshold({})", self.0))
        }
    }
    let mut stack = HookStack::new();
    stack.push(Plain);
    stack.push(Threshold(2));
    assert_eq!(stack.names(), ["Plain", "Threshold(2)"]);
}
