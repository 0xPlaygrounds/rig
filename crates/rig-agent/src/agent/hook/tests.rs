use super::*;

/// The call the shared test events answer.
fn call_id() -> &'static CallId {
    static CALL: std::sync::OnceLock<CallId> = std::sync::OnceLock::new();
    CALL.get_or_init(|| CallId::from_wire("tc1"))
}

use crate::tool::{ToolErrorKind, ToolExecutionError};

/// Builds a dispatch event for `kind` answering the shared test block.
fn dispatch_event(kind: &EffectKind) -> DispatchEvent<'_> {
    DispatchEvent {
        id: EffectId::from_raw(1),
        kind,
        turn: 1,
        call_id: Some(call_id()),
        context: None,
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
        call_id: Some(call_id()),
        context: None,
    }
}

/// [`outcome_event`] with the context the tool answered with, beside the
/// outcome as the engine carries it (format 5).
fn outcome_event_with<'a>(
    kind: &'a EffectKind,
    outcome: &'a Result<Outcome, ErrorReport>,
    context: &'a ToolContext,
) -> OutcomeEvent<'a> {
    OutcomeEvent {
        context: Some(context),
        ..outcome_event(kind, outcome)
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
    });
    let action = stack
        .on_outcome(
            &HookContext::new(false, None, None),
            outcome_event_with(&kind, &outcome, &context),
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

// ---- hook stack composition and model-selection routing ----

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use serde_json::{Value, json};

fn ctx() -> HookContext {
    HookContext::new(false, Some("test-agent".to_string()), None)
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

fn tool_call_kind() -> EffectKind {
    EffectKind::ToolCall {
        name: "add".into(),
        args: "{}".into(),
    }
}

fn invalid_tool_call_context() -> InvalidToolCallContext {
    InvalidToolCallContext {
        tool_name: "unknown".into(),
        tool_call_id: Some(rig_core::message::CallId::from_wire("tc1")),
        args: Some("{}".into()),
        available_tools: vec!["add".into()],
        allowed_tools: vec!["add".into()],
        tool_choice: None,
        chat_history: vec![],
        is_streaming: false,
        reason: crate::run::policy::InvalidToolCallReason::UnknownTool,
    }
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

/// The patch a stack had accumulated when it denied: what the engine reads
/// for the skipped result's arguments (`HookContext::take_salvaged_patch`).
fn salvaged_args(context: &HookContext, id: EffectId) -> Option<Value> {
    context.take_salvaged_patch(id).and_then(|kind| match kind {
        EffectKind::ToolCall { args, .. } => serde_json::from_str(&args).ok(),
        _ => None,
    })
}

struct ChangeToolFamilyHook;
impl AgentHook for ChangeToolFamilyHook {
    async fn on_dispatch(&self, _: &HookContext, _: DispatchEvent<'_>) -> DispatchAction {
        DispatchAction::Patch(EffectKind::Custom {
            kind: "invalid-tool-replacement".into(),
            payload: json!({}),
        })
    }
}

#[tokio::test]
async fn nested_wrong_family_patch_preserves_valid_tool_and_stops_later_hooks() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut inner = HookStack::with(RewriteHook(json!({"x": 3})));
    inner.push(ChangeToolFamilyHook);
    inner.push(OnDispatchOnly(calls.clone()));
    let mut outer = HookStack::with(RewriteHook(json!({"x": 1})));
    outer.push(inner);
    outer.push(OnDispatchOnly(calls.clone()));
    let context = ctx();
    let kind = tool_call_kind();
    let event = dispatch_event(&kind);
    let action = outer.on_dispatch(&context, event).await;
    assert!(
        matches!(action, DispatchAction::Deny(ref report) if report.kind == ErrorKind::Internal)
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(salvaged_args(&context, event.id), Some(json!({"x": 3})));
}
