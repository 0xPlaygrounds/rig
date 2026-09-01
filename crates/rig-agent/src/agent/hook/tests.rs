use super::*;
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
    let ctx = HookContext::new(false, None);
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
    let ctx = HookContext::new(false, None);
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
    let ctx = HookContext::new(false, None);
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

struct Patcher(f64);
impl AgentHook for Patcher {
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
    let inner = HookStack::with(Patcher(0.1));
    let mut outer = HookStack::with(inner);
    outer.push(Patcher(0.2));
    let prompt = Message::user("hi");
    let action = outer
        .on_completion_call(
            &HookContext::new(false, None),
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

#[derive(Clone)]
struct CallRewriter {
    seen: Arc<std::sync::Mutex<Vec<String>>>,
    replacement: serde_json::Value,
}

impl AgentHook for CallRewriter {
    async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
        self.seen.lock().unwrap().push(event.args.to_string());
        ToolCallAction::rewrite(self.replacement.clone())
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

    let action = stack
        .on_tool_call(
            &HookContext::new(false, None),
            ToolCall {
                tool_name: "tool",
                tool_call_id: Some("provider-id"),
                internal_call_id: InternalCallId::new(),
                args: r#"{"step":0}"#,
            },
        )
        .await;

    assert_eq!(
        *seen.lock().unwrap(),
        vec![r#"{"step":0}"#.to_string(), r#"{"step":1}"#.to_string()]
    );
    assert_eq!(
        action,
        ToolCallAction::rewrite(serde_json::json!({"step": 2}))
    );
}

#[derive(Clone)]
struct ResultRewriter {
    seen: Arc<std::sync::Mutex<Vec<(String, ToolErrorKind, String)>>>,
    replacement: String,
}

impl AgentHook for ResultRewriter {
    async fn on_tool_result(
        &self,
        _ctx: &HookContext,
        event: ToolResultEvent<'_>,
    ) -> ToolResultAction {
        self.seen.lock().unwrap().push((
            event.presentation.render(),
            event.raw_result.error().unwrap().kind(),
            event.tool_context.result::<String>().unwrap().clone(),
        ));
        ToolResultAction::rewrite(self.replacement.clone())
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
    context.insert_result("request-metadata".to_string());

    let action = stack
        .on_tool_result(
            &HookContext::new(false, None),
            ToolResultEvent {
                tool_name: "tool",
                tool_call_id: None,
                internal_call_id: InternalCallId::new(),
                args: "{}",
                presentation: raw.output(),
                raw_result: &raw,
                tool_context: &context,
            },
        )
        .await;

    assert_eq!(action, ToolResultAction::rewrite("truncated"));
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
        context.result::<String>().map(String::as_str),
        Some("request-metadata")
    );
}

struct StopThenCount {
    stop: bool,
    calls: Arc<AtomicUsize>,
}

impl AgentHook for StopThenCount {
    async fn on_tool_result(
        &self,
        _ctx: &HookContext,
        _event: ToolResultEvent<'_>,
    ) -> ToolResultAction {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.stop {
            ToolResultAction::stop("terminal")
        } else {
            ToolResultAction::keep()
        }
    }
}

#[tokio::test]
async fn terminal_result_action_short_circuits_later_hooks() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut stack = HookStack::with(StopThenCount {
        stop: true,
        calls: calls.clone(),
    });
    stack.push(StopThenCount {
        stop: false,
        calls: calls.clone(),
    });
    let raw = ToolResult::success(ToolOutput::text("ok"));
    let context = ToolContext::new();
    let action = stack
        .on_tool_result(
            &HookContext::new(false, None),
            ToolResultEvent {
                tool_name: "tool",
                tool_call_id: None,
                internal_call_id: InternalCallId::new(),
                args: "{}",
                presentation: raw.output(),
                raw_result: &raw,
                tool_context: &context,
            },
        )
        .await;

    assert_eq!(action, ToolResultAction::stop("terminal"));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}
