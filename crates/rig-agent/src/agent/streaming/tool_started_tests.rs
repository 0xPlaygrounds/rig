//! A streamed run reports each tool call it dispatches the moment it starts,
//! before the call's hooks and body run, with the id its other items carry.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures::StreamExt;
use rig_core::message::{CallId, ToolCall};
use tokio::sync::Notify;

use super::MultiTurnStreamItem;
use crate::agent::hook::{AgentHook, HookContext};
use crate::agent::{
    AgentBuilder, DispatchAction, DispatchEvent, InvalidToolCallAction, InvalidToolCallContext,
};
use crate::test_utils::{
    MockAddTool, MockCompletionModel, MockOperationArgs, MockStreamEvent, MockToolError,
};
use crate::tool::{Tool, ToolContext};

/// The call a start item reports.
fn started(item: &MultiTurnStreamItem) -> Option<&ToolCall> {
    match item {
        MultiTurnStreamItem::ToolExecutionStarted { tool_call } => Some(tool_call),
        _ => None,
    }
}

/// A tool named `wait` that returns only once `gate` is notified.
#[derive(Clone)]
struct WaitTool {
    gate: Arc<Notify>,
}

impl Tool for WaitTool {
    const NAME: &'static str = "wait";
    type Error = MockToolError;
    type Args = MockOperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Wait for a signal".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        MockAddTool.parameters()
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.gate.notified().await;
        Ok(0)
    }
}

/// A tool named `add` that records whether the consumer had seen a start
/// item when its body ran.
#[derive(Clone)]
struct ObservedAdd {
    seen: Arc<AtomicBool>,
    seen_when_ran: Arc<AtomicBool>,
}

impl Tool for ObservedAdd {
    const NAME: &'static str = "add";
    type Error = MockToolError;
    type Args = MockOperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        MockAddTool.description()
    }

    fn parameters(&self) -> serde_json::Value {
        MockAddTool.parameters()
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.seen_when_ran
            .store(self.seen.load(Ordering::SeqCst), Ordering::SeqCst);
        Ok(0)
    }
}

fn two_call_model(first: &str, second: &str) -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("part_1", first, serde_json::json!({"x": 1, "y": 2}))
                .with_call_id("call_1"),
            MockStreamEvent::tool_call("part_2", second, serde_json::json!({"x": 3, "y": 4}))
                .with_call_id("call_2"),
            MockStreamEvent::final_response_with_total_tokens(4),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_total_tokens(6),
        ],
    ])
}

/// Under `tool_concurrency(2)`, `wait` returns only after the consumer saw
/// the start of `add`, the call beside it. A start reported after the batch
/// settles would never arrive, so the run would hang.
#[tokio::test]
async fn a_call_start_reaches_the_consumer_while_its_sibling_still_runs() {
    let gate = Arc::new(Notify::new());
    let agent = AgentBuilder::new(two_call_model("wait", "add"))
        .tool(WaitTool { gate: gate.clone() })
        .tool(MockAddTool)
        .build();
    let mut stream = agent
        .prompt("use tools")
        .max_turns(3)
        .tool_concurrency(2)
        .stream();

    let mut items = Vec::new();
    let drained = tokio::time::timeout(Duration::from_secs(5), async {
        while let Some(item) = stream.next().await {
            let item = item.expect("the run succeeds");
            if started(&item).is_some_and(|call| call.function.name.as_str() == "add") {
                gate.notify_one();
            }
            items.push(item);
        }
    })
    .await;
    assert!(
        drained.is_ok(),
        "the start of `add` must reach the consumer while `wait` still runs"
    );

    // One start per call, each after the model's call items and before
    // any commit or result, carrying the id they carry.
    let order: Vec<String> = items
        .iter()
        .filter_map(|item| match item {
            MultiTurnStreamItem::ToolCall { tool_call } => Some(format!("call:{}", tool_call.id)),
            MultiTurnStreamItem::ToolExecutionStarted { tool_call } => {
                Some(format!("start:{}", tool_call.id))
            }
            MultiTurnStreamItem::ToolExecutionCommitted { tool_call } => {
                Some(format!("ran:{}", tool_call.id))
            }
            MultiTurnStreamItem::ToolResult { tool_result } => {
                Some(format!("result:{}", tool_result.call))
            }
            _ => None,
        })
        .collect();
    let id = |wire: &str| CallId::from_wire(wire).to_string();
    let (one, two) = (id("call_1"), id("call_2"));
    assert_eq!(
        order,
        [
            format!("call:{one}"),
            format!("call:{two}"),
            format!("start:{one}"),
            format!("start:{two}"),
            format!("ran:{one}"),
            format!("result:{one}"),
            format!("ran:{two}"),
            format!("result:{two}"),
        ]
    );
}

/// At the default sequential concurrency, a call's start is taken by the
/// consumer before that call's body runs.
#[tokio::test]
async fn a_sequential_call_start_arrives_before_its_body_runs() {
    let seen = Arc::new(AtomicBool::new(false));
    let seen_when_ran = Arc::new(AtomicBool::new(false));
    let agent = AgentBuilder::new(two_call_model("add", "add"))
        .tool(ObservedAdd {
            seen: seen.clone(),
            seen_when_ran: seen_when_ran.clone(),
        })
        .build();
    let mut stream = agent.prompt("use tools").max_turns(3).stream();

    let mut starts = 0;
    while let Some(item) = stream.next().await {
        let item = item.expect("the run succeeds");
        if started(&item).is_some() {
            starts += 1;
            seen.store(true, Ordering::SeqCst);
        }
    }
    assert_eq!(starts, 2, "every dispatched call reports one start");
    assert!(
        seen_when_ran.load(Ordering::SeqCst),
        "the consumer must have taken the start before the body ran"
    );
}

struct SkipEveryToolCall;

impl AgentHook for SkipEveryToolCall {
    async fn on_dispatch(&self, _: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        match event.tool_args() {
            Some(_) => DispatchAction::skip("blocked by policy"),
            None => DispatchAction::proceed(),
        }
    }
}

/// A call a dispatch hook skips still started: its hooks ran. It reports a
/// start and a result, but no execution commit.
#[tokio::test]
async fn a_hook_skipped_call_reports_a_start_but_no_commit() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("part_1", "add", serde_json::json!({"x": 1, "y": 2}))
                .with_call_id("call_1"),
            MockStreamEvent::final_response_with_total_tokens(4),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_total_tokens(6),
        ],
    ]);
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();
    let mut stream = agent
        .prompt("use tools")
        .max_turns(3)
        .add_hook(SkipEveryToolCall)
        .stream();

    let (mut starts, mut commits, mut results) = (0, 0, 0);
    while let Some(item) = stream.next().await {
        match item.expect("the run succeeds") {
            MultiTurnStreamItem::ToolExecutionStarted { .. } => starts += 1,
            MultiTurnStreamItem::ToolExecutionCommitted { .. } => commits += 1,
            MultiTurnStreamItem::ToolResult { .. } => results += 1,
            _ => {}
        }
    }
    assert_eq!((starts, commits, results), (1, 0, 1));
}

struct SkipUnknownTool;

impl AgentHook for SkipUnknownTool {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        _context: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        Some(InvalidToolCallAction::skip("no such tool"))
    }
}

/// A turn whose invalid call is answered by recovery dispatches none of its
/// calls, so it reports no start, though it is still answered.
#[tokio::test]
async fn a_turn_answered_by_invalid_call_recovery_reports_no_start() {
    let agent = AgentBuilder::new(two_call_model("missing", "add"))
        .tool(MockAddTool)
        .build();
    let mut stream = agent
        .prompt("use tools")
        .max_turns(3)
        .add_hook(SkipUnknownTool)
        .stream();

    let (mut starts, mut results) = (0, 0);
    while let Some(item) = stream.next().await {
        match item.expect("the run succeeds") {
            MultiTurnStreamItem::ToolExecutionStarted { .. } => starts += 1,
            MultiTurnStreamItem::ToolResult { .. } => results += 1,
            _ => {}
        }
    }
    assert_eq!(starts, 0, "no call of the recovered turn is dispatched");
    assert!(results > 0, "the recovered turn is still answered");
}
