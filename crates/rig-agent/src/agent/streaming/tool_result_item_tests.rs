//! A streamed run reports each tool's accepted output the moment its outcome
//! hooks accept it, one item per element of a returned top-level JSON array,
//! without waiting for the rest of the batch.

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use rig_core::effect::Answered;
use rig_core::message::{CallId, ToolResultContent};
use serde_json::json;
use tokio::sync::Notify;

use super::MultiTurnStreamItem;
use crate::agent::hook::{AgentHook, HookContext};
use crate::agent::{AgentBuilder, DispatchAction, DispatchEvent, OutcomeAction, OutcomeEvent};
use crate::test_utils::{
    MockAddTool, MockCompletionModel, MockOperationArgs, MockStreamEvent, MockToolError,
};
use crate::tool::{Tool, ToolContext, ToolOutput, ToolResult};

/// One reported item: call id, tool name, its content, index and count.
type Seen = (String, String, Vec<ToolResultContent>, usize, usize);

fn seen(item: &MultiTurnStreamItem) -> Option<Seen> {
    match item {
        MultiTurnStreamItem::ToolResultItem {
            tool_result,
            index,
            count,
        } => Some((
            tool_result.call.to_string(),
            tool_result.name.to_string(),
            tool_result.content.clone(),
            *index,
            *count,
        )),
        _ => None,
    }
}

fn id(wire: &str) -> String {
    CallId::from_wire(wire).to_string()
}

fn json_block(value: serde_json::Value) -> Vec<ToolResultContent> {
    vec![ToolResultContent::Json { value }]
}

/// A tool named `NAME` that returns `output()` once `gate`, if any, is
/// notified.
macro_rules! fixed_tool {
    ($ty:ident, $name:literal, $out:ty, $value:expr) => {
        #[derive(Clone, Default)]
        struct $ty {
            gate: Option<Arc<Notify>>,
        }

        impl Tool for $ty {
            const NAME: &'static str = $name;
            type Error = MockToolError;
            type Args = MockOperationArgs;
            type Output = $out;

            fn description(&self) -> String {
                $name.to_string()
            }

            fn parameters(&self) -> serde_json::Value {
                MockAddTool.parameters()
            }

            async fn call(
                &self,
                _context: &mut ToolContext,
                _args: Self::Args,
            ) -> Result<Self::Output, Self::Error> {
                if let Some(gate) = &self.gate {
                    gate.notified().await;
                }
                Ok($value)
            }
        }
    };
}

fixed_tool!(WaitTool, "wait", i32, 7);
fixed_tool!(ListTool, "list", Vec<&'static str>, vec!["a", "b"]);
fixed_tool!(EmptyTool, "empty", Vec<i32>, Vec::new());
fixed_tool!(TextTool, "text", String, "[1, 2]".to_string());
fixed_tool!(ObjectTool, "object", serde_json::Value, json!({"value": 3}));

/// A model whose first turn calls `names` in order, as `call_1`, `call_2`...
fn calls_model(names: &[&str]) -> MockCompletionModel {
    let mut first: Vec<MockStreamEvent> = names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            MockStreamEvent::tool_call(
                format!("part_{}", i + 1),
                *name,
                json!({"x": i + 1, "y": 2}),
            )
            .with_call_id(format!("call_{}", i + 1))
        })
        .collect();
    first.push(MockStreamEvent::final_response_with_total_tokens(4));
    MockCompletionModel::from_stream_turns([
        first,
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_total_tokens(6),
        ],
    ])
}

/// Under `tool_concurrency(2)`, `wait` returns only after the consumer saw
/// the last output item of `list`, the call beside it. An item reported
/// after the batch settles would never arrive, so the run would hang.
#[tokio::test]
async fn a_finished_tools_output_reaches_the_consumer_while_its_sibling_still_runs() {
    let gate = Arc::new(Notify::new());
    let agent = AgentBuilder::new(calls_model(&["wait", "list"]))
        .tool(WaitTool {
            gate: Some(gate.clone()),
        })
        .tool(ListTool::default())
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
            if seen(&item).is_some_and(|(_, name, _, index, _)| name == "list" && index == 1) {
                gate.notify_one();
            }
            items.push(item);
        }
    })
    .await;
    assert!(
        drained.is_ok(),
        "the output of `list` must reach the consumer while `wait` still runs"
    );

    let (one, two) = (id("call_1"), id("call_2"));
    let reported: Vec<Seen> = items.iter().filter_map(seen).collect();
    assert_eq!(
        reported,
        [
            (two.clone(), "list".into(), json_block(json!("a")), 0, 2),
            (two.clone(), "list".into(), json_block(json!("b")), 1, 2),
            (one.clone(), "wait".into(), json_block(json!(7)), 0, 1),
        ]
    );

    // Live items come in completion order, before the batch's commits and
    // results, which keep call order.
    let order: Vec<String> = items
        .iter()
        .filter_map(|item| match item {
            MultiTurnStreamItem::ToolResultItem { tool_result, .. } => {
                Some(format!("item:{}", tool_result.call))
            }
            MultiTurnStreamItem::ToolResult { tool_result } => {
                Some(format!("result:{}", tool_result.call))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        order,
        [
            format!("item:{two}"),
            format!("item:{two}"),
            format!("item:{one}"),
            format!("result:{one}"),
            format!("result:{two}"),
        ]
    );
}

/// Only a returned top-level JSON array is split. An object or scalar is one
/// item, an empty array is none, and text is never reparsed as JSON, even
/// when it reads as an array.
#[tokio::test]
async fn only_a_returned_json_array_is_split_into_items() {
    let agent = AgentBuilder::new(calls_model(&["object", "empty", "text"]))
        .tool(ObjectTool::default())
        .tool(EmptyTool::default())
        .tool(TextTool::default())
        .build();
    let mut stream = agent.prompt("use tools").max_turns(3).stream();

    let mut reported = Vec::new();
    while let Some(item) = stream.next().await {
        reported.extend(seen(&item.expect("the run succeeds")));
    }
    assert_eq!(
        reported,
        [
            (
                id("call_1"),
                "object".into(),
                json_block(json!({"value": 3})),
                0,
                1
            ),
            (
                id("call_3"),
                "text".into(),
                vec![ToolResultContent::text("[1, 2]")],
                0,
                1
            ),
        ]
    );
}

/// Rewrites `list`'s output to `["z"]` and skips `object` before it runs.
struct RewriteAndSkip;

impl AgentHook for RewriteAndSkip {
    async fn on_dispatch(&self, _: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        match event.tool_name() {
            Some("object") => DispatchAction::skip("blocked by policy"),
            _ => DispatchAction::proceed(),
        }
    }

    async fn on_outcome(&self, _: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        match event.tool_result() {
            Some(result) if result.output().as_json() == Some(&json!(["a", "b"])) => {
                OutcomeAction::replace(Ok(Answered::ToolResult {
                    result: ToolResult::success(ToolOutput::json(json!(["z"]))),
                }))
            }
            _ => OutcomeAction::proceed(),
        }
    }
}

/// The items are what the outcome hooks accepted, the same content the
/// committed result carries; a hook-skipped call reports its skip reason as
/// one error item.
#[tokio::test]
async fn items_carry_the_output_the_hooks_accepted() {
    let agent = AgentBuilder::new(calls_model(&["list", "object"]))
        .tool(ListTool::default())
        .tool(ObjectTool::default())
        .build();
    let mut stream = agent
        .prompt("use tools")
        .max_turns(3)
        .add_hook(RewriteAndSkip)
        .stream();

    let (mut items, mut results) = (Vec::new(), Vec::new());
    while let Some(item) = stream.next().await {
        match item.expect("the run succeeds") {
            MultiTurnStreamItem::ToolResultItem { tool_result, .. } => items.push(tool_result),
            MultiTurnStreamItem::ToolResult { tool_result } => results.push(tool_result),
            _ => {}
        }
    }
    assert_eq!(items.len(), 2);
    assert_eq!(
        items.first().map(|item| &item.content),
        Some(&json_block(json!("z")))
    );
    assert_eq!(
        results.first().map(|result| &result.content),
        Some(&json_block(json!(["z"])))
    );
    let skipped = items.get(1).expect("the skipped call reports an item");
    assert!(skipped.is_error);
    assert_eq!(
        skipped.content,
        vec![ToolResultContent::text("blocked by policy")]
    );
    assert_eq!(results.get(1), Some(skipped));
}

/// Stops the run at the call to `object`.
struct StopObject;

impl AgentHook for StopObject {
    async fn on_dispatch(&self, _: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        match event.tool_name() {
            Some("object") => DispatchAction::stop("stop the run"),
            _ => DispatchAction::proceed(),
        }
    }
}

/// An item already reported stays reported when a later call of its batch
/// ends the run; the failed batch commits no result.
#[tokio::test]
async fn an_item_reported_before_a_sibling_ends_the_run_stays_reported() {
    let agent = AgentBuilder::new(calls_model(&["list", "object"]))
        .tool(ListTool::default())
        .tool(ObjectTool::default())
        .build();
    let mut stream = agent
        .prompt("use tools")
        .max_turns(3)
        .add_hook(StopObject)
        .stream();

    let (mut items, mut results, mut failed) = (0, 0, false);
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::ToolResultItem { .. }) => items += 1,
            Ok(MultiTurnStreamItem::ToolResult { .. }) => results += 1,
            Ok(_) => {}
            Err(_) => failed = true,
        }
    }
    assert!(failed, "the hook ends the run");
    assert_eq!((items, results), (2, 0));
}
