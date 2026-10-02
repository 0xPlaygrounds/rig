//! rig#2447: a tool call whose arguments are not a JSON object never ends a
//! run. The reply keeps the call with its raw text; the agent never runs the
//! tool and answers the call with an error result, so the model reads why and
//! calls again, as pi does. Each test proves what the *model* sees on the
//! next request.

use super::MultiTurnStreamItem;
use crate::agent::AgentBuilder;
use crate::agent::hook::{AgentHook, HookContext};
use crate::agent::{InvalidToolCallAction, InvalidToolCallContext};
use crate::test_utils::{MockAddTool, MockCompletionModel, MockStreamEvent};
use futures::StreamExt;
use rig_core::message::{Message, ToolResult, ToolResultContent, UserContent};

const RAW: &str = "{\"x\": 2, \"y\": \x01";

/// A stream whose tool call closes with malformed arguments, followed by a
/// healthy second turn the model would produce after the error result.
fn model_with_malformed_call() -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call_name_delta("tool_call_1", "add"),
            MockStreamEvent::tool_call_arguments_delta("tool_call_1", RAW),
            MockStreamEvent::tool_call_end("tool_call_1"),
            MockStreamEvent::final_response_with_total_tokens(4),
        ],
        vec![
            MockStreamEvent::text("recovered"),
            MockStreamEvent::final_response_with_total_tokens(6),
        ],
    ])
}

/// A hook that fails the test if consulted: malformed arguments are not an
/// invalid call to resolve.
#[derive(Clone)]
struct NeverConsulted;

impl AgentHook for NeverConsulted {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        context: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        panic!("malformed arguments reached the invalid-call hook: {context:?}");
    }
}

/// The result answering `tool_call_1` in the second request.
fn answer(history: &[Message]) -> Option<&ToolResult> {
    history.iter().find_map(|message| match message {
        Message::User { content } => content.iter().find_map(|item| match item {
            UserContent::ToolResult(result)
                if result.call.provider().map(|id| id.as_str()) == Some("tool_call_1") =>
            {
                Some(result)
            }
            _ => None,
        }),
        Message::System { .. } | Message::Assistant(_) => None,
    })
}

#[tokio::test]
async fn malformed_arguments_are_answered_with_an_error_result() {
    let model = model_with_malformed_call();
    let recorded = model.clone();
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();
    let mut stream = agent
        .prompt("add 2 and something")
        .max_turns(3)
        .add_hook(NeverConsulted)
        .stream();
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item.expect("a malformed call does not fail the run"));
    }
    assert!(items.iter().any(|item| matches!(
        item,
        MultiTurnStreamItem::FinalResponse(response) if response.output() == "recovered"
    )));

    let requests = recorded.requests();
    assert_eq!(requests.len(), 2, "the model gets another turn");
    let result = answer(&requests[1].chat_history).expect("the call is answered");
    assert!(result.is_error, "the answer is an error result: {result:?}");
    assert!(
        result.content.iter().any(|content| matches!(
            content,
            ToolResultContent::Text(text)
                if text.text.contains("not a JSON object") && text.text.contains(RAW)
        )),
        "the answer names the problem and the arguments sent: {result:?}"
    );
}
