//! The `UnhandledInvalidToolCall` policy on the streaming surface, for a
//! fresh run and a resumed one.

use futures::StreamExt;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent, MockTurn};
use rig_core::tool::{Tool, ToolContext, ToolExecutionError};
use serde::Deserialize;
use serde_json::json;

use crate::AgentRun;
use crate::agent::{AgentBuilder, MultiTurnStreamItem, StreamingResult};
use crate::run::UnhandledInvalidToolCall;

#[derive(Deserialize)]
struct AddArgs {
    x: i64,
    y: i64,
}

struct Add;

impl Tool for Add {
    const NAME: &'static str = "add";
    type Args = AddArgs;
    type Output = i64;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "adds".into()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}})
    }

    async fn call(&self, _context: &mut ToolContext, args: AddArgs) -> Result<i64, Self::Error> {
        Ok(args.x + args.y)
    }
}

async fn collect(mut stream: StreamingResult) -> Result<String, String> {
    let mut output = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::FinalResponse(response)) => output = Some(response.output()),
            Ok(_) => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    output.ok_or_else(|| "no final response".to_owned())
}

/// A persisted run with `policy`, resumed through a runner left at its
/// default policy, against a model that calls an unknown tool. Returns the
/// blocking and the streamed outcome.
async fn resumed_outputs(
    policy: UnhandledInvalidToolCall,
) -> (Result<String, String>, Result<String, String>) {
    let run = AgentRun::new("go").with_unhandled_invalid_tool_call(policy);
    let saved = serde_json::to_string(&run).map_err(|error| error.to_string());
    let restore = || -> Result<AgentRun, String> {
        let saved = saved.clone()?;
        serde_json::from_str(&saved).map_err(|error| error.to_string())
    };

    let unary = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::tool_call(
        "call-1",
        "multiply",
        json!({"x": 2, "y": 3}),
    )]))
    .tool(Add)
    .build();
    let blocking = match restore() {
        Ok(run) => unary
            .resume(run)
            .await
            .map(|response| response.output())
            .map_err(|error| error.to_string()),
        Err(error) => Err(error),
    };

    let streaming = AgentBuilder::new(MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::tool_call("call-1", "multiply", json!({"x": 2, "y": 3})),
        MockStreamEvent::final_response_with_default_usage(),
    ]]))
    .tool(Add)
    .build();
    let streamed = match restore() {
        Ok(run) => collect(streaming.resume(run).stream()).await,
        Err(error) => Err(error),
    };
    (blocking, streamed)
}

/// A resumed run's persisted policy governs both surfaces, whatever the
/// resuming runner's default: `Ignore` drops the call on both, `Fail` fails
/// both.
#[tokio::test]
async fn a_resumed_run_applies_its_own_policy_on_both_surfaces() {
    let (blocking, streamed) = resumed_outputs(UnhandledInvalidToolCall::Ignore).await;
    assert_eq!(blocking, Ok(String::new()));
    assert_eq!(streamed, Ok(String::new()));

    let (blocking, streamed) = resumed_outputs(UnhandledInvalidToolCall::Fail).await;
    for failed in [blocking, streamed] {
        let failed = failed.expect_err("Fail fails the run");
        assert!(
            failed.contains("unknown or disallowed tool `multiply`"),
            "{failed}"
        );
    }
}

/// A call that streams live under an allowed tool and is renamed by a later
/// fragment keeps the tool its start named, so its start always has an end
/// and the run never ignores a call it already forwarded.
#[tokio::test]
async fn a_call_renamed_after_its_start_keeps_its_streamed_name() {
    use rig_core::message::AssistantContent;
    use rig_core::streaming::{Item, PartKind, StreamEvent};

    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call_name_delta("tool_1", "add"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "{\"x\":1,"),
            MockStreamEvent::tool_call_name_delta("tool_1", "multiply"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "\"y\":2}"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::text("3"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]))
    .tool(Add)
    .build();
    let run =
        AgentRun::new("go").with_unhandled_invalid_tool_call(UnhandledInvalidToolCall::Ignore);
    let mut stream = agent.resume(run.max_turns(2)).stream();
    let (mut starts, mut ends, mut results) = (Vec::new(), Vec::new(), 0);
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Start {
                kind: PartKind::ToolCall,
                name,
                ..
            }))) => starts.push(name.map(|name| name.as_str().to_owned())),
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(call),
                ..
            }))) => ends.push(Some(call.function.name.as_str().to_owned())),
            Ok(MultiTurnStreamItem::ToolResult { .. }) => results += 1,
            Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
            Ok(_) => {}
            Err(error) => panic!("unexpected streaming error: {error}"),
        }
    }
    assert_eq!(starts, [Some("add".to_owned())]);
    assert_eq!(ends, starts);
    assert_eq!(results, 1);
}
