//! The runner's `UnhandledInvalidToolCall::Ignore` policy on the
//! streaming surface, which the invalid-call matrix found unapplied.

use futures::StreamExt;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_core::tool::{Tool, ToolContext, ToolExecutionError};
use serde::Deserialize;
use serde_json::json;

use crate::agent::{AgentBuilder, MultiTurnStreamItem};
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

async fn output_under(policy: UnhandledInvalidToolCall) -> Result<String, String> {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::tool_call("call-1", "multiply", json!({"x": 2, "y": 3})),
        MockStreamEvent::final_response_with_default_usage(),
    ]]))
    .tool(Add)
    .build();
    let mut stream = agent
        .stream_prompt("go")
        .unhandled_invalid_tool_call(policy)
        .stream()
        .await;
    let mut output = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::FinalResponse(response)) => output = Some(response.output),
            Ok(_) => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    output.ok_or_else(|| "no final response".to_owned())
}

/// The streaming surface applies the policy as the blocking one does: an
/// unknown call under `Ignore` is dropped and the turn goes on — with
/// nothing else in it, an empty answer — and under `Fail` fails the run.
/// (It used to fail under both.)
#[tokio::test]
async fn the_streaming_surface_applies_the_unhandled_policy() {
    assert_eq!(
        output_under(UnhandledInvalidToolCall::Ignore).await,
        Ok(String::new())
    );
    let failed = output_under(UnhandledInvalidToolCall::Fail)
        .await
        .expect_err("Fail fails the run");
    assert!(failed.contains("UnknownToolCall"), "{failed}");
}
