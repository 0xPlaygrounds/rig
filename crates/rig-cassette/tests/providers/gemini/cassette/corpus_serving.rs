//! Two tool-call turns on an id-less wire, from one recording: the
//! `gemini_tool_call_turns` golden under the default serving policy, and
//! Matrix C's Gemini cell under serial serving. The policy changes how the
//! bus serves, not what the program asks.

use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::effect::EffectFamily;
use rig::providers::gemini;
use rig_cassette::agent::AgentReplayExt;
use rig_test_support::cassette_models::GeminiModels;

use rig::tool::PortableTool;
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::with_gemini_cassette;
use crate::goldens::families;

const CHAIN_PREAMBLE: &str = "You are a calculator assistant. You MUST use the provided \
     tools for every arithmetic operation instead of computing results yourself. Perform the steps \
     in order, using the result of each step as an input to the next. Once you have the final tool \
     result, reply with the final numeric answer in plain text.";

#[derive(Deserialize)]
struct OperationArgs {
    x: i64,
    y: i64,
}

fn operands() -> Value {
    json!({
        "type": "object",
        "properties": {
            "x": { "type": "number", "description": "The first operand" },
            "y": { "type": "number", "description": "The second operand" }
        },
        "required": ["x", "y"]
    })
}

struct Add;

impl PortableTool for Add {
    const NAME: &'static str = "add";
    type Args = OperationArgs;
    type Output = i64;
    type Error = std::convert::Infallible;

    fn description(&self) -> String {
        "Add x and y together".to_owned()
    }

    fn parameters(&self) -> Value {
        operands()
    }

    async fn call(&self, args: OperationArgs) -> Result<i64, Self::Error> {
        Ok(args.x + args.y)
    }
}

struct Subtract;

impl PortableTool for Subtract {
    const NAME: &'static str = "subtract";
    type Args = OperationArgs;
    type Output = i64;
    type Error = std::convert::Infallible;

    fn description(&self) -> String {
        "Subtract y from x (i.e. x - y)".to_owned()
    }

    fn parameters(&self) -> Value {
        operands()
    }

    async fn call(&self, args: OperationArgs) -> Result<i64, Self::Error> {
        Ok(args.x - args.y)
    }
}

/// Run the two-turn chain under `policy` and return its stamped log.
async fn two_tool_turns(
    client: GeminiModels,
    policy: rig::serve::ServingPolicy,
) -> rig_cassette::effect_log::EffectLog {
    let recorder = rig_cassette::effect_log::EffectLogRecorder::new();
    let agent = rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_2_5_FLASH))
        .name("stress-agent")
        .configure_bus(policy)
        .preamble(CHAIN_PREAMBLE)
        .temperature(0.0)
        .tool(Add)
        .tool(Subtract)
        .record_to(recorder.clone())
        .build();
    let mut stream = agent
        .prompt(
            "First add 20 and 5 with the add tool. Then subtract 4 from that sum with the \
             subtract tool. Report the final number.",
        )
        .max_turns(6)
        .stream();
    let mut saw_final = false;
    while let Some(item) = stream.next().await {
        if let Ok(MultiTurnStreamItem::FinalResponse(_)) = item {
            saw_final = true;
        }
    }
    drop(stream);
    assert!(saw_final, "the stream yields a final response");
    agent.stamp(recorder.take())
}

/// Every id in the log is minted from the block that assembled the call, so
/// the record is the same on every run: nothing the engine mints is random.
#[tokio::test]
async fn tool_call_turns_effect_log_is_the_golden_fixture() {
    with_gemini_cassette(
        "hook_stress/streaming_lifecycle_ordering_and_context_streaming_flag",
        |client| async move {
            let log = two_tool_turns(client, rig::serve::ServingPolicy::default()).await;
            let tool_ids: Vec<&rig::message::CallId> = log
                .records
                .iter()
                .filter_map(|record| match &record.outcome {
                    Ok(rig::effect::Outcome::Completion(response)) => Some(response),
                    _ => None,
                })
                .flat_map(|response| response.choice.iter())
                .filter_map(|content| match content {
                    rig::message::AssistantContent::ToolCall(call) => Some(&call.id),
                    _ => None,
                })
                .collect();
            assert!(!tool_ids.is_empty(), "the program calls tools");
            assert!(
                tool_ids.iter().all(|id| id.is_local()),
                "every id-less wire call is named by its block: {tool_ids:?}"
            );
            crate::goldens::golden_effects("gemini_tool_call_turns", &log);
        },
    )
    .await;
}

#[tokio::test]
async fn two_turns_serial_effect_log_is_the_golden_fixture() {
    with_gemini_cassette(
        "hook_stress/streaming_lifecycle_ordering_and_context_streaming_flag",
        |client| async move {
            let policy = rig::serve::ServingPolicy {
                serial_per_handler: true,
                ..rig::serve::ServingPolicy::default()
            };
            let log = two_tool_turns(client, policy).await;
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Completion
                ]
            );
            assert_eq!(log.header.bus.map(|bus| bus.serial_per_handler), Some(true));
            crate::goldens::golden_effects("gemini_serving_two_turns_serial", &log);
        },
    )
    .await;
}
