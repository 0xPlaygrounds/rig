//! xAI streaming tools smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, ToolDefinition};
use rig::message::ToolChoice;
use rig::providers::xai;
use rig::streaming::StreamingPrompt;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::support::{
    REQUIRED_ZERO_ARG_TOOL_PROMPT, assert_stream_contains_zero_arg_tool_call_named,
    assert_tool_call_precedes_later_text, collect_stream_observation, zero_arg_tool_definition,
};

const XAI_STATUS_TOOL_PREAMBLE: &str = "\
You are a terse assistant. A function named `get_status_word` is available. \
Call that function once before answering. After the function result is available, \
answer in one short sentence that repeats the exact returned word.";
const XAI_STATUS_TOOL_PROMPT: &str = "Use the get_status_word function once, then reply with the exact returned word in one short sentence.";
const XAI_STATUS_TOOL_OUTPUT: &str = "azure";

#[derive(Deserialize, Serialize)]
struct NoArgs {}

#[derive(Debug, thiserror::Error)]
#[error("status tool error")]
struct StatusToolError;

#[derive(Deserialize, Serialize)]
struct StatusWordTool;

impl Tool for StatusWordTool {
    const NAME: &'static str = "get_status_word";
    type Error = StatusToolError;
    type Args = NoArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Return a harmless status word.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": [],
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(XAI_STATUS_TOOL_OUTPUT.to_string())
    }
}

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn raw_stream_emits_required_zero_arg_tool_call() {
    let client = xai::Client::from_env();
    let model = client.completion_model(xai::completion::GROK_4);
    let request = model
        .completion_request(REQUIRED_ZERO_ARG_TOOL_PROMPT)
        .tool(zero_arg_tool_definition("ping"))
        .tool_choice(ToolChoice::Required)
        .build();
    let stream = model.stream(request).await.expect("stream should start");

    assert_stream_contains_zero_arg_tool_call_named(stream, "ping", true).await;
}

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn responses_stream_preserves_tool_result_flow() {
    let client = xai::Client::from_env();
    let agent = client
        .agent(xai::completion::GROK_4)
        .preamble(XAI_STATUS_TOOL_PREAMBLE)
        .tool(StatusWordTool)
        .build();

    let mut stream = agent
        .stream_prompt(XAI_STATUS_TOOL_PROMPT)
        .multi_turn(5)
        .await;
    let observation = collect_stream_observation(&mut stream).await;

    assert_tool_call_precedes_later_text(
        &observation,
        "get_status_word",
        &[XAI_STATUS_TOOL_OUTPUT],
    );
}
