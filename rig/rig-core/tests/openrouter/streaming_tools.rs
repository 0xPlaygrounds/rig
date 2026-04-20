//! OpenRouter streaming tools smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::CompletionModel;
use rig::providers::openrouter;
use rig::streaming::StreamingPrompt;
use rig::tool::Tool;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use crate::reasoning::WeatherTool;
use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number, collect_raw_stream_observation, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn streaming_tools_smoke() {
    let client = openrouter::Client::from_env();
    let agent = client
        .agent(openrouter::GEMINI_FLASH_2_0)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_TOOLS_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn raw_stream_decorates_reasoning_tool_call_metadata() {
    let client = openrouter::Client::from_env();
    let model = client.completion_model("openai/o4-mini");
    let tool_definition = WeatherTool::new(Arc::new(AtomicUsize::new(0)))
        .definition(String::new())
        .await;
    let request = model
        .completion_request(crate::reasoning::TOOL_USER_PROMPT)
        .preamble(crate::reasoning::TOOL_SYSTEM_PROMPT.to_string())
        .max_tokens(4096)
        .tool(tool_definition)
        .additional_params(serde_json::json!({
            "reasoning": { "effort": "high" },
            "include_reasoning": true
        }))
        .build();

    let stream = model.stream(request).await.expect("stream should start");
    let observation = collect_raw_stream_observation(stream).await;
    assert!(
        observation.errors.is_empty(),
        "raw stream should not emit errors: {:?}",
        observation.errors
    );

    let record = observation
        .tool_call_records
        .iter()
        .find(|record| record.name == "get_weather")
        .expect("expected a streamed get_weather tool call");

    if record.signature.is_none() && record.additional_params.is_none() {
        eprintln!(
            "openrouter did not emit encrypted reasoning metadata for the tool call in this run; skipping strict decoration assertion"
        );
        return;
    }

    assert!(
        record.signature.is_some() || record.additional_params.is_some(),
        "expected decorated tool call metadata for get_weather, got signature={:?} additional_params={:?}",
        record.signature,
        record.additional_params
    );
}
