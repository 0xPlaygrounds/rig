//! Dedicated Claude Opus 4.7 live smoke tests.

use base64::{Engine, prelude::BASE64_STANDARD};
use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::message::Image;
use rig_core::completion::{Chat, Message, Prompt};
use rig_core::message::{DocumentSourceKind, ImageMediaType};
use rig_core::providers::anthropic::{self, completion::CLAUDE_OPUS_4_7};
use rig_core::streaming::{StreamingChat, StreamingPrompt};

use crate::reasoning::{self, ReasoningRoundtripAgent, WeatherTool};
use crate::support::{
    Adder, BASIC_PREAMBLE, BASIC_PROMPT, EXTRACTOR_TEXT, IMAGE_FIXTURE_PATH, STREAMING_PREAMBLE,
    STREAMING_PROMPT, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, STRUCTURED_OUTPUT_PROMPT,
    SmokePerson, SmokeStructuredOutput, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT,
    assert_contains_any_case_insensitive, assert_mentions_expected_number,
    assert_nonempty_response, assert_smoke_structured_output, collect_stream_final_response,
};

fn opus_4_7_thinking_params() -> serde_json::Value {
    serde_json::json!({
        "thinking": { "type": "adaptive" }
    })
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn messages_prompt_smoke() {
    let client = anthropic::Client::from_env().expect("client should build");
    let agent = client
        .agent(CLAUDE_OPUS_4_7)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn messages_streaming_prompt_smoke() {
    let client = anthropic::Client::from_env().expect("client should build");
    let agent = client
        .agent(CLAUDE_OPUS_4_7)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn messages_tools_smoke() {
    let client = anthropic::Client::from_env().expect("client should build");
    let agent = client
        .agent(CLAUDE_OPUS_4_7)
        .preamble(TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let response = agent
        .prompt(TOOLS_PROMPT)
        .await
        .expect("tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn messages_streaming_tools_smoke() {
    let client = anthropic::Client::from_env().expect("client should build");
    let agent = client
        .agent(CLAUDE_OPUS_4_7)
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
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn messages_structured_output_smoke() {
    let client = anthropic::Client::from_env().expect("client should build");
    let agent = client
        .agent(CLAUDE_OPUS_4_7)
        .output_schema::<SmokeStructuredOutput>()
        .build();

    let response = agent
        .prompt(STRUCTURED_OUTPUT_PROMPT)
        .await
        .expect("structured output prompt should succeed");
    let structured: SmokeStructuredOutput =
        serde_json::from_str(&response).expect("structured output should deserialize");

    assert_smoke_structured_output(&structured);
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn messages_extractor_smoke() {
    let client = anthropic::Client::from_env().expect("client should build");
    let extractor = client.extractor::<SmokePerson>(CLAUDE_OPUS_4_7).build();

    let response = extractor
        .extract_with_usage(EXTRACTOR_TEXT)
        .await
        .expect("extractor request should succeed");

    assert_nonempty_response(
        response
            .data
            .first_name
            .as_deref()
            .expect("first name should be present"),
    );
    assert_nonempty_response(
        response
            .data
            .last_name
            .as_deref()
            .expect("last name should be present"),
    );
    assert!(response.usage.total_tokens > 0, "usage should be populated");
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn messages_image_input_smoke() {
    let client = anthropic::Client::from_env().expect("client should build");
    let agent = client
        .agent(CLAUDE_OPUS_4_7)
        .preamble("You are an image describer.")
        .build();
    let image_bytes = std::fs::read(IMAGE_FIXTURE_PATH).expect("fixture image should be readable");
    let image = Image {
        data: DocumentSourceKind::base64(&BASE64_STANDARD.encode(image_bytes)),
        media_type: Some(ImageMediaType::JPEG),
        ..Default::default()
    };

    let response = agent
        .prompt(image)
        .await
        .expect("image prompt should succeed");

    assert_nonempty_response(&response);
    assert_contains_any_case_insensitive(&response, &["ant", "insect"]);
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn messages_adaptive_thinking_nonstreaming_smoke() {
    let client = anthropic::Client::from_env().expect("client should build");
    reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
        client.completion_model(CLAUDE_OPUS_4_7),
        Some(opus_4_7_thinking_params()),
    ))
    .await;
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn messages_adaptive_thinking_streaming_smoke() {
    let client = anthropic::Client::from_env().expect("client should build");
    reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
        client.completion_model(CLAUDE_OPUS_4_7),
        Some(opus_4_7_thinking_params()),
    ))
    .await;
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn messages_adaptive_thinking_tool_roundtrip_smoke() {
    let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let client = anthropic::Client::from_env().expect("client should build");
    let agent = client
        .agent(CLAUDE_OPUS_4_7)
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(16384)
        .tool(WeatherTool::new(call_count.clone()))
        .additional_params(opus_4_7_thinking_params())
        .build();

    let result = agent
        .chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
        .await
        .expect("adaptive thinking tool chat should succeed");

    reasoning::assert_nonstreaming_universal(&result, &call_count, "anthropic");
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn messages_adaptive_thinking_streaming_tool_roundtrip_smoke() {
    let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let client = anthropic::Client::from_env().expect("client should build");
    let agent = client
        .agent(CLAUDE_OPUS_4_7)
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(16384)
        .tool(WeatherTool::new(call_count.clone()))
        .additional_params(opus_4_7_thinking_params())
        .build();

    let stream = agent
        .stream_chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
        .multi_turn(3)
        .await;

    let stats = reasoning::collect_stream_stats(stream, "anthropic").await;
    reasoning::assert_universal(&stats, &call_count, "anthropic");
}
