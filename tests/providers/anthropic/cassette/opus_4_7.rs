//! Dedicated Claude Opus 4.7 live smoke tests.

use std::sync::Arc;

use base64::{Engine, prelude::BASE64_STANDARD};
use rig::agent::AgentConfig;
use rig::completion::Message;
use rig::completion::message::Image;
use rig::extract::{ExtractOptions, extract_with_options};
use rig::message::{DocumentSourceKind, ImageMediaType};
use rig::provider::Runtime;
use rig::providers::anthropic::completion::CLAUDE_OPUS_4_7;
use rig_agent::test_utils::validate_extraction_fields;

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
async fn messages_prompt_smoke() {
    super::super::support::with_anthropic_cassette(
        "opus_4_7/messages_prompt_smoke",
        |client| async move {
            let agent = client
                .agent(CLAUDE_OPUS_4_7)
                .preamble(BASIC_PREAMBLE)
                .build();

            let response = agent
                .prompt(BASIC_PROMPT)
                .await
                .expect("prompt should succeed");

            assert_nonempty_response(&response);
        },
    )
    .await;
}

#[tokio::test]
async fn messages_streaming_prompt_smoke() {
    super::super::support::with_anthropic_cassette(
        "opus_4_7/messages_streaming_prompt_smoke",
        |client| async move {
            let agent = client
                .agent(CLAUDE_OPUS_4_7)
                .preamble(STREAMING_PREAMBLE)
                .build();

            let mut stream = Box::pin(agent.runner(STREAMING_PROMPT).stream_run());
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming prompt should succeed");

            assert_nonempty_response(&response);
        },
    )
    .await;
}

#[tokio::test]
async fn messages_tools_smoke() {
    super::super::support::with_anthropic_cassette(
        "opus_4_7/messages_tools_smoke",
        |client| async move {
            let agent = client
                .agent(CLAUDE_OPUS_4_7)
                .preamble(TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .default_max_turns(2)
                .build();

            let response = agent
                .prompt(TOOLS_PROMPT)
                .await
                .expect("tool prompt should succeed");

            assert_mentions_expected_number(&response, -3);
        },
    )
    .await;
}

#[tokio::test]
async fn messages_streaming_tools_smoke() {
    super::super::support::with_anthropic_cassette(
        "opus_4_7/messages_streaming_tools_smoke",
        |client| async move {
            let agent = client
                .agent(CLAUDE_OPUS_4_7)
                .preamble(STREAMING_TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .default_max_turns(2)
                .build();

            let mut stream = Box::pin(agent.runner(STREAMING_TOOLS_PROMPT).stream_run());
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming tool prompt should succeed");

            assert_mentions_expected_number(&response, -3);
        },
    )
    .await;
}

#[tokio::test]
async fn messages_structured_output_smoke() {
    super::super::support::with_anthropic_cassette(
        "opus_4_7/messages_structured_output_smoke",
        |client| async move {
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
        },
    )
    .await;
}

#[tokio::test]
async fn messages_extractor_smoke() {
    super::super::support::with_anthropic_cassette(
        "opus_4_7/messages_extractor_smoke",
        |client| async move {
            let response = extract_with_options::<SmokePerson>(
                AgentConfig::new(),
                client.provider_config(CLAUDE_OPUS_4_7),
                Arc::new(Runtime::new()),
                EXTRACTOR_TEXT,
                ExtractOptions::classic_extractor(),
            )
            .await
            .expect("extractor request should succeed");

            validate_extraction_fields(
                "anthropic_opus_4_7_extractor_smoke",
                response.value.first_name.as_deref(),
                response.value.last_name.as_deref(),
                response.value.job.as_deref(),
                response.usage,
            )
            .expect("portable extraction contract should hold");

            assert_nonempty_response(
                response
                    .value
                    .first_name
                    .as_deref()
                    .expect("first name should be present"),
            );
            assert_nonempty_response(
                response
                    .value
                    .last_name
                    .as_deref()
                    .expect("last name should be present"),
            );
            assert!(response.usage.total_tokens > 0, "usage should be populated");
        },
    )
    .await;
}

#[tokio::test]
async fn messages_image_input_smoke() {
    super::super::support::with_anthropic_cassette(
        "opus_4_7/messages_image_input_smoke",
        |client| async move {
            let agent = client
                .agent(CLAUDE_OPUS_4_7)
                .preamble("You are an image describer.")
                .build();
            let image_bytes =
                std::fs::read(IMAGE_FIXTURE_PATH).expect("fixture image should be readable");
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
        },
    )
    .await;
}

#[tokio::test]
async fn messages_adaptive_thinking_nonstreaming_smoke() {
    super::super::support::with_anthropic_cassette(
        "opus_4_7/messages_adaptive_thinking_nonstreaming_smoke",
        |client| async move {
            reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
                client.provider_config(CLAUDE_OPUS_4_7),
                Some(opus_4_7_thinking_params()),
            ))
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn messages_adaptive_thinking_streaming_smoke() {
    super::super::support::with_anthropic_cassette(
        "opus_4_7/messages_adaptive_thinking_streaming_smoke",
        |client| async move {
            reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
                client.provider_config(CLAUDE_OPUS_4_7),
                Some(opus_4_7_thinking_params()),
            ))
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn messages_adaptive_thinking_tool_roundtrip_smoke() {
    let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    super::super::support::with_anthropic_cassette(
        "opus_4_7/messages_adaptive_thinking_tool_roundtrip_smoke",
        |client| async move {
            let agent = client
                .agent(CLAUDE_OPUS_4_7)
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(16384)
                .tool(WeatherTool::new(call_count.clone()))
                .additional_params(opus_4_7_thinking_params())
                .default_max_turns(2)
                .build();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
                .await
                .expect("adaptive thinking tool chat should succeed");

            reasoning::assert_nonstreaming_universal(&result, &call_count, "anthropic");
        },
    )
    .await;
}

#[tokio::test]
async fn messages_adaptive_thinking_streaming_tool_roundtrip_smoke() {
    let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    super::super::support::with_anthropic_cassette(
        "opus_4_7/messages_adaptive_thinking_streaming_tool_roundtrip_smoke",
        |client| async move {
            let agent = client
                .agent(CLAUDE_OPUS_4_7)
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(16384)
                .tool(WeatherTool::new(call_count.clone()))
                .additional_params(opus_4_7_thinking_params())
                .build();

            let stream = Box::pin(
                agent
                    .runner(reasoning::TOOL_USER_PROMPT)
                    .history(Vec::<Message>::new())
                    .max_turns(3)
                    .stream_run(),
            );

            let stats = reasoning::collect_stream_stats(stream, "anthropic").await;
            reasoning::assert_universal(&stats, &call_count, "anthropic");
        },
    )
    .await;
}
