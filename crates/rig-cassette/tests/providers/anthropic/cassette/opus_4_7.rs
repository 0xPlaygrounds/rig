//! Dedicated Claude Opus 4.7 live smoke tests.

use rig::completion::Message;
use rig::providers::anthropic::completion::CLAUDE_OPUS_4_7;
use rig_agent::test_utils::validate_extraction_fields;

use crate::reasoning::{self, WeatherTool};
use crate::support::{
    Adder, EXTRACTOR_TEXT, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT,
    STRUCTURED_OUTPUT_PROMPT, SmokePerson, SmokeStructuredOutput, Subtract, TOOLS_PREAMBLE,
    TOOLS_PROMPT, assert_mentions_expected_number, assert_nonempty_response,
    assert_smoke_structured_output, collect_stream_final_response,
};

fn opus_4_7_thinking_params() -> serde_json::Value {
    serde_json::json!({
        "thinking": { "type": "adaptive" }
    })
}

#[tokio::test]
async fn messages_tools_smoke() {
    super::super::support::with_anthropic_cassette(
        "opus_4_7/messages_tools_smoke",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(CLAUDE_OPUS_4_7))
                .preamble(TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .default_max_turns(2)
                .build();

            let response = agent
                .prompt(TOOLS_PROMPT)
                .await
                .expect("tool prompt should succeed")
                .output();

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
            let agent = rig::AgentBuilder::new(client.completion(CLAUDE_OPUS_4_7))
                .preamble(STREAMING_TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .default_max_turns(2)
                .build();

            let mut stream = agent.prompt(STREAMING_TOOLS_PROMPT).stream();
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
            let agent = rig::AgentBuilder::new(client.completion(CLAUDE_OPUS_4_7))
                .output_schema::<SmokeStructuredOutput>()
                .build();

            let response = agent
                .prompt(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("structured output prompt should succeed")
                .output();
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
            let extractor = rig::extractor::ExtractorBuilder::<SmokePerson>::new(
                client.completion(CLAUDE_OPUS_4_7),
            )
            .build();

            let response = extractor
                .extract(EXTRACTOR_TEXT)
                .await
                .expect("extractor request should succeed");

            validate_extraction_fields(
                "anthropic_opus_4_7_extractor_smoke",
                response.output.first_name.as_deref(),
                response.output.last_name.as_deref(),
                response.output.job.as_deref(),
                response.usage,
            )
            .expect("portable extraction contract should hold");

            assert_nonempty_response(
                response
                    .output
                    .first_name
                    .as_deref()
                    .expect("first name should be present"),
            );
            assert_nonempty_response(
                response
                    .output
                    .last_name
                    .as_deref()
                    .expect("last name should be present"),
            );
            assert!(
                response.usage.total_tokens.is_some_and(|n| n > 0),
                "usage should be populated"
            );
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
            let agent = rig::AgentBuilder::new(client.completion(CLAUDE_OPUS_4_7))
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(16384)
                .tool(WeatherTool::new(call_count.clone()))
                .additional_params(opus_4_7_thinking_params())
                .default_max_turns(2)
                .build();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
                .await
                .expect("adaptive thinking tool chat should succeed")
                .output();

            reasoning::assert_nonstreaming_universal(&result, &call_count, "anthropic");
        },
    )
    .await;
}
