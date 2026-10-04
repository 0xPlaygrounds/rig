//! Gemini tool-choice cassette coverage.

use rig::message::ToolChoice;
use rig::providers::gemini;

use crate::support::{
    Adder, Subtract, assert_mentions_expected_number, collect_stream_observation,
};

#[tokio::test]
async fn none_streaming_does_not_emit_tool_calls() {
    super::super::support::with_gemini_cassette(
        "tool_choice/none_streaming_no_tools",
        |client| async move {
            let agent =
                rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_2_5_FLASH))
                    .preamble("You are a deterministic calculator test. Answer directly in text.")
                    .temperature(0.0)
                    .tool(Adder)
                    .tool(Subtract)
                    .tool_choice(ToolChoice::None)
                    .build();

            let mut stream = agent
                .prompt("Calculate 20 + 22 directly in text. Do not call tools.")
                .stream();
            let observation = collect_stream_observation(&mut stream).await;

            assert!(
                observation.errors.is_empty(),
                "stream should not emit errors: {:?}",
                observation.errors
            );
            assert!(
                observation.got_final_response,
                "stream should emit a final response"
            );
            assert!(
                observation.tool_calls.is_empty(),
                "expected no tool calls, saw {:?}",
                observation.tool_calls
            );
            assert_eq!(observation.tool_results, 0, "expected no tool results");
            assert_mentions_expected_number(
                observation
                    .final_response_text
                    .as_deref()
                    .expect("stream should produce a final response"),
                42,
            );
        },
    )
    .await;
}
