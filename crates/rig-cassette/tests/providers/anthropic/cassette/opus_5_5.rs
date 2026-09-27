//! Claude Opus 5.5 cassette coverage.
//!
//! Opus 5.5 rejects a forced tool choice and always thinks. The extractor
//! cell proves extraction works with no user change: the recorded request
//! asks for native structured output instead of forcing `submit`. The
//! streaming cell runs a tool loop with `display: "updates"`, whose beta flag
//! and replayed thinking blocks the recorded requests carry.

use rig::providers::anthropic::completion::{
    CLAUDE_OPUS_5_5, Effort, THINKING_DISPLAY_UPDATES_BETA, Thinking, ThinkingDisplay,
};
use rig_agent::test_utils::validate_extraction_fields;
use serde_json::Value;

use crate::support::{
    Adder, EXTRACTOR_TEXT, STREAMING_TOOLS_PREAMBLE, SmokePerson, Subtract,
    assert_mentions_expected_number, collect_stream_final_response,
};

/// Every recorded request body of `scenario`, decoded, in wire order.
fn recorded_requests(scenario: &str) -> Vec<Value> {
    crate::cassettes::recorded_interaction_bodies("anthropic", scenario)
        .into_iter()
        .map(|(request, _)| serde_json::from_str(&request).expect("request body is JSON"))
        .collect()
}

#[tokio::test]
async fn extractor_uses_native_output() {
    super::super::support::with_anthropic_cassette(
        "opus_5_5/extractor_uses_native_output",
        |client| async move {
            let extractor = rig::extractor::ExtractorBuilder::<SmokePerson>::new(
                client.completion(CLAUDE_OPUS_5_5),
            )
            .build();

            let response = extractor
                .extract(EXTRACTOR_TEXT)
                .await
                .expect("extraction succeeds on a model that rejects forced tool choice");

            validate_extraction_fields(
                "anthropic_opus_5_5_extractor",
                response.output.first_name.as_deref(),
                response.output.last_name.as_deref(),
                response.output.job.as_deref(),
                response.usage,
            )
            .expect("portable extraction contract should hold");
        },
    )
    .await;

    let requests = recorded_requests("opus_5_5/extractor_uses_native_output");
    assert_eq!(requests.len(), 1, "one extraction, one request");
    assert_eq!(requests[0].get("tool_choice"), None);
    assert_eq!(requests[0].get("tools"), None);
    assert_eq!(
        requests[0]["output_config"]["format"]["type"],
        "json_schema"
    );
}

/// A task with several tool calls, so the model thinks and narrates between
/// them.
const MULTI_STEP_PROMPT: &str = "Compute (17 - 5) + (8 - 23). Use one tool call per step, and say \
     in one short sentence what you found before each next call.";

#[tokio::test]
async fn streaming_tool_loop_with_progress_updates() {
    super::super::support::with_anthropic_cassette(
        "opus_5_5/streaming_tool_loop_with_progress_updates",
        |client| async move {
            let mut model = client.completion(CLAUDE_OPUS_5_5);
            model.wire = model
                .wire
                .with_effort(Effort::High)
                .with_thinking(Thinking::adaptive().with_display(ThinkingDisplay::Updates));
            let agent = rig::AgentBuilder::new(model)
                .preamble(STREAMING_TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .max_tokens(32768)
                .default_max_turns(6)
                .build();

            let mut stream = agent.prompt(MULTI_STEP_PROMPT).stream();
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("the streamed tool loop completes");

            assert_mentions_expected_number(&response, -3);
        },
    )
    .await;

    let requests = recorded_requests("opus_5_5/streaming_tool_loop_with_progress_updates");
    assert!(requests.len() >= 2, "tool calls and their follow-ups");
    for request in &requests {
        assert_eq!(request["thinking"]["display"], "updates");
        assert_eq!(request["output_config"]["effort"], "high");
    }
    // Every replayed thinking block carries its signature, including empty ones.
    let replayed: Vec<&Value> = requests
        .iter()
        .flat_map(|request| request["messages"].as_array().into_iter().flatten())
        .filter(|message| message["role"] == "assistant")
        .flat_map(|message| message["content"].as_array().into_iter().flatten())
        .filter(|block| block["type"] == "thinking")
        .collect();
    assert!(!replayed.is_empty(), "the loop replays thinking blocks");
    assert!(
        replayed.iter().all(|block| block["signature"]
            .as_str()
            .is_some_and(|sig| !sig.is_empty())),
        "every replayed thinking block keeps its signature"
    );

    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/cassettes/anthropic")
        .join("opus_5_5/streaming_tool_loop_with_progress_updates.yaml");
    let recorded = std::fs::read_to_string(&fixture).expect("the fixture is committed");
    assert_eq!(
        recorded
            .matches(&format!("value: {THINKING_DISPLAY_UPDATES_BETA}"))
            .count(),
        requests.len(),
        "every request carries the updates beta flag"
    );
}
