//! Anthropic structured output smoke test.

use rig::agent::OutputMode;
use rig::completion::Prompt;
use rig::prelude::*;
use rig::providers::anthropic::{self, completion::CLAUDE_SONNET_4_6};
use rig::test_utils::RecordingHttpClient;
use rig_agent::test_utils::decode_structured_output;

use super::super::support::with_anthropic_cassette;
use crate::support::{
    STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput, assert_smoke_structured_output,
    smoke_structured_output_value,
};

fn text_response(text: &str) -> String {
    serde_json::json!({
        "content": [{"type": "text", "text": text}],
        "id": "msg_runtime_acceptance",
        "model": CLAUDE_SONNET_4_6,
        "role": "assistant",
        "type": "message",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_read_input_tokens": null,
            "cache_creation_input_tokens": null
        }
    })
    .to_string()
}

fn output_tool_response(name: &str) -> String {
    serde_json::json!({
        "content": [{
            "type": "tool_use",
            "id": "toolu_runtime_acceptance",
            "name": name,
            "input": smoke_structured_output_value()
        }],
        "id": "msg_runtime_acceptance",
        "model": CLAUDE_SONNET_4_6,
        "role": "assistant",
        "type": "message",
        "stop_reason": "tool_use",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_read_input_tokens": null,
            "cache_creation_input_tokens": null
        }
    })
    .to_string()
}

#[tokio::test]
async fn structured_output_smoke() {
    with_anthropic_cassette(
        "structured_output/structured_output_smoke",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .output_schema::<SmokeStructuredOutput>()
                .build();

            let response = agent
                .prompt(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("structured output prompt should succeed");
            let structured: SmokeStructuredOutput =
                decode_structured_output("anthropic_structured_output_smoke", &response)
                    .expect("structured output should deserialize");

            assert_smoke_structured_output(&structured);
        },
    )
    .await;
}

#[tokio::test]
async fn classic_tool_mode_maps_through_anthropic_messages() {
    let http = RecordingHttpClient::new(output_tool_response("final_result"));
    let client = anthropic::Client::builder()
        .api_key("test-key")
        .http_client(http.clone())
        .build()
        .expect("Anthropic test client should build");
    let agent = client
        .agent(CLAUDE_SONNET_4_6)
        .output_schema::<SmokeStructuredOutput>()
        .output_mode(OutputMode::Tool)
        .build();

    let response = agent
        .prompt(STRUCTURED_OUTPUT_PROMPT)
        .await
        .expect("classic Anthropic Tool-mode run should succeed");
    let structured: SmokeStructuredOutput =
        decode_structured_output("anthropic_classic_tool_mode", &response)
            .expect("output-tool arguments should deserialize");
    assert_smoke_structured_output(&structured);

    let requests = http.requests();
    assert_eq!(requests.len(), 1);
    let body: serde_json::Value =
        serde_json::from_slice(&requests[0].body).expect("request should be JSON");
    assert_eq!(body["tools"][0]["name"], "final_result");
    assert!(body.get("output_config").is_none());
}

#[tokio::test]
async fn classic_prompted_mode_maps_through_anthropic_messages() {
    let output = smoke_structured_output_value();
    let http = RecordingHttpClient::new(text_response(&output.to_string()));
    let client = anthropic::Client::builder()
        .api_key("test-key")
        .http_client(http.clone())
        .build()
        .expect("Anthropic test client should build");
    let agent = client
        .agent(CLAUDE_SONNET_4_6)
        .output_schema::<SmokeStructuredOutput>()
        .output_mode(OutputMode::Prompted)
        .build();

    let response = agent
        .prompt(STRUCTURED_OUTPUT_PROMPT)
        .await
        .expect("classic Anthropic Prompted-mode run should succeed");
    let structured: SmokeStructuredOutput =
        decode_structured_output("anthropic_classic_prompted_mode", &response)
            .expect("prompted JSON should deserialize");
    assert_smoke_structured_output(&structured);

    let requests = http.requests();
    assert_eq!(requests.len(), 1);
    let body: serde_json::Value =
        serde_json::from_slice(&requests[0].body).expect("request should be JSON");
    assert!(body.get("tools").is_none());
    assert!(body.get("output_config").is_none());
    assert!(
        body["system"][0]["text"]
            .as_str()
            .is_some_and(|text| text.contains("JSON Schema") && text.contains("ONLY"))
    );
}
