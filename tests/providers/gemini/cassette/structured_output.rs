//! Gemini structured output smoke test.

use rig::agent::OutputMode;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::gemini;
use rig::test_utils::{MockHttpResponse, SequencedHttpClient};
use rig_agent::test_utils::decode_structured_output;

use super::super::support::with_gemini_cassette;
use crate::support::{
    STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput, assert_smoke_structured_output,
    smoke_structured_output_value,
};

fn text_response(text: &str, response_id: &str) -> String {
    serde_json::json!({
        "candidates": [{
            "content": {
                "parts": [{"text": text}],
                "role": "model"
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "modelVersion": gemini::completion::GEMINI_2_5_FLASH,
        "responseId": response_id,
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 1,
            "totalTokenCount": 2
        }
    })
    .to_string()
}

fn output_tool_response(name: &str) -> String {
    serde_json::json!({
        "candidates": [{
            "content": {
                "parts": [{
                    "functionCall": {
                        "args": smoke_structured_output_value(),
                        "name": name
                    }
                }],
                "role": "model"
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "modelVersion": gemini::completion::GEMINI_2_5_FLASH,
        "responseId": "gemini-runtime-recovery-final",
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 1,
            "totalTokenCount": 2
        }
    })
    .to_string()
}

#[tokio::test]
async fn structured_output_smoke() {
    with_gemini_cassette(
        "structured_output/structured_output_smoke",
        |client| async move {
            let agent = client
                .agent("gemini-3-flash-preview")
                .output_schema::<SmokeStructuredOutput>()
                .output_mode(OutputMode::Native)
                .build();

            let response = agent
                .prompt(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("structured output prompt should succeed");
            let structured: SmokeStructuredOutput =
                decode_structured_output("gemini_structured_output_smoke", &response)
                    .expect("structured output should deserialize");

            assert_smoke_structured_output(&structured);
        },
    )
    .await;
}

#[tokio::test]
async fn classic_invalid_output_recovers_through_gemini_generate_content() {
    let http = SequencedHttpClient::new([
        MockHttpResponse::success(text_response("not valid JSON", "gemini-runtime-invalid")),
        MockHttpResponse::success(output_tool_response("final_result")),
    ]);
    let client = gemini::Client::builder()
        .api_key("test-key")
        .http_client(http.clone())
        .build()
        .expect("Gemini test client should build");
    let agent = client
        .agent(gemini::completion::GEMINI_2_5_FLASH)
        .output_schema::<SmokeStructuredOutput>()
        .output_mode(OutputMode::Tool)
        .default_max_turns(2)
        .build();

    let response = agent
        .prompt(STRUCTURED_OUTPUT_PROMPT)
        .await
        .expect("classic Gemini output recovery should succeed");
    let structured: SmokeStructuredOutput =
        decode_structured_output("gemini_classic_recovery", &response)
            .expect("recovered output-tool arguments should deserialize");
    assert_smoke_structured_output(&structured);

    let requests = http.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(http.remaining_responses(), 0);
    let first: serde_json::Value =
        serde_json::from_slice(&requests[0].body).expect("request should be JSON");
    let second: serde_json::Value =
        serde_json::from_slice(&requests[1].body).expect("request should be JSON");
    assert_eq!(
        first["tools"][0]["functionDeclarations"][0]["name"],
        "final_result"
    );
    assert!(
        first
            .pointer("/generationConfig/responseJsonSchema")
            .is_none()
    );
    assert!(second.to_string().contains("final_result"));
}
