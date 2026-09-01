//! OpenAI-compatible response regressions that use an in-memory HTTP backend.

use rig::prelude::*;
use rig::providers::openai;
use rig_core::test_utils::RecordingHttpClient;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
struct KeywordPayload {
    keywords: Vec<String>,
}

#[tokio::test]
async fn extractor_accepts_nullable_strict_in_echoed_tool_definition() {
    let response = json!({
        "id": "resp_local",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "error": null,
        "incomplete_details": null,
        "instructions": null,
        "max_output_tokens": null,
        "model": "gpt-oss-120b",
        "usage": {
            "input_tokens": 42,
            "input_tokens_details": { "cached_tokens": 0 },
            "output_tokens": 12,
            "output_tokens_details": { "reasoning_tokens": 0 },
            "total_tokens": 54
        },
        "output": [{
            "type": "function_call",
            "id": "fc_submit",
            "call_id": "call_submit",
            "name": "submit",
            "arguments": r#"{"keywords":["fruit","produce"]}"#,
            "status": "completed"
        }],
        "tools": [{
            "type": "function",
            "name": "submit",
            "description": "Submit the extracted keywords.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": { "type": "string" }
                    }
                },
                "required": ["keywords"]
            },
            "strict": null
        }]
    });
    let http_client = RecordingHttpClient::new(response.to_string());
    let client = openai::Client::builder()
        .api_key("test-key")
        .base_url("http://localhost:8000/v1")
        .http_client(http_client.clone())
        .build()
        .expect("OpenAI-compatible client should build");

    let extracted = client
        .extractor::<KeywordPayload>("gpt-oss-120b")
        .build()
        .extract("What fruit is mentioned in the database?")
        .await
        .expect("nullable strict should not prevent extraction");

    assert_eq!(
        extracted,
        KeywordPayload {
            keywords: vec!["fruit".to_string(), "produce".to_string()]
        }
    );

    let requests = http_client.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].uri, "http://localhost:8000/v1/responses");
    let request: serde_json::Value =
        serde_json::from_slice(&requests[0].body).expect("request body should be valid JSON");
    assert_eq!(request["tools"][0]["name"], "submit");
    assert!(
        request["tools"][0].get("strict").is_none(),
        "non-strict extractor tools should omit strict on the request"
    );
}
