//! Cassette coverage for mistral.rs OpenAI-compatible tool call responses.

use serde_json::Value;

use super::super::support::{
    DEFAULT_API_KEY, SYSTEM_PROMPT, model_name, with_mistralrs_raw_cassette,
};

#[tokio::test]
async fn raw_chat_completion_emits_requested_tool_call() {
    with_mistralrs_raw_cassette(
        "tools/raw_chat_completion_emits_requested_tool_call",
        |base_url| async move {
            let raw = reqwest::Client::new()
                .post(format!("{}/chat/completions", base_url.trim_end_matches('/')))
                .bearer_auth(
                    std::env::var("MISTRALRS_API_KEY")
                        .unwrap_or_else(|_| DEFAULT_API_KEY.to_string()),
                )
                .json(&serde_json::json!({
                    "model": model_name(),
                    "messages": [
                        { "role": "system", "content": SYSTEM_PROMPT },
                        {
                            "role": "user",
                            "content": "/no_think Call the report_usage tool with reason set to cost_tracking."
                        }
                    ],
                    "max_tokens": 128,
                    "temperature": 0.2,
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "report_usage",
                            "description": "Report why token usage is needed.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "reason": { "type": "string" }
                                },
                                "required": ["reason"]
                            }
                        }
                    }],
                    "tool_choice": {
                        "type": "function",
                        "function": { "name": "report_usage" }
                    }
                }))
                .send()
                .await
                .expect("raw tool call request should be sent")
                .error_for_status()
                .expect("raw tool call response should be successful")
                .json::<Value>()
                .await
                .expect("raw tool call response should deserialize");
            let tool_calls = raw["choices"][0]["message"]["tool_calls"]
                .as_array()
                .expect("mistral.rs response should include tool_calls");

            assert!(
                !tool_calls.is_empty(),
                "mistral.rs tool call response should include at least one tool call: {raw:?}"
            );
            assert_eq!(
                tool_calls[0]["function"]["name"].as_str(),
                Some("report_usage")
            );
            assert!(
                raw.get("usage")
                    .and_then(|usage| usage.get("total_tokens"))
                    .and_then(Value::as_u64)
                    .is_some(),
                "tool call response should include usage: {raw:?}"
            );
        },
    )
    .await;
}
