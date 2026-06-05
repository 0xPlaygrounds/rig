//! Cassette coverage for mistral.rs usage without OpenAI `output_tokens_details`.

use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use serde_json::Value;

use super::super::support::{SYSTEM_PROMPT, model_name, with_mistralrs_completions_cassette};

#[tokio::test]
async fn chat_completion_usage_without_output_tokens_details_deserializes() {
    with_mistralrs_completions_cassette(
        "usage/chat_completion_usage_without_output_tokens_details_deserializes",
        |client| async move {
            let response = client
                .completion_model(model_name())
                .completion_request("/no_think Explain usage accounting in one sentence.")
                .preamble(SYSTEM_PROMPT.to_string())
                .max_tokens(64)
                .send()
                .await
                .expect("usage check completion should succeed");
            let raw = serde_json::to_value(&response.raw_response)
                .expect("raw chat completion response should serialize");
            let usage = raw
                .get("usage")
                .expect("mistral.rs response should include usage");

            for field in ["prompt_tokens", "total_tokens"] {
                assert!(
                    usage.get(field).and_then(Value::as_u64).is_some(),
                    "usage should include numeric {field}: {usage:?}"
                );
            }
            assert!(
                usage.get("output_tokens_details").is_none(),
                "mistral.rs compatibility fixture should omit output_tokens_details: {usage:?}"
            );
        },
    )
    .await;
}
