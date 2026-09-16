//! Cassette coverage for mistral.rs usage without OpenAI `output_tokens_details`.

use rig::completion::CompletionModel;
use serde_json::Value;

use super::super::support::{SYSTEM_PROMPT, model_name, with_mistralrs_completions_cassette};

#[tokio::test]
async fn chat_completion_usage_without_output_tokens_details_deserializes() {
    with_mistralrs_completions_cassette(
        "usage/chat_completion_usage_without_output_tokens_details_deserializes",
        |client| async move {
            let model = client.chat(model_name());
            let request = model
                .completion_request("/no_think Explain usage accounting in one sentence.")
                .preamble(SYSTEM_PROMPT.to_string())
                .max_tokens(64)
                .build();
            // A single cassette interaction: the usage-shape assertions read
            // mistral.rs's own reply document off the response the completion
            // path folded, so the shape and the fold are the same reply.
            let response = model
                .completion(request)
                .await
                .expect("usage check completion should succeed");
            let usage = response
                .raw
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
