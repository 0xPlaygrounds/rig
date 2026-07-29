//! Cassette coverage for mistral.rs usage without OpenAI `output_tokens_details`.

use rig::completion::{CompletionModel, CompletionRequest};
use rig::prelude::*;
use serde_json::Value;

use super::super::support::{SYSTEM_PROMPT, model_name, with_mistralrs_completions_cassette};

#[tokio::test]
async fn chat_completion_usage_without_output_tokens_details_deserializes() {
    with_mistralrs_completions_cassette(
        "usage/chat_completion_usage_without_output_tokens_details_deserializes",
        |client| async move {
            let model = client.completion_model(model_name());
            let request = CompletionRequest {
                max_tokens: Some(64),
                ..CompletionRequest::with_history(
                    Some(SYSTEM_PROMPT),
                    Vec::new(),
                    "/no_think Explain usage accounting in one sentence.",
                )
            };
            let _response = model
                .completion(request)
                .await
                .expect("usage check completion should succeed");
            // The normalized response no longer exposes the raw payload, so the
            // wire-shape assertions read the recorded cassette body directly.
            let bodies = crate::cassettes::recorded_response_bodies(
                "mistralrs",
                "usage/chat_completion_usage_without_output_tokens_details_deserializes",
            );
            let raw: Value = serde_json::from_str(
                bodies
                    .last()
                    .expect("cassette should contain a recorded response body"),
            )
            .expect("recorded mistral.rs chat completion body should be JSON");
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
