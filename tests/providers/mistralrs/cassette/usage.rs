//! Cassette coverage for mistral.rs usage without OpenAI `output_tokens_details`.

use rig::completion::CompletionModel;
use rig::completion::NormalizeCompletionResponse;
use rig::prelude::*;
use serde_json::Value;

use super::super::support::{SYSTEM_PROMPT, model_name, with_mistralrs_completions_cassette};

#[tokio::test]
async fn chat_completion_usage_without_output_tokens_details_deserializes() {
    with_mistralrs_completions_cassette(
        "usage/chat_completion_usage_without_output_tokens_details_deserializes",
        |client| async move {
            let model = client.completion_model(model_name());
            let request = model
                .completion_request("/no_think Explain usage accounting in one sentence.")
                .preamble(SYSTEM_PROMPT.to_string())
                .max_tokens(64)
                .build();
            // A single cassette interaction: keep mistral.rs's own wire response
            // for the usage-shape assertions, and still check that the
            // normalization the completion path applies accepts it.
            let wire_response = model
                .raw_completion(request)
                .await
                .expect("usage check completion should succeed");
            let raw = serde_json::to_value(&wire_response)
                .expect("raw chat completion response should serialize");
            let _normalized: rig::completion::CompletionResponse = wire_response
                .normalize("openai")
                .expect("usage check completion should normalize");
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
