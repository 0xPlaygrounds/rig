//! vLLM OpenAI-compatible Responses API regression tests.

use rig::completion::CompletionModel;
use rig::completion::NormalizeCompletionResponse;
use rig::prelude::*;
use rig::providers::openai;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::ProviderCassette;
use futures::FutureExt;

async fn with_openai_vllm_cassette<F, Fut>(scenario: &'static str, test_body: F)
where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let base_url =
        std::env::var("VLLM_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8000/v1".to_string());
    let cassette = ProviderCassette::start("openai", scenario, &base_url).await;
    let client = openai::Client::builder()
        .api_key("dummy-vllm-key")
        .base_url(cassette.base_url())
        .build()
        .expect("vLLM OpenAI-compatible client should build");

    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

#[tokio::test]
async fn responses_api_accepts_null_metadata() {
    with_openai_vllm_cassette(
        "vllm/responses_api_accepts_null_metadata",
        |client| async move {
            let model = client.completion_model("Qwen/Qwen3-0.6B");
            let request = model
                .completion_request("Reply with a short acknowledgement.")
                .max_tokens(8)
                .build();

            // `metadata` is a provider-native wire field, so it is read off the
            // Responses API's own response type. The cassette records a single
            // interaction, so the normalized response is derived from that same
            // raw response instead of issuing a second request.
            let raw = model
                .raw_completion(request)
                .await
                .expect("vLLM Responses API completion with null metadata should deserialize");

            assert!(
                raw.additional_parameters.metadata.is_empty(),
                "vLLM returns metadata: null; Rig should preserve the public map API as an empty map"
            );

            let response: rig::completion::CompletionResponse = raw.normalize("openai")
                .expect("vLLM Responses API completion should normalize");
            assert!(
                response.choice.iter().next().is_some(),
                "response should contain assistant content"
            );
        },
    )
    .await;
}
