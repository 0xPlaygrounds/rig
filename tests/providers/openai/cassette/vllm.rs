//! vLLM OpenAI-compatible Responses API regression tests.

use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::openai::OpenAI;
use rig::providers::openai::responses_api::CompletionResponse as ProviderResponse;
use serde::Deserialize;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::ProviderCassette;
use futures::FutureExt;

async fn with_openai_vllm_cassette<F, Fut>(scenario: &'static str, test_body: F)
where
    F: FnOnce(Bound<OpenAI>) -> Fut,
    Fut: Future<Output = ()>,
{
    let base_url =
        std::env::var("VLLM_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8000/v1".to_string());
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "openai",
        scenario,
        &base_url,
    )
    .await;
    let client = OpenAI::new("dummy-vllm-key")
        .with_base_url(cassette.base_url())
        .bound()
        .expect("vLLM OpenAI-compatible client should build");

    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

#[tokio::test]
async fn responses_api_accepts_null_metadata() {
    with_openai_vllm_cassette(
        "vllm/responses_api_accepts_null_metadata",
        |client| async move {
            let model = client.completion("Qwen/Qwen3-0.6B");
            let request = model
                .completion_request("Reply with a short acknowledgement.")
                .max_tokens(8)
                .build();

            // `metadata` is a provider-native wire field, so it is read off the
            // Responses API's own response type, deserialized from
            // `CompletionResponse::raw`. One request therefore yields both
            // views, which is what the single recorded interaction allows.
            let response = model
                .completion(request)
                .await
                .expect("vLLM Responses API completion with null metadata should deserialize");
            let reply = ProviderResponse::deserialize(&response.raw)
                .expect("`raw` is the serialized responses_api::CompletionResponse");

            assert!(
                reply.additional_parameters.metadata.is_empty(),
                "vLLM returns metadata: null; Rig should preserve the public map API as an empty map"
            );
            assert!(
                !response.choice.is_empty(),
                "response should contain assistant content"
            );
        },
    )
    .await;
}
