//! vLLM OpenAI-compatible Responses API regression tests.

use rig::completion::CompletionRequest;
use rig::http_runtime::HttpRuntime;
use rig::providers::openai;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::ProviderCassette;
use futures::FutureExt;

/// Connection details for the vLLM-flavoured OpenAI cassette proxy.
struct VllmCassette {
    api_key: String,
    base_url: String,
}

impl VllmCassette {
    fn config(&self, model: impl Into<String>) -> openai::responses_api::functions::Config {
        openai::responses_api::functions::Config::new(model)
            .with_api_key(self.api_key.clone())
            .with_base_url(self.base_url.clone())
    }

    fn http(&self) -> HttpRuntime {
        HttpRuntime::new()
    }
}

async fn with_openai_vllm_cassette<F, Fut>(scenario: &'static str, test_body: F)
where
    F: FnOnce(VllmCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    let base_url =
        std::env::var("VLLM_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8000/v1".to_string());
    let cassette = ProviderCassette::start("openai", scenario, &base_url).await;
    let client = VllmCassette {
        api_key: "dummy-vllm-key".to_string(),
        base_url: cassette.base_url(),
    };

    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

#[tokio::test]
async fn responses_api_accepts_null_metadata() {
    const SCENARIO: &str = "vllm/responses_api_accepts_null_metadata";
    with_openai_vllm_cassette("vllm/responses_api_accepts_null_metadata", |client| async move {
        let cfg = client.config("Qwen/Qwen3-0.6B");
        let request = CompletionRequest {
            max_tokens: Some(8),
            ..CompletionRequest::from_prompt("Reply with a short acknowledgement.")
        };

        let response =
            openai::responses_api::functions::complete(&cfg, &client.http(), request)
                .await
                .expect("vLLM Responses API completion with null metadata should deserialize");

        assert!(
            response.choice.iter().next().is_some(),
            "response should contain assistant content"
        );

        // The `metadata: null` normalization is wire-level behavior; check it
        // by deserializing the recorded body into the provider response type
        // (replay only: the cassette file is written after the test body in
        // record mode).
        if crate::cassettes::CassetteMode::current() == crate::cassettes::CassetteMode::Replay {
            let bodies = crate::cassettes::recorded_response_bodies("openai", SCENARIO);
            assert_eq!(bodies.len(), 1, "scenario should record a single interaction");
            let raw: openai::responses_api::CompletionResponse = serde_json::from_str(&bodies[0])
                .expect("recorded vLLM body should deserialize as a Responses API response");
            assert!(
                raw.additional_parameters.metadata.is_empty(),
                "vLLM returns metadata: null; Rig should preserve the public map API as an empty map"
            );
        }
    })
    .await;
}
