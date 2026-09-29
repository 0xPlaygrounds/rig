//! Dual-key Chat Completions traffic recorded from a local scripted server, not a live provider.

use std::future::Future;
use std::panic::AssertUnwindSafe;

use axum::{Json, Router, routing::post};
use futures::FutureExt;
use rig::completion::CompletionRequest;
use rig::providers::openai::OpenAIConfig;
use rig_test_support::cassette_models::OpenAiModels;
use serde_json::{Value, json};

use crate::cassettes::ProviderCassette;

#[tokio::test]
async fn local_dual_reasoning_keys_preserve_tool_call() {
    with_local_reasoning_keys_cassette(
        "openai_compatible/dual_reasoning_keys",
        |client| async move {
            let response = client
                .chat("local-dual-key-model")
                .call(CompletionRequest::new("Look up Tokyo weather."))
                .await
                .expect("both reasoning keys must decode");
            assert_eq!(response.reasoning(), "preferred reasoning");
            let calls = response.tool_calls().collect::<Vec<_>>();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].id.wire(), "call_local_1");
            assert_eq!(calls[0].function.name, "lookup");
            assert_eq!(calls[0].function.arguments, json!({"city": "Tokyo"}));
        },
    )
    .await;
    let message =
        crate::cassettes::recorded_json_response("openai", "openai_compatible/dual_reasoning_keys")
            ["choices"][0]["message"]
            .clone();
    assert_eq!(message["reasoning"], "fallback reasoning");
    assert_eq!(message["reasoning_content"], "preferred reasoning");
    assert_eq!(
        message["tool_calls"][0]["function"]["arguments"],
        "{\"city\":\"Tokyo\"}"
    );
}

async fn with_local_reasoning_keys_cassette<F, Fut>(scenario: &'static str, test_body: F)
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("local server bind");
    let addr = listener.local_addr().expect("local address");
    let task = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new().route("/v1/chat/completions", post(local_reply)),
        )
        .await
        .expect("local server");
    });
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "openai",
        scenario,
        &format!("http://{addr}/v1"),
    )
    .await;
    let config = OpenAIConfig::new("dummy-local-key").with_base_url(cassette.base_url());
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(
        config,
        rig::rig_reqwest::shared(),
    )))
    .catch_unwind()
    .await;
    task.abort();
    cassette.finish_after_test(result).await;
}

async fn local_reply() -> Json<Value> {
    Json(json!({
        "id": "chatcmpl_local_1", "object": "chat.completion", "created": 0, "model": "local-dual-key-model",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": null,
            "reasoning": "fallback reasoning", "reasoning_content": "preferred reasoning",
            "tool_calls": [{"id": "call_local_1", "type": "function", "function": {
                "name": "lookup", "arguments": "{\"city\":\"Tokyo\"}"
            }}]
        }}], "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    }))
}
