//! Cassette coverage for mistral.rs `/v1/chat/completions` responses.

use rig::completion::CompletionRequest;
use rig::http_runtime::HttpRuntime;
use rig::providers::openai;

use rig::prelude::*;
use serde_json::Value;

use super::super::support::{SYSTEM_PROMPT, model_name, with_mistralrs_cassette};

#[tokio::test]
async fn raw_chat_completion_surfaces_reasoning_or_text() {
    with_mistralrs_cassette(
        "chat_completions/raw_chat_completion_surfaces_reasoning_or_text",
        |env| async move {
            let cfg = env.chat_config(model_name());
            let rt = HttpRuntime::new();
            let request = CompletionRequest {
                max_tokens: Some(256),
                ..CompletionRequest::with_history(
                    Some(SYSTEM_PROMPT),
                    Vec::new(),
                    "Think briefly, then answer in one sentence why token usage should be reported.",
                )
            };
            let _response = openai::functions::complete(&cfg, &rt, request)
                .await
                .expect("raw chat completion should succeed");
            // The normalized response no longer exposes the raw payload, so the
            // wire-shape assertions read the recorded cassette body directly.
            let bodies = crate::cassettes::recorded_response_bodies(
                "mistralrs",
                "chat_completions/raw_chat_completion_surfaces_reasoning_or_text",
            );
            let raw: Value = serde_json::from_str(
                bodies
                    .last()
                    .expect("cassette should contain a recorded response body"),
            )
            .expect("recorded mistral.rs chat completion body should be JSON");
            let message = &raw["choices"][0]["message"];
            let text = message
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let reasoning = message
                .get("reasoning_content")
                .and_then(Value::as_str)
                .or_else(|| message.get("reasoning").and_then(Value::as_str))
                .unwrap_or_default();

            assert!(
                !text.is_empty() || !reasoning.is_empty(),
                "mistral.rs chat response should contain content or reasoning"
            );
            let usage = raw
                .get("usage")
                .expect("mistral.rs chat response should include usage");
            assert!(
                usage.get("prompt_tokens").and_then(Value::as_u64).is_some(),
                "usage should include prompt_tokens: {usage:?}"
            );
            assert!(
                usage.get("total_tokens").and_then(Value::as_u64).is_some(),
                "usage should include total_tokens: {usage:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn chat_completions_agent_prompt_completes() {
    with_mistralrs_cassette(
        "chat_completions/chat_completions_agent_prompt_completes",
        |env| async move {
            let agent = AgentBuilder::new(env.chat_provider(model_name()))
                .preamble(SYSTEM_PROMPT)
                .max_tokens(128)
                .build();

            let response = agent
                .prompt(
                    "/no_think Explain why a local OpenAI-compatible server should return token usage.",
                )
                .await
                .expect("Rig OpenAI Chat Completions API prompt should succeed");

            assert!(
                !response.trim().is_empty(),
                "no_think chat-completions prompt should return visible text"
            );
        },
    )
    .await;
}
