//! Cassette coverage for mistral.rs `/v1/chat/completions` responses.

use rig::client::CompletionClient;
use rig::completion::{CompletionModel, Prompt};
use serde_json::Value;

use super::super::support::{SYSTEM_PROMPT, model_name, with_mistralrs_completions_cassette};

#[tokio::test]
async fn raw_chat_completion_surfaces_reasoning_or_text() {
    with_mistralrs_completions_cassette(
        "chat_completions/raw_chat_completion_surfaces_reasoning_or_text",
        |client| async move {
            let response = client
                .completion_model(model_name())
                .completion_request(
                    "Think briefly, then answer in one sentence why token usage should be reported.",
                )
                .preamble(SYSTEM_PROMPT.to_string())
                .max_tokens(256)
                .send()
                .await
                .expect("raw chat completion should succeed");
            let raw = serde_json::to_value(&response.raw_response)
                .expect("raw chat completion response should serialize");
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
    with_mistralrs_completions_cassette(
        "chat_completions/chat_completions_agent_prompt_completes",
        |client| async move {
            let agent = client
                .agent(model_name())
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
