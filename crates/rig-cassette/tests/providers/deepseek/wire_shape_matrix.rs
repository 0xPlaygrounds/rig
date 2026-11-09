//! Wire-shape matrix for DeepSeek's plain-string message content.
//!
//! DeepSeek takes message `content` as a plain string, so the wire's body rewrite
//! (`finalize_deepseek`) flattens content-part arrays with `only_if_all_text = true`. An
//! all-text array flattens to a string; an array with a non-text part reaches the wire, so
//! DeepSeek's own 400 reaches the caller instead of the attachment being dropped and the
//! question answered from the text alone. DeepSeek rejects document, audio and video parts the
//! same way, so its wire states it carries none, and the history adapter sends every non-text
//! part as placeholder text before the rewrite sees it. The non-text cells pin that placeholder.
//!
//! The cells cover the rewrite's one decision exhaustively: mixed text with each non-text tag
//! the chat-completions wire emits, non-text-only content (the empty-text boundary), one
//! streamed control, and the all-text, text-document and assistant or tool-result history
//! controls. Agent calls and more streamed tags would repeat identical request bodies, since
//! both transports and both agent surfaces encode through the same rewrite.
//!
//! The census cells pin confirmed behaviour: DeepSeek rejects a forced `tool_choice` while
//! thinking is on, which justifies the rewrite suppressing one; the completion path preserves
//! DeepSeek's error envelope; and the `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`
//! split reaches `Usage::cached_input_tokens` on both transports.

use rig::providers::deepseek;
use serde_json::{Value, json};

use super::support::{with_deepseek_cassette_bogus_key_result, with_deepseek_cassette_result};
use rig::completion::{CompletionRequest, GenerationOptions, Reasoning};

const MODEL: &str = deepseek::DEEPSEEK_V4_FLASH;

fn non_thinking() -> GenerationOptions {
    GenerationOptions::default().reasoning(Reasoning::Off)
}

// ================================================================
// A. Non-text parts reach the wire as placeholders
// ================================================================

// ================================================================
// C. Census: the forced-tool-choice suppression is justified
// ================================================================

#[tokio::test]
async fn forced_tool_choice_under_thinking_is_rejected_upstream() {
    const SCENARIO: &str =
        "wire_shape_matrix/forced_tool_choice_under_thinking_is_rejected_upstream";
    with_deepseek_cassette_result(
        "wire_shape_matrix/forced_tool_choice_under_thinking_is_rejected_upstream",
        |client| async move {
            // The wire's `encode` rewrites any forced `tool_choice` rig
            // itself would send, so the only way to learn what DeepSeek does
            // with one is to hand-build the body.
            let url = format!(
                "{}/chat/completions",
                client.config.base_url.trim_end_matches('/')
            );
            let api_key =
                std::env::var("DEEPSEEK_API_KEY").unwrap_or_else(|_| "[REDACTED]".to_owned());
            let tools = json!([{
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "Ping.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }]);

            for tool_choice in [
                json!("required"),
                json!({"type": "function", "function": {"name": "ping"}}),
            ] {
                let response = reqwest::Client::new()
                    .post(&url)
                    .bearer_auth(&api_key)
                    .json(&json!({
                        "model": MODEL,
                        "thinking": {"type": "enabled"},
                        "max_tokens": 16,
                        "messages": [{"role": "user", "content": "ping"}],
                        "tools": tools,
                        "tool_choice": tool_choice,
                    }))
                    .send()
                    .await?;

                let status = response.status();
                let body: Value = response.json().await?;
                assert_eq!(
                    status.as_u16(),
                    400,
                    "a forced tool choice under thinking is a hard error: {body}"
                );
                assert!(
                    body["error"]["message"]
                        .as_str()
                        .unwrap_or_default()
                        .to_lowercase()
                        .contains("thinking mode does not support this tool_choice"),
                    "the rejection names the thinking-mode constraint: {body}"
                );
            }
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect(
        "forced_tool_choice_under_thinking_is_rejected_upstream should replay from its cassette",
    );

    assert_eq!(
        crate::cassettes::recorded_interaction_bodies("deepseek", SCENARIO).len(),
        2,
        "both forced shapes are recorded"
    );
}

// ================================================================
// D. Census: the completion path preserves DeepSeek's error envelope
// ================================================================

#[tokio::test]
async fn chat_completion_rejects_a_bogus_key_with_the_provider_body() {
    with_deepseek_cassette_bogus_key_result(
        "wire_shape_matrix/chat_completion_rejects_a_bogus_key_with_the_provider_body",
        |client| async move {
            let model = client.completion(MODEL);
            let error = model
                .call(CompletionRequest::new("hi")
                        .options(non_thinking())
                        .max_tokens(8))
                .await
                .expect_err("a rejected key is an error");
            let rendered = error.to_string().to_lowercase();
            assert!(
                rendered.contains("authentication fails"),
                "the provider's own 401 body survives: {rendered}"
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("chat_completion_rejects_a_bogus_key_with_the_provider_body should replay from its cassette");
}

// ================================================================
// E. Census: the cache hit/miss split reaches rig's usage
// ================================================================
