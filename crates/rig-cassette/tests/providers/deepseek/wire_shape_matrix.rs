//! Edge matrix for rig#2354's third fix, plus the wire census the hunt turned
//! up.
//!
//! **The fix.** DeepSeek takes message `content` as a plain string, so the
//! wire's DeepSeek body rewrite (`finalize_deepseek`) flattens content-part
//! arrays. It passed
//! `only_if_all_text = false`, which *drops* every non-text part — so an
//! attached image, audio clip or PDF was silently deleted and DeepSeek
//! answered the question from the remaining text alone, with nothing anywhere
//! saying the attachment was gone. Perplexity, the tree's other plain-text-only
//! provider, passes `true` so the part survives to the wire and the API's own
//! rejection reaches the caller. DeepSeek now does the same: verified live,
//! `{"type":"image_url",...}` comes back `400 Failed to deserialize the JSON
//! body into the target type: messages[0]: unknown variant `image_url`,
//! expected `text``. All-text arrays still flatten to a plain string, byte for
//! byte as before — which is why no existing fixture moved.
//!
//! DeepSeek answers a document, audio or video part with the same 400, so
//! its wire states it carries none of them, and the history adapter sends
//! every non-text part as its placeholder text before this rewrite sees it.
//! The non-text cells pin that placeholder.
//!
//! The fix-relevant live matrix is the complete input partition seen by the
//! changed `flatten_text_content_parts(..., only_if_all_text)` call:
//!
//! | partition | recorded cells | why this exhausts the branch |
//! |---|---:|---|
//! | mixed text + non-text user content | 5 | every emitted non-text chat-completions tag: base64 image, URL image, PDF/file, audio and video |
//! | non-text-only user content | 1 | pins the empty-text boundary that previously collapsed the whole message to `""` |
//! | streaming transport | 1 | image rejection control; blocking and streaming encode the same request through the same body rewrite before transport selection |
//! | unchanged flattening controls | 3 | all-text user parts, normalized text documents, and assistant/tool-result history |
//!
//! That is 10 fix-relevant recorded cells. The input space is smaller than 24
//! because the changed helper has only one decision — every part has text, or
//! at least one does not — and the five non-text variants above are the full
//! wire enum, not samples from an open-ended set. Agent calls add no request
//! mapping branch: both agent surfaces delegate to the same completion model.
//! Recording agent duplicates or all five tags again on streaming would
//! therefore repeat identical encoded request bodies rather than exercise
//! another path.
//!
//! **The census** (confirmed non-bugs, recorded so they stay confirmed):
//! DeepSeek really does reject a forced `tool_choice` while thinking is on, so
//! the rewrite's suppression of one is justified; the completion path
//! preserves DeepSeek's error envelope; and the `prompt_cache_hit_tokens` /
//! `prompt_cache_miss_tokens` split reaches `Usage::cached_input_tokens` on
//! both transports.

use rig::providers::deepseek;
use serde_json::{Value, json};

use super::support::{
    recorded_interactions, with_deepseek_cassette_bogus_key_result,
    with_deepseek_wire_shape_cassette_result,
};
use rig::completion::CompletionRequest;

const MODEL: &str = deepseek::DEEPSEEK_V4_FLASH;

fn non_thinking_params() -> Value {
    json!({ "thinking": { "type": "disabled" } })
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
    with_deepseek_wire_shape_cassette_result(
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
        recorded_interactions(SCENARIO).len(),
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
                        .additional_params(non_thinking_params())
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
