//! Anthropic redacted-thinking regression tests.
//!
//! Uses Anthropic's documented magic string to deterministically trigger
//! `redacted_thinking` blocks, then locks down that Rig surfaces them as
//! redacted reasoning and replays them back across turns without the API
//! rejecting the history.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use futures::StreamExt;
use rig::completion::Message;
use rig::message::AssistantContent;
use rig::providers::anthropic;
use rig::streaming::Item;
use rig::streaming::StreamEvent;

use super::super::support::with_anthropic_cassette;
use rig::completion::CompletionRequest;

/// Anthropic's documented test string that forces the model to emit
/// `redacted_thinking` blocks when extended thinking is enabled.
const REDACTED_THINKING_MAGIC_STRING: &str = "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB";

fn redacted_thinking_prompt() -> String {
    format!("{REDACTED_THINKING_MAGIC_STRING} Reply with the single word OK.")
}

fn thinking_params() -> serde_json::Value {
    serde_json::json!({
        "thinking": { "type": "enabled", "budget_tokens": 1024 }
    })
}

fn has_redacted_reasoning(content: &AssistantContent) -> bool {
    matches!(
        content,
        AssistantContent::Reasoning(reasoning) if reasoning.redacted
    )
}

#[tokio::test]
async fn redacted_thinking_roundtrip_nonstreaming() {
    with_anthropic_cassette(
        "messages_thinking/redacted_thinking_roundtrip_nonstreaming",
        |client| async move {
            let model = client.completion(anthropic::completion::CLAUDE_SONNET_4_6);

            let first_request = CompletionRequest::new(redacted_thinking_prompt())
                .max_tokens(4096)
                .additional_params(thinking_params());
            let first_response = model
                .call(first_request)
                .await
                .expect("redacted-thinking completion should succeed");

            assert!(
                first_response.choice.iter().any(has_redacted_reasoning),
                "the magic string must surface a redacted reasoning block, got {:?}",
                first_response.choice
            );

            // Replay the redacted thinking block back in a follow-up turn; the
            // API must accept the opaque data verbatim.
            let second_request =
                CompletionRequest::new("Thanks. Now reply with the single word DONE.")
                    .max_tokens(4096)
                    .additional_params(thinking_params())
                    .message(Message::user(redacted_thinking_prompt()))
                    .message(Message::Assistant(rig::message::AssistantMessage {
                        content: first_response.choice.clone(),
                        ..first_response.head()
                    }));

            let second_response = model
                .call(second_request)
                .await
                .expect("history containing redacted_thinking should be accepted");

            let text: String = second_response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect();
            assert!(
                !text.trim().is_empty(),
                "follow-up turn should produce text after replaying redacted thinking"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn redacted_thinking_streaming() {
    with_anthropic_cassette(
        "messages_thinking/redacted_thinking_streaming",
        |client| async move {
            let model = client.completion(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = CompletionRequest::new(redacted_thinking_prompt())
                .max_tokens(4096)
                .additional_params(thinking_params());

            let mut stream = model
                .stream(request)
                .expect("redacted-thinking streaming request should start");

            let mut saw_redacted_reasoning = false;
            let mut streamed_text = String::new();

            while let Some(item) = stream.next().await {
                match item.expect("stream item should be ok") {
                    Item::Event(StreamEvent::End {
                        content: AssistantContent::Reasoning(reasoning),
                        ..
                    }) if reasoning.redacted => {
                        saw_redacted_reasoning = true;
                    }
                    Item::Event(StreamEvent::Text { text, .. }) => streamed_text.push_str(&text),
                    _ => {}
                }
            }

            assert!(
                saw_redacted_reasoning,
                "the stream should surface a redacted reasoning block"
            );
            assert!(
                !streamed_text.trim().is_empty(),
                "the stream should still produce the visible answer text"
            );
        },
    )
    .await;
}
