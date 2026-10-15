use super::*;
use crate::completion::CompletionRequest;
use crate::driver::Model;
use crate::providers::anthropic::{completion::CLAUDE_SONNET_4_6, wire::AnthropicConfig};
use crate::test_utils::{MockHttpResponse, SequencedHttpClient};

/// A Messages reply with text and a tool call, as `POST /v1/messages`
/// answers it whole.
fn messages_body() -> String {
    serde_json::json!({
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "model": CLAUDE_SONNET_4_6,
        "content": [
            {"type": "text", "text": "Looking."},
            {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "a.rs"}}
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 12, "output_tokens": 7}
    })
    .to_string()
}

/// A reply body that arrives some other way, such as a batch result, decodes
/// to the response a unary call that received the same body gives.
#[tokio::test]
async fn a_whole_body_decodes_as_a_unary_call_gives_it() {
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let request = CompletionRequest::new("read a.rs");
    let http = SequencedHttpClient::new([MockHttpResponse::success(messages_body())]);
    let called = Model::new(wire.clone(), http)
        .call(request.clone())
        .await
        .expect("the call succeeds");

    let decoded = decode_reply(
        &wire,
        &request,
        Mode::Unary,
        [WireFrame::Bytes(messages_body().into_bytes())],
    )
    .expect("the body decodes");

    assert_eq!(decoded, called);
}

/// Frames that stop before the provider ended the reply fail as a live
/// call's would.
#[test]
fn frames_that_stop_early_are_truncated() {
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let start = serde_json::json!({
        "type": "message_start",
        "message": {"id": "msg_1", "type": "message", "role": "assistant", "content": [],
            "model": CLAUDE_SONNET_4_6, "usage": {"input_tokens": 1, "output_tokens": 0}}
    });
    let decoded = decode_reply(
        &wire,
        &CompletionRequest::new("hi"),
        Mode::Streaming,
        [WireFrame::Text(start.to_string())],
    );
    assert!(
        matches!(decoded, Err(ProviderError::Truncated)),
        "{decoded:?}"
    );
}

/// What a model keeps from preparing a request is shared by its clones and
/// never makes two models differ.
#[test]
fn what_a_model_kept_never_makes_two_models_differ() {
    let kept = Prepared::default();
    let clone = kept.clone();
    assert!(Arc::ptr_eq(&kept.0, &clone.0), "a model's clones share it");
    assert_eq!(kept, Prepared::default());
}
