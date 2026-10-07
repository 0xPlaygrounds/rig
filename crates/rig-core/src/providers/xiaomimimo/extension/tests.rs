//! Xiaomi MiMo's extras. No MiMo recording exists, so the test decodes a
//! reply built here.

use serde_json::{Value, json};

use super::*;
use crate::completion::{CompletionRequest, CompletionResponse};
use crate::providers::anthropic::extension::AnthropicExt;
use crate::providers::anthropic::wire::{
    AnthropicConfig, Messages, XIAOMIMIMO as MESSAGES_XIAOMIMIMO,
};
use crate::providers::openai::wire::{OpenAIConfig, XIAOMIMIMO};
use crate::test_utils::provider_extensions::{chat_reply, reply_of};
use crate::wire::{Mode, WireFrame};

#[tokio::test]
async fn extras_from_a_built_reply() {
    let annotation = json!({
        "type": "url_citation",
        "url_citation": {"url": "https://www.rust-lang.org", "title": "Rust"}
    });
    let reply = reply_of(
        OpenAIConfig::with_key(&XIAOMIMIMO, "key").chat(crate::providers::xiaomimimo::MIMO_V2_5),
        chat_reply(json!({"choices": [{"message": {"annotations": [annotation.clone()]}}]})),
    )
    .await;
    let extras = reply
        .extras::<XiaomiMimoExt>()
        .unwrap_or_else(|| panic!("a MiMo reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(extras.annotations, Some(vec![annotation]));
}

fn messages_wire() -> Messages {
    AnthropicConfig::with_key(&MESSAGES_XIAOMIMIMO, "sk-test")
        .completion(crate::providers::xiaomimimo::MIMO_V2_FLASH)
}

/// The response `reply` folds into on Xiaomi MiMo's Messages wire.
fn folded(reply: &Value) -> CompletionResponse {
    crate::test_utils::decode_reply(
        &messages_wire(),
        &CompletionRequest::new("hi"),
        Mode::Unary,
        [WireFrame::Text(reply.to_string())],
        reply.clone(),
    )
    .expect("the reply folds")
}

/// Built here, not recorded: no Xiaomi MiMo Messages reply is recorded.
#[test]
fn extras_read_the_messages_stop_fields() {
    let reply = json!({
        "type": "message", "id": "msg_1", "model": crate::providers::xiaomimimo::MIMO_V2_FLASH, "role": "assistant",
        "content": [{"type": "text", "text": "alpha"}],
        "stop_reason": "stop_sequence", "stop_sequence": "alpha",
        "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    let response = folded(&reply);
    let extras = response
        .extras::<XiaomiMimoExt>()
        .expect("a Xiaomi MiMo reply")
        .expect("the extras read");
    assert_eq!(extras.stop_reason.as_deref(), Some("stop_sequence"));
    assert_eq!(extras.stop_sequence.as_deref(), Some("alpha"));
    assert!(response.extras::<AnthropicExt>().is_none());
}
