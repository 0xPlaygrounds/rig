//! Z.AI's options as the bodies they encode to, and its extras. The encoder
//! tests are unit tests because no recording sends typed provider options.
//! No Z.AI recording exists, so the extras test decodes a reply built here.

use serde_json::{Value, json};

use super::*;
use crate::completion::{CompletionRequest, CompletionResponse, GenerationOptions, Reasoning};
use crate::providers::anthropic::extension::Anthropic;
use crate::providers::anthropic::wire::{AnthropicConfig, Messages, ZAI as MESSAGES_ZAI};
use crate::providers::openai::wire::{Chat, OpenAIConfig, ZAI};
use crate::test_utils::provider_extensions::{
    assert_no_reserved_leaf, body_with, chat_reply, encoded_body, reply_of, request_with,
};
use crate::wire::{Mode, WireFrame};

const MODEL: &str = crate::providers::zai::GLM_4_6;

fn chat_wire() -> Chat {
    OpenAIConfig::with_key(&ZAI, "key").chat(MODEL)
}

fn body(chat: ZaiChat) -> Value {
    body_with::<Zai, _>(&chat_wire(), &ZaiOptions::new().chat(chat))
}

#[test]
fn do_sample_lands_at_top_level() {
    assert_eq!(body(ZaiChat::new().do_sample(false))["do_sample"], false);
}

#[test]
fn request_id_lands_at_top_level() {
    assert_eq!(
        body(ZaiChat::new().request_id("req-1"))["request_id"],
        "req-1"
    );
}

#[test]
fn user_id_lands_at_top_level() {
    assert_eq!(body(ZaiChat::new().user_id("user-1"))["user_id"], "user-1");
}

#[test]
fn clear_thinking_joins_the_mapped_thinking() {
    let request =
        request_with::<Zai>(&ZaiOptions::new().chat(ZaiChat::new().clear_thinking(false)))
            .options(GenerationOptions::default().reasoning(Reasoning::Off));
    let body =
        encoded_body(&chat_wire(), request, Mode::Unary).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        body["thinking"],
        json!({"type": "disabled", "clear_thinking": false})
    );
}

#[test]
fn no_option_writes_a_leaf_the_request_or_a_mapped_option_owns() {
    let chat = ZaiChat::new()
        .do_sample(true)
        .request_id("r")
        .user_id("u")
        .clear_thinking(true);
    assert_no_reserved_leaf::<Zai, _>(
        &[
            chat_wire(),
            OpenAIConfig::with_key(&ZAI, "key").chat("glm-5"),
        ],
        &ZaiOptions::new().chat(chat),
    );
}

#[tokio::test]
async fn extras_from_a_built_reply() {
    let reply = reply_of(chat_wire(), chat_reply(json!({"request_id": "req-9"}))).await;
    let extras = reply
        .extras::<Zai>()
        .unwrap_or_else(|| panic!("a Z.AI reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(extras.request_id.as_deref(), Some("req-9"));
}

fn messages_wire() -> Messages {
    AnthropicConfig::with_key(&MESSAGES_ZAI, "sk-test").completion(crate::providers::zai::GLM_4_6)
}

/// The response `reply` folds into on Z.AI's Messages wire.
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

/// Built here, not recorded: no Z.AI Messages reply is recorded.
#[test]
fn extras_read_the_messages_stop_fields() {
    let reply = json!({
        "type": "message", "id": "msg_1", "model": crate::providers::zai::GLM_4_6, "role": "assistant",
        "content": [{"type": "text", "text": "alpha"}],
        "stop_reason": "stop_sequence", "stop_sequence": "alpha",
        "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    let response = folded(&reply);
    let extras = response
        .extras::<Zai>()
        .expect("a Z.AI reply")
        .expect("the extras read");
    assert_eq!(extras.stop_reason.as_deref(), Some("stop_sequence"));
    assert_eq!(extras.stop_sequence.as_deref(), Some("alpha"));
    assert!(response.extras::<Anthropic>().is_none());
}
