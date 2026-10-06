//! Z.AI's options as the bodies they encode to, and its extras. The encoder
//! tests are unit tests because no recording sends typed provider options.
//! No Z.AI recording exists, so the extras test decodes a reply built here.

use serde_json::{Value, json};

use super::*;
use crate::completion::{GenerationOptions, Reasoning};
use crate::providers::openai::wire::{Chat, OpenAIConfig, ZAI};
use crate::test_utils::provider_extensions::{
    assert_no_reserved_leaf, body_with, chat_reply, encoded_body, reply_of, request_with,
};
use crate::wire::Mode;

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
