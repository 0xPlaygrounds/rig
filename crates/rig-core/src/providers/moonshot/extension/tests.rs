//! Moonshot's options as the bodies they encode to, and its extras. The
//! encoder tests are unit tests because no recording sends typed provider
//! options. No Moonshot recording exists, so the extras test decodes a
//! reply built here.

use serde_json::{Value, json};

use super::*;
use crate::completion::{CompletionRequest, CompletionResponse, GenerationOptions, Reasoning};
use crate::providers::anthropic::extension::AnthropicExt;
use crate::providers::anthropic::wire::{AnthropicConfig, MOONSHOT as MESSAGES_MOONSHOT, Messages};
use crate::providers::moonshot::{KIMI_K2_6, KIMI_K3};
use crate::providers::openai::wire::{Chat, MOONSHOT, OpenAIConfig};
use crate::test_utils::provider_extensions::{
    assert_no_reserved_leaf, body_with, chat_reply, encoded_body, reply_of, request_with,
};
use crate::wire::{Mode, WireFrame};

fn chat_wire(model: &str) -> Chat {
    OpenAIConfig::with_key(&MOONSHOT, "key").chat(model)
}

fn options(chat: MoonshotChat) -> MoonshotOptions {
    MoonshotOptions::new().chat(chat)
}

#[test]
fn thinking_keep_joins_the_mapped_thinking() {
    let request = request_with::<MoonshotExt>(&options(
        MoonshotChat::new().thinking_keep(ThinkingKeep::All),
    ))
    .options(GenerationOptions::default().reasoning(Reasoning::Off));
    let body = encoded_body(&chat_wire(KIMI_K2_6), request, Mode::Unary)
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(body["thinking"], json!({"type": "disabled", "keep": "all"}));
}

#[test]
fn prompt_cache_key_lands_at_top_level() {
    let body = body_with::<MoonshotExt, _>(
        &chat_wire(KIMI_K3),
        &options(MoonshotChat::new().prompt_cache_key("k")),
    );
    assert_eq!(body["prompt_cache_key"], "k");
}

#[test]
fn no_option_writes_a_leaf_the_request_or_a_mapped_option_owns() {
    let options = options(
        MoonshotChat::new()
            .thinking_keep(ThinkingKeep::All)
            .prompt_cache_key("k"),
    );
    assert_no_reserved_leaf::<MoonshotExt, _>(
        &[chat_wire(KIMI_K2_6), chat_wire(KIMI_K3)],
        &options,
    );
}

#[tokio::test]
async fn extras_from_a_built_reply() {
    let reply = reply_of(
        chat_wire(KIMI_K3),
        chat_reply(json!({
            "choices": [{"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}],
            "usage": {"prompt_tokens_details": {"cached_tokens": 0}}
        })),
    )
    .await;
    let extras = reply
        .extras::<MoonshotExt>()
        .unwrap_or_else(|| panic!("a Moonshot reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        extras
            .choice_usage
            .as_ref()
            .and_then(|usage| usage.get("total_tokens")),
        Some(&json!(2))
    );
    assert_eq!(
        extras.prompt_tokens_details,
        Some(json!({"cached_tokens": 0}))
    );
}

fn messages_wire() -> Messages {
    AnthropicConfig::with_key(&MESSAGES_MOONSHOT, "sk-test")
        .completion(crate::providers::moonshot::KIMI_K2_6)
}

/// The response `reply` folds into on Moonshot's Messages wire.
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

/// Built here, not recorded: no Moonshot Messages reply is recorded.
#[test]
fn extras_read_the_messages_stop_fields() {
    let reply = json!({
        "type": "message", "id": "msg_1", "model": crate::providers::moonshot::KIMI_K2_6, "role": "assistant",
        "content": [{"type": "text", "text": "alpha"}],
        "stop_reason": "stop_sequence", "stop_sequence": "alpha",
        "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    let response = folded(&reply);
    let extras = response
        .extras::<MoonshotExt>()
        .expect("a Moonshot reply")
        .expect("the extras read");
    assert_eq!(extras.stop_reason.as_deref(), Some("stop_sequence"));
    assert_eq!(extras.stop_sequence.as_deref(), Some("alpha"));
    assert!(response.extras::<AnthropicExt>().is_none());
}
