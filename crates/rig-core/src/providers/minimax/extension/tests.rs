//! MiniMax's options as the bodies they encode to, and its extras. The
//! encoder test is a unit test because no recording sends typed provider
//! options. No MiniMax recording exists, so the extras test decodes a reply
//! built here.

use serde_json::{Value, json};

use super::*;
use crate::completion::{CompletionRequest, CompletionResponse, ProviderOptions};
use crate::error::ProviderError;
use crate::operation::Completion;
use crate::providers::anthropic::extension::AnthropicExt;
use crate::providers::anthropic::wire::{AnthropicConfig, MINIMAX as MESSAGES_MINIMAX, Messages};
use crate::providers::minimax::{MINIMAX_M2_5, MINIMAX_M2_7};
use crate::providers::openai::wire::{Chat, MINIMAX, OpenAIConfig};
use crate::test_utils::provider_extensions::{
    assert_no_reserved_leaf, body_with, chat_reply, reply_of,
};
use crate::wire::{Body, Mode, Operation, Wire, WireFrame};

fn chat_wire(model: &str) -> Chat {
    OpenAIConfig::with_key(&MINIMAX, "key").chat(model)
}

fn options() -> MiniMaxOptions {
    MiniMaxOptions::new()
        .chat(MiniMaxChat::new().reasoning_split(true))
        .messages(MiniMaxMessages::new().metadata_user_id("u-1"))
}

#[test]
fn reasoning_split_lands_at_top_level() {
    let body = body_with::<MiniMaxExt, _>(&chat_wire(MINIMAX_M2_7), &options());
    assert_eq!(body["reasoning_split"], true);
}

#[test]
fn no_option_writes_a_leaf_the_request_or_a_mapped_option_owns() {
    assert_no_reserved_leaf::<MiniMaxExt, _>(
        &[
            chat_wire(MINIMAX_M2_7),
            chat_wire(MINIMAX_M2_5),
            chat_wire("MiniMax-M3"),
        ],
        &options(),
    );
}

#[tokio::test]
async fn extras_from_a_built_reply() {
    let reply = reply_of(
        chat_wire(MINIMAX_M2_7),
        chat_reply(json!({
            "choices": [{"message": {"reasoning_details": [{"type": "reasoning.text", "text": "hm"}]}}],
            "base_resp": {"status_code": 0, "status_msg": ""}
        })),
    )
    .await;
    let extras = reply
        .extras::<MiniMaxExt>()
        .unwrap_or_else(|| panic!("a MiniMax reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        extras.reasoning_details,
        Some(vec![json!({"type": "reasoning.text", "text": "hm"})])
    );
    assert_eq!(
        extras
            .base_resp
            .as_ref()
            .and_then(|status| status.get("status_code")),
        Some(&json!(0))
    );
}

fn messages_wire() -> Messages {
    AnthropicConfig::with_key(&MESSAGES_MINIMAX, "sk-test")
        .completion(crate::providers::minimax::MINIMAX_M2_7)
}

/// The response `reply` folds into on MiniMax's Messages wire.
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

/// The body the wire sends for `request`.
fn sent(request: CompletionRequest) -> Result<Value, ProviderError> {
    let wire = messages_wire();
    let request = Completion::prepare(request, &wire.describe())?;
    let encoded = wire.encode(request, Mode::Unary)?;
    let Body::Bytes(bytes) = encoded.request.body() else {
        return Err(ProviderError::request("a Messages body is JSON"));
    };
    Ok(serde_json::from_slice(bytes)?)
}

#[test]
fn messages_metadata_user_id_is_sent_on_the_messages_route() {
    let options = ProviderOptions::new()
        .with::<MiniMaxExt>(
            &MiniMaxOptions::new().messages(MiniMaxMessages::new().metadata_user_id("u-1")),
        )
        .expect("MiniMax options are sections");
    let body = sent(CompletionRequest::new("hi").provider_options(options)).expect("encodes");
    assert_eq!(body["metadata"], json!({"user_id": "u-1"}));
}

/// Built here, not recorded: no MiniMax Messages reply is recorded.
#[test]
fn extras_read_the_messages_stop_fields() {
    let reply = json!({
        "type": "message", "id": "msg_1", "model": crate::providers::minimax::MINIMAX_M2_7, "role": "assistant",
        "content": [{"type": "text", "text": "alpha"}],
        "stop_reason": "stop_sequence", "stop_sequence": "alpha",
        "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    let response = folded(&reply);
    let extras = response
        .extras::<MiniMaxExt>()
        .expect("a MiniMax reply")
        .expect("the extras read");
    assert_eq!(extras.stop_reason.as_deref(), Some("stop_sequence"));
    assert_eq!(extras.stop_sequence.as_deref(), Some("alpha"));
    assert!(response.extras::<AnthropicExt>().is_none());
}

/// Each `MiniMaxOptions` field setter equals its section form, as a value
/// and as the JSON it serializes to.
#[test]
fn field_setters_equal_the_section_form() {
    let pairs = [
        (
            MiniMaxOptions::new().reasoning_split(true),
            MiniMaxOptions::new().chat(MiniMaxChat::new().reasoning_split(true)),
        ),
        (
            MiniMaxOptions::new().metadata_user_id("u-1"),
            MiniMaxOptions::new().messages(MiniMaxMessages::new().metadata_user_id("u-1")),
        ),
        (
            MiniMaxOptions::new()
                .reasoning_split(false)
                .metadata_user_id("u-1"),
            MiniMaxOptions::new()
                .chat(MiniMaxChat::new().reasoning_split(false))
                .messages(MiniMaxMessages::new().metadata_user_id("u-1")),
        ),
    ];
    for (short, long) in pairs {
        assert_eq!(short, long);
        let short = serde_json::to_value(&short).expect("options serialize");
        assert_eq!(
            short,
            serde_json::to_value(&long).expect("options serialize")
        );
        assert_ne!(
            short,
            serde_json::to_value(MiniMaxOptions::new()).expect("options serialize")
        );
    }
}
