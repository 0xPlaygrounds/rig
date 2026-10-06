//! MiniMax's options as the bodies they encode to, and its extras. The
//! encoder test is a unit test because no recording sends typed provider
//! options. No MiniMax recording exists, so the extras test decodes a reply
//! built here.

use serde_json::json;

use super::*;
use crate::providers::minimax::{MINIMAX_M2_5, MINIMAX_M2_7};
use crate::providers::openai::wire::{Chat, MINIMAX, OpenAIConfig};
use crate::test_utils::provider_extensions::{
    assert_no_reserved_leaf, body_with, chat_reply, reply_of,
};

fn chat_wire(model: &str) -> Chat {
    OpenAIConfig::with_key(&MINIMAX, "key").chat(model)
}

fn options() -> MiniMaxOptions {
    MiniMaxOptions::new().chat(MiniMaxChat::new().reasoning_split(true))
}

#[test]
fn reasoning_split_lands_at_top_level() {
    let body = body_with::<MiniMax, _>(&chat_wire(MINIMAX_M2_7), &options());
    assert_eq!(body["reasoning_split"], true);
}

#[test]
fn no_option_writes_a_leaf_the_request_or_a_mapped_option_owns() {
    assert_no_reserved_leaf::<MiniMax, _>(
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
        .extras::<MiniMax>()
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
