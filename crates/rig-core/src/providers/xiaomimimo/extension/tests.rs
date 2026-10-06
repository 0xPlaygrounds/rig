//! Xiaomi MiMo's extras. No MiMo recording exists, so the test decodes a
//! reply built here.

use serde_json::json;

use super::*;
use crate::providers::openai::wire::{OpenAIConfig, XIAOMIMIMO};
use crate::test_utils::provider_extensions::{chat_reply, reply_of};

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
        .extras::<XiaomiMimo>()
        .unwrap_or_else(|| panic!("a MiMo reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(extras.annotations, Some(vec![annotation]));
}
