//! Together AI's options as the bodies they encode to, and its extras. The
//! encoder tests are unit tests because no recording sends typed provider
//! options. No Together recording exists, so the extras test decodes a
//! reply built here.

use serde_json::{Value, json};

use super::*;
use crate::providers::openai::wire::{Chat, OpenAIConfig, TOGETHER};
use crate::test_utils::provider_extensions::{
    assert_no_reserved_leaf, body_with, chat_reply, reply_of,
};

const MODEL: &str = "Qwen/Qwen3-235B-A22B-Thinking-2507";

fn chat_wire() -> Chat {
    OpenAIConfig::with_key(&TOGETHER, "key").chat(MODEL)
}

fn body(options: &TogetherOptions) -> Value {
    body_with::<TogetherExt, _>(&chat_wire(), options)
}

#[test]
fn chat_template_kwargs_land_at_top_level() {
    let body = body(&TogetherOptions::new().chat_template_kwarg("enable_thinking", json!(false)));
    assert_eq!(
        body["chat_template_kwargs"],
        json!({"enable_thinking": false})
    );
}

#[test]
fn top_k_lands_at_top_level() {
    assert_eq!(body(&TogetherOptions::new().top_k(50))["top_k"], 50);
}

#[test]
fn min_p_lands_at_top_level() {
    assert_eq!(body(&TogetherOptions::new().min_p(0.05))["min_p"], 0.05);
}

#[test]
fn repetition_penalty_lands_at_top_level() {
    let body = body(&TogetherOptions::new().repetition_penalty(1.2));
    assert_eq!(body["repetition_penalty"], 1.2);
}

#[test]
fn safety_model_lands_at_top_level() {
    let body = body(&TogetherOptions::new().safety_model("meta-llama/Meta-Llama-Guard-3-8B"));
    assert_eq!(body["safety_model"], "meta-llama/Meta-Llama-Guard-3-8B");
}

#[test]
fn no_option_writes_a_leaf_the_request_or_a_mapped_option_owns() {
    let options = TogetherOptions::new()
        .chat_template_kwarg("enable_thinking", json!(true))
        .top_k(1)
        .min_p(0.1)
        .repetition_penalty(1.0)
        .safety_model("guard");
    assert_no_reserved_leaf::<TogetherExt, _>(&[chat_wire()], &options);
}

#[tokio::test]
async fn extras_from_a_built_reply() {
    let reply = reply_of(
        chat_wire(),
        chat_reply(json!({
            "warnings": [{"message": "max_tokens was reduced"}],
            "choices": [{"message": {"reasoning": "The user greets."}}]
        })),
    )
    .await;
    let extras = reply
        .extras::<TogetherExt>()
        .unwrap_or_else(|| panic!("a Together reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        extras.warnings,
        Some(vec![json!({"message": "max_tokens was reduced"})])
    );
    assert_eq!(extras.reasoning.as_deref(), Some("The user greets."));
}
