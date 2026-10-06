//! Mistral's options as the bodies they encode to, and its extras read
//! from a recorded reply. The encoder tests are unit tests because no
//! recording sends typed provider options.

use serde_json::{Value, json};

use super::*;
use crate::providers::openai::wire::{Chat, MISTRAL, OpenAIConfig};
use crate::test_utils::provider_extensions::{
    assert_no_reserved_leaf, body_with, recorded_reply, reply_of,
};

fn chat_wire(model: &str) -> Chat {
    OpenAIConfig::with_key(&MISTRAL, "key").chat(model)
}

fn body(options: &MistralOptions) -> Value {
    body_with::<Mistral, _>(&chat_wire("magistral-medium-latest"), options)
}

#[test]
fn prompt_mode_lands_as_reasoning() {
    let body = body(&MistralOptions::new().prompt_mode(PromptMode::Reasoning));
    assert_eq!(body["prompt_mode"], "reasoning");
}

#[test]
fn safe_prompt_lands_at_top_level() {
    assert_eq!(
        body(&MistralOptions::new().safe_prompt(true))["safe_prompt"],
        true
    );
}

#[test]
fn prompt_cache_key_lands_at_top_level() {
    let body = body(&MistralOptions::new().prompt_cache_key("tenant-1"));
    assert_eq!(body["prompt_cache_key"], "tenant-1");
}

#[test]
fn frequency_penalty_lands_at_top_level() {
    let body = body(&MistralOptions::new().frequency_penalty(0.3));
    assert_eq!(body["frequency_penalty"], 0.3);
}

#[test]
fn presence_penalty_lands_at_top_level() {
    let body = body(&MistralOptions::new().presence_penalty(0.4));
    assert_eq!(body["presence_penalty"], 0.4);
}

#[test]
fn prediction_lands_as_content_prediction() {
    let body = body(&MistralOptions::new().prediction("draft"));
    assert_eq!(
        body["prediction"],
        json!({"type": "content", "content": "draft"})
    );
}

#[test]
fn no_option_writes_a_leaf_the_request_or_a_mapped_option_owns() {
    let options = MistralOptions::new()
        .prompt_mode(PromptMode::Reasoning)
        .safe_prompt(false)
        .prompt_cache_key("k")
        .frequency_penalty(0.1)
        .presence_penalty(0.1)
        .prediction("p");
    assert_no_reserved_leaf::<Mistral, _>(
        &[
            chat_wire("magistral-medium-latest"),
            chat_wire("mistral-large-latest"),
        ],
        &options,
    );
}

#[tokio::test]
async fn extras_from_a_unary_recording() {
    let reply = reply_of(
        chat_wire("voxtral-small-latest"),
        recorded_reply(
            "mistral",
            "multimodal_content/blocking_raw_model_sends_audio",
            0,
        ),
    )
    .await;
    let extras = reply
        .extras::<Mistral>()
        .unwrap_or_else(|| panic!("a Mistral reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(extras.service_tier.as_deref(), Some("standard"));
    assert_eq!(extras.prompt_audio_seconds, Some(5));
    assert_eq!(extras.num_cached_tokens, None);
    let details = extras.prompt_tokens_details.unwrap_or_default();
    assert_eq!(details.audio_tokens, Some(375));
    assert_eq!(details.cached_tokens, Some(9));
}
