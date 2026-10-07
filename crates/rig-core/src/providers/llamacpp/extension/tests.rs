//! llama.cpp's options as the bodies they encode to, and its extras read
//! from a recorded reply. The encoder tests are unit tests because no
//! recording sends typed provider options.

use serde_json::{Value, json};

use super::*;
use crate::providers::openai::wire::{Chat, LLAMACPP, OpenAIConfig};
use crate::test_utils::provider_extensions::{
    assert_no_reserved_leaf, body_with, recorded_reply, reply_of,
};

fn chat_wire() -> Chat {
    OpenAIConfig::with_key(&LLAMACPP, "key").chat(crate::providers::llamacpp::LLAMA_CPP)
}

fn body(options: &LlamaCppOptions) -> Value {
    body_with::<LlamaCppExt, _>(&chat_wire(), options)
}

#[test]
fn chat_template_kwargs_land_at_top_level() {
    let body = body(&LlamaCppOptions::new().chat_template_kwarg("enable_thinking", json!(false)));
    assert_eq!(
        body["chat_template_kwargs"],
        json!({"enable_thinking": false})
    );
}

#[test]
fn reasoning_format_lands_at_top_level() {
    let body = body(&LlamaCppOptions::new().reasoning_format(ReasoningFormat::DeepseekLegacy));
    assert_eq!(body["reasoning_format"], "deepseek-legacy");
}

#[test]
fn n_probs_lands_at_top_level() {
    assert_eq!(body(&LlamaCppOptions::new().n_probs(3))["n_probs"], 3);
}

#[test]
fn samplers_land_at_top_level() {
    let body = body(&LlamaCppOptions::new().samplers(["top_k", "temperature"]));
    assert_eq!(body["samplers"], json!(["top_k", "temperature"]));
}

#[test]
fn top_k_lands_at_top_level() {
    assert_eq!(body(&LlamaCppOptions::new().top_k(20))["top_k"], 20);
}

#[test]
fn min_p_lands_at_top_level() {
    assert_eq!(body(&LlamaCppOptions::new().min_p(0.1))["min_p"], 0.1);
}

#[test]
fn typical_p_lands_at_top_level() {
    assert_eq!(
        body(&LlamaCppOptions::new().typical_p(0.9))["typical_p"],
        0.9
    );
}

#[test]
fn mirostat_lands_at_top_level() {
    assert_eq!(body(&LlamaCppOptions::new().mirostat(2))["mirostat"], 2);
}

#[test]
fn mirostat_tau_lands_at_top_level() {
    assert_eq!(
        body(&LlamaCppOptions::new().mirostat_tau(5.0))["mirostat_tau"],
        5.0
    );
}

#[test]
fn mirostat_eta_lands_at_top_level() {
    assert_eq!(
        body(&LlamaCppOptions::new().mirostat_eta(0.1))["mirostat_eta"],
        0.1
    );
}

#[test]
fn id_slot_lands_at_top_level() {
    assert_eq!(body(&LlamaCppOptions::new().id_slot(1))["id_slot"], 1);
}

#[test]
fn timings_per_token_lands_at_top_level() {
    let body = body(&LlamaCppOptions::new().timings_per_token(true));
    assert_eq!(body["timings_per_token"], true);
}

#[test]
fn no_option_writes_a_leaf_the_request_or_a_mapped_option_owns() {
    let options = LlamaCppOptions::new()
        .chat_template_kwarg("enable_thinking", json!(true))
        .reasoning_format(ReasoningFormat::Auto)
        .n_probs(1)
        .samplers(["top_k"])
        .top_k(1)
        .min_p(0.1)
        .typical_p(0.9)
        .mirostat(1)
        .mirostat_tau(5.0)
        .mirostat_eta(0.1)
        .id_slot(0)
        .timings_per_token(false);
    assert_no_reserved_leaf::<LlamaCppExt, _>(&[chat_wire()], &options);
}

#[tokio::test]
async fn extras_from_a_unary_recording() {
    let reply = reply_of(
        chat_wire(),
        recorded_reply("llamacpp", "agent/completion_smoke", 0),
    )
    .await;
    let extras = reply
        .extras::<LlamaCppExt>()
        .unwrap_or_else(|| panic!("a llama.cpp reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    let timings = extras.timings.unwrap_or_default();
    assert_eq!(timings.cache_n, Some(0));
    assert_eq!(timings.prompt_n, Some(40));
    assert_eq!(timings.prompt_ms, Some(84.787));
    assert_eq!(timings.predicted_n, Some(46));
    assert_eq!(timings.predicted_ms, Some(254.504));
}

/// A stream's `raw` is the unary document: the recorded stream's timings
/// read as the recorded unary answer of the same prompt states them. Each
/// answer measures its own milliseconds.
#[tokio::test]
async fn extras_read_alike_from_a_recorded_stream() {
    use crate::test_utils::provider_extensions::{recorded_stream, streamed_reply_of};

    let read = |reply: crate::completion::CompletionResponse| {
        reply
            .extras::<LlamaCppExt>()
            .unwrap_or_else(|| panic!("a llama.cpp reply"))
            .unwrap_or_else(|error| panic!("{error}"))
            .timings
            .unwrap_or_default()
    };
    let streamed = read(
        streamed_reply_of(
            chat_wire(),
            recorded_stream(
                "llamacpp",
                "truncation_matrix/streaming_tool_call_cut_mid_arguments",
                0,
            ),
        )
        .await,
    );
    let unary = read(
        reply_of(
            chat_wire(),
            recorded_reply(
                "llamacpp",
                "truncation_matrix/tool_call_cut_mid_arguments",
                0,
            ),
        )
        .await,
    );
    assert!(streamed.prompt_ms.is_some() && streamed.predicted_ms.is_some());
    assert_eq!(streamed.cache_n, unary.cache_n);
    assert_eq!(streamed.prompt_n, unary.prompt_n);
    assert_eq!(streamed.predicted_n, unary.predicted_n);
}
