//! Sampling parameters, crossed against what `llama-server` does with them.
//!
//! **Server**: the default configuration — `unsloth/Qwen3-1.7B-GGUF` Q4_K_M,
//! `--jinja --seed 42 --temp 0 -c 4096`, `llama-server` b10964-b29c606e2 —
//! except where a cell names another. The smoke tier is enough for every cell
//! here: what is under test is whether a parameter reaches the wire and what
//! the server does with it, not whether the model is clever.
//!
//! | Cell | Parameter | Pinned |
//! | --- | --- | --- |
//! | [`a_fixed_seed_and_an_absent_seed_are_both_accepted`] | `seed` | present round-trips; absent falls back to the server's `--seed` |
//! | [`additional_params_wins_over_the_typed_field_it_collides_with`] | precedence | `additional_params` overrides a typed builder call, silently |
//!
//! # Two things worth knowing
//!
//! `stop` is **not** a field on rig's [`CompletionRequest`], so every stop cell
//! goes through `additional_params`. That is the supported route — the shared
//! OpenAI request merges `additional_params` into the body — but it means stop
//! sequences are untyped for every provider, and a caller gets no help with
//! case: stop matching is case-sensitive.
//!
//! `temperature: 0.0` is the interesting half of the temperature cell.
//! Serializing a zero as "absent" is a classic defect in OpenAI-compatible
//! clients — it turns a deterministic request into a sampled one and nothing
//! in the response says so. The cell reads the recorded request bytes rather
//! than trusting the builder.

use rig::providers::openai::wire::Chat;
use rig::providers::openai::wire::{LLAMACPP, OpenAIConfig};
use rig::wire::{Body, Mode, Wire};
use serde_json::{Value, json};

use crate::cassettes::recorded_json_request;

use super::super::cassette_support::*;
use rig::completion::CompletionRequest;

/// Qwen3 emits a `<think>` trace before answering and the chat-completions
/// route has no switch for it, so prompts that need a short literal answer
/// prefix `/no_think`, which the model's own template honours.
const NO_THINK: &str = "/no_think ";

// ---------------------------------------------------------------------------
// temperature
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// max_tokens
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// stop sequences
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// seed
// ---------------------------------------------------------------------------

/// `seed` present round-trips; absent falls back to the server's `--seed`.
///
/// Both halves are worth a cell because a recording harness depends on them:
/// the corpus is reproducible only because the *server* was started with
/// `--seed 42`, and a request that silently injected a different seed would
/// make every fixture in this suite unreproducible without saying so.
#[tokio::test]
async fn a_fixed_seed_and_an_absent_seed_are_both_accepted() {
    with_llamacpp_cassette("sampling_matrix/seed_fixed", |client| async move {
        let model = client.completion(CASSETTE_MODEL);
        model
            .call(
                CompletionRequest::new(format!("{NO_THINK}Say ok."))
                    .max_tokens(32)
                    .additional_params(json!({ "seed": 7 })),
            )
            .await
            .expect("an explicit seed should be accepted");
    })
    .await;

    with_llamacpp_cassette("sampling_matrix/seed_absent", |client| async move {
        let model = client.completion(CASSETTE_MODEL);
        model
            .call(CompletionRequest::new(format!("{NO_THINK}Say ok.")).max_tokens(32))
            .await
            .expect("no seed should be accepted");
    })
    .await;

    assert_eq!(
        recorded_json_request("llamacpp", "sampling_matrix/seed_fixed")["seed"],
        json!(7)
    );
    assert!(
        recorded_json_request("llamacpp", "sampling_matrix/seed_absent")
            .get("seed")
            .is_none(),
        "rig must not invent a seed the caller did not ask for; the corpus's \
         determinism comes from the server's --seed"
    );
}

// ---------------------------------------------------------------------------
// Precedence
// ---------------------------------------------------------------------------

/// `additional_params` **overrides** a typed field of the same name.
///
/// Half this matrix reaches the wire through `additional_params` — `stop`,
/// `seed`, `grammar`, `n`, `logprobs` all have no typed home — so which side
/// wins when the two collide is load-bearing for reading any of these
/// fixtures, and nothing stated it.
///
/// It is the escape hatch's `#[serde(flatten)]` that decides: the typed fields
/// serialize first and the flattened map is written over them, so
/// `additional_params` wins. That is defensible as an escape hatch, and it is
/// silent — a caller who sets `max_tokens(7)` and then passes an
/// `additional_params` blob that happens to carry `max_tokens` gets 99 with no
/// warning.
///
/// Definitional rather than observed: this is rig's serialization, not
/// llama.cpp's parsing, so it is checked without a server.
#[test]
fn additional_params_wins_over_the_typed_field_it_collides_with() {
    // `encode` needs no socket, so this stays a plain unit test.

    let request = rig::completion::CompletionRequest {
        model: None,
        chat_history: vec![rig::message::Message::User {
            content: vec![rig::message::UserContent::text("hi")],
        }],
        documents: vec![],
        tools: vec![],
        temperature: Some(0.0),
        max_tokens: Some(7),
        tool_choice: None,
        additional_params: Some(json!({ "max_tokens": 99, "top_k": 3 })),
        output_schema: None,
        record_telemetry_content: false,
        accept_unknown_finish_reasons: false,
    };

    let encoded = Chat::new(OpenAIConfig::with_key(&LLAMACPP, ""), "m")
        .encode(request, Mode::Unary)
        .expect("the request should encode");
    let Body::Bytes(bytes) = encoded.request.body() else {
        panic!("the chat wire sends a serialized body, not a multipart form")
    };
    let body: Value = serde_json::from_slice(bytes).expect("the chat body is JSON");

    assert_eq!(
        body["max_tokens"],
        json!(99),
        "the escape hatch overrides the typed field it collides with"
    );
    assert_eq!(
        body["top_k"],
        json!(3),
        "and a key with no typed counterpart passes through unchanged — which is \
         how `stop`, `seed`, `grammar`, `n` and `logprobs` reach the wire in this \
         matrix"
    );
    assert_eq!(
        body["temperature"],
        json!(0.0),
        "a typed field the blob does not name is untouched"
    );
}
