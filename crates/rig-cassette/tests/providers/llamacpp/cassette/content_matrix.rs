//! Message-content shapes, and what the chat template does with each.
//!
//! **Server**: the default configuration — `unsloth/Qwen3-1.7B-GGUF` Q4_K_M,
//! `--jinja --seed 42 --temp 0 -c 4096`, `llama-server` b10964-b29c606e2.
//! Every claim here is about the request rig builds and the template's
//! tolerance for it, not about whether the model is clever, so the smoke tier
//! is the right tier for all of them.
//!
//! | Cell | Shape | Pinned |
//! | --- | --- | --- |
//! | [`an_answer_fully_consumed_by_a_stop_sequence_is_an_empty_turn`] | empty content | 200 on the wire, an empty successful turn |
//!
//! # Why the empty-answer cell exists
//!
//! A `stop` sequence that matches the model's very first token leaves
//! `content: ""` with `finish_reason: "stop"` and a perfectly healthy 200.
//! Rig keeps that as an empty turn, as pi does: the core fold judges
//! emptiness once for every wire, and the fixture holds the 200 beside it.

use serde_json::{Value, json};

use crate::cassettes::recorded_statuses_and_bodies;

use super::super::cassette_support::*;
use rig::completion::CompletionRequest;

/// A turn whose whole answer is eaten by a stop sequence is an empty turn:
/// llama.cpp answers `200` with `finish_reason: "stop"` and `content: ""`,
/// and rig keeps it as pi does, a success with no blocks, decided once by
/// the core fold for whole and streamed replies alike.
#[tokio::test]
async fn an_answer_fully_consumed_by_a_stop_sequence_is_an_empty_turn() {
    with_llamacpp_cassette(
        "content_matrix/empty_answer_with_stop",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let response = model
                .call(
                    CompletionRequest::new("Reply with exactly this and nothing else: STOPWORD")
                        .max_tokens(64)
                        // Qwen3 opens every turn with a `<think>` block, so
                        // this matches the model's very first emitted token
                        // and the whole answer is consumed before a character
                        // of it exists.
                        .stop(["<think>"]),
                )
                .await
                .expect("an empty turn that ended cleanly decodes");
            assert!(response.choice.is_empty(), "{:?}", response.choice);
            assert!(!response.stop().is_failure(), "{:?}", response.stop());
        },
    )
    .await;

    let recorded =
        recorded_statuses_and_bodies("llamacpp", "content_matrix/empty_answer_with_stop");
    let (status, body) = &recorded[0];
    assert_eq!(
        *status, 200,
        "the server did not fail; the emptiness is the whole story"
    );
    let response: Value = serde_json::from_str(body).expect("response should be JSON");
    let content = response["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(
        content.is_empty(),
        "the recorded content must itself be empty for this cell to mean anything: {content:?}"
    );
    assert_eq!(
        response["choices"][0]["finish_reason"],
        json!("stop"),
        "a stop sequence, not a length cut"
    );
}
