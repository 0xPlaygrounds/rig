//! The embeddings dimension matrix — and the width llama.cpp will not resize.
//!
//! **Server**: `--embeddings --pooling mean` on
//! `Qwen/Qwen3-Embedding-0.6B-GGUF` Q8_0, `--seed 42 --temp 0 -c 2048`,
//! `llama-server` b10499-6d05498. A real embedding model, not a causal LM
//! pooled into one; what a causal LM does under the same flags is
//! `error_matrix::embeddings_on_a_causal_lm_return_pooled_numbers`.
//!
//! | Cell | Dimension | Pinned |
//! | --- | --- | --- |
//! | [`the_native_width_comes_back_when_none_is_declared`] | no `ndims` | the model's own width, and `dimensions` never on the wire |
//! | [`a_declared_width_that_matches_is_accepted`] | correct `ndims` | the declaration is believed because it is true |
//! | [`a_declared_width_llamacpp_cannot_honour_is_refused`] | wrong `ndims` | `MismatchedDimensions`, not a silent 1,024 behind an `ndims()` of 128 |
//! | [`several_inputs_come_back_in_order_at_one_width`] | batch | one vector per input, same width, input order |
//!
//! # The defect this matrix exists for
//!
//! `llama-server` has no `dimensions` handling at all — the string does not
//! appear anywhere under `tools/server/`. A request asking for 128 answers
//! 200 with the loaded model's 1,024-wide vectors, and before this PR rig's
//! `ndims()` went on reporting 128.
//!
//! That number is what a vector store sizes its index from, so the mismatch
//! surfaces as an index that cannot hold its own vectors — far from the call
//! that caused it, and with nothing in the response saying anything was wrong.
//! Two changes close it, at the two layers involved: this provider stops
//! sending a field the server ignores, and the shared OpenAI-compatible
//! embeddings driver compares an **explicitly declared** width against the one
//! that came back. A handle built without a width is untouched: it reports
//! whatever the provider's own table says and has nothing to disagree with.

use rig::client::EmbeddingsClient;
use rig::embeddings::{EmbeddingError, EmbeddingModel};
use serde_json::Value;

use crate::cassettes::{recorded_json_request, recorded_statuses_and_bodies};

use super::super::cassette_support::*;

/// The width `Qwen/Qwen3-Embedding-0.6B-GGUF` returns, measured.
const NATIVE_WIDTH: usize = 1024;

fn recorded_widths(scenario: &str) -> Vec<usize> {
    let recorded = recorded_statuses_and_bodies("llamacpp", scenario);
    let (status, body) = recorded.last().expect("an interaction");
    assert_eq!(*status, 200, "{scenario}: {body}");
    let response: Value = serde_json::from_str(body).expect("response should be JSON");
    response["data"]
        .as_array()
        .unwrap_or_else(|| panic!("{scenario}: data array: {response}"))
        .iter()
        .map(|datum| {
            datum["embedding"]
                .as_array()
                .unwrap_or_else(|| panic!("{scenario}: embedding array"))
                .len()
        })
        .collect()
}

#[tokio::test]
async fn the_native_width_comes_back_when_none_is_declared() {
    with_llamacpp_embeddings_cassette("embedding_matrix/native_width", |client| async move {
        let model = client.embedding_model(CASSETTE_EMBEDDING_MODEL);
        let embeddings = model
            .embed_texts(["hello".to_string()])
            .await
            .expect("an undeclared width should succeed");

        assert_eq!(embeddings[0].vec.len(), NATIVE_WIDTH);
    })
    .await;

    let request = recorded_json_request("llamacpp", "embedding_matrix/native_width");
    assert!(
        request.get("dimensions").is_none(),
        "llama.cpp ignores `dimensions`, so this provider does not send it: {request}"
    );
    assert_eq!(
        recorded_widths("embedding_matrix/native_width"),
        vec![NATIVE_WIDTH]
    );
}

/// Declaring the width the model actually has is fine, and still sends nothing.
#[tokio::test]
async fn a_declared_width_that_matches_is_accepted() {
    with_llamacpp_embeddings_cassette(
        "embedding_matrix/declared_width_matches",
        |client| async move {
            let model = client.embedding_model_with_ndims(CASSETTE_EMBEDDING_MODEL, NATIVE_WIDTH);
            assert_eq!(model.ndims(), NATIVE_WIDTH);

            let embeddings = model
                .embed_texts(["hello".to_string()])
                .await
                .expect("a correct declaration should succeed");
            assert_eq!(embeddings[0].vec.len(), NATIVE_WIDTH);
        },
    )
    .await;

    let request = recorded_json_request("llamacpp", "embedding_matrix/declared_width_matches");
    assert!(
        request.get("dimensions").is_none(),
        "a declaration is not a request; nothing goes on the wire: {request}"
    );
}

/// Declaring a width llama.cpp cannot produce fails loudly.
///
/// The request is still sent — llama.cpp is the only thing that knows the
/// loaded model's width, so there is nothing to check against beforehand — and
/// the disagreement is caught on the way back.
#[tokio::test]
async fn a_declared_width_llamacpp_cannot_honour_is_refused() {
    with_llamacpp_embeddings_cassette(
        "embedding_matrix/declared_width_mismatches",
        |client| async move {
            let model = client.embedding_model_with_ndims(CASSETTE_EMBEDDING_MODEL, 128);
            let error = model
                .embed_texts(["hello".to_string()])
                .await
                .expect_err("llama.cpp cannot resize embeddings");

            match error {
                EmbeddingError::MismatchedDimensions {
                    provider,
                    requested,
                    returned,
                } => {
                    assert_eq!(provider, "llamacpp");
                    assert_eq!(requested, 128, "the width the caller declared");
                    assert_eq!(returned, NATIVE_WIDTH, "the width the model has");
                }
                other => panic!("expected MismatchedDimensions, got {other:?}"),
            }
        },
    )
    .await;

    // The premise, from the bytes: rig did *not* ask for 128 on the wire (the
    // field is meaningless here), and the server answered 200 with its own
    // width. Both halves matter — an error raised from a 4xx would be a
    // different, easier story.
    let request = recorded_json_request("llamacpp", "embedding_matrix/declared_width_mismatches");
    assert!(request.get("dimensions").is_none(), "{request}");
    let recorded =
        recorded_statuses_and_bodies("llamacpp", "embedding_matrix/declared_width_mismatches");
    assert_eq!(
        recorded[0].0, 200,
        "the server is perfectly happy; the disagreement is rig's to notice"
    );
    assert_eq!(
        recorded_widths("embedding_matrix/declared_width_mismatches"),
        vec![NATIVE_WIDTH]
    );
}

#[tokio::test]
async fn several_inputs_come_back_in_order_at_one_width() {
    with_llamacpp_embeddings_cassette("embedding_matrix/batch", |client| async move {
        let model = client.embedding_model(CASSETTE_EMBEDDING_MODEL);
        let inputs = [
            "alpha".to_string(),
            "bravo".to_string(),
            "charlie".to_string(),
        ];
        let embeddings = model
            .embed_texts(inputs.clone())
            .await
            .expect("a batch should succeed");

        assert_eq!(embeddings.len(), 3);
        for (embedding, input) in embeddings.iter().zip(inputs.iter()) {
            assert_eq!(
                &embedding.document, input,
                "each vector must stay paired with the text it came from"
            );
            assert_eq!(embedding.vec.len(), NATIVE_WIDTH);
        }
        assert_ne!(
            embeddings[0].vec, embeddings[1].vec,
            "different inputs must produce different vectors, or the batch \
             collapsed onto one"
        );
    })
    .await;

    let request = recorded_json_request("llamacpp", "embedding_matrix/batch");
    assert_eq!(
        request["input"],
        serde_json::json!(["alpha", "bravo", "charlie"]),
        "the batch goes out as one request in input order"
    );
    assert_eq!(
        recorded_widths("embedding_matrix/batch"),
        vec![NATIVE_WIDTH; 3]
    );
}
