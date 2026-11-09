//! Recorded matrix for the Doubleword embedding-width contract.
//!
//! Every cell that names a width rig can promise pins one invariant: the vectors
//! Doubleword returns are exactly `EmbeddingModel::ndims()` wide. A vector store
//! sizes its index from that number (`rig-neo4j` validates and creates indexes
//! with it, `rig-sqlite` sizes its table from it), so a width rig reports but
//! never receives is a broken index. The `width_default_*` and
//! `builder_*_default_*` cells pin the model's default width, and the
//! `width_<n>_*` cells pin that a requested `dimensions` reaches the wire and the
//! response. The provider-refusal cells return no vectors and sit outside the
//! invariant.
//!
//! Widths outside Doubleword's documented 32 to 4096 range are rejected before
//! the request is built, so they are unit-tested in
//! `crates/rig-core/src/providers/doubleword/embedding.rs` instead. Each cell
//! asserts against the bytes its own cassette recorded: the width the response
//! carried and the `dimensions` the request sent.

use axum::http;
use rig::providers::doubleword;
use rig_test_support::cassette_models::OpenAiModels;

use super::super::support::{RecordedEmbeddingCall, with_doubleword_embedding_cassette};

const MODEL: &str = doubleword::QWEN3_EMBEDDING_8B;

const PROBE: &str = "width probe";

/// An input Doubleword itself refuses, at a width that still had to reach the
/// wire for the refusal to be the provider's rather than rig's.
async fn assert_rejected_input(client: &OpenAiModels, input: &str) {
    let error = client
        .embedding(MODEL, Some(512))
        .call(vec![input.to_string()])
        .await
        .map(|response| response.embeddings)
        .expect_err("Doubleword should reject a contentless input");

    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::BAD_REQUEST),
        "expected Doubleword's own 400: {error}"
    );
}

fn assert_rejected_call(calls: &[RecordedEmbeddingCall], expected_on_the_wire: usize) {
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].requested_dimensions,
        Some(expected_on_the_wire),
        "the width must have reached the wire before the provider refused"
    );
    assert!(
        calls[0].returned_widths.is_empty(),
        "a refused turn returns no vectors"
    );
}

// ================================================================
// Width sweep, single input
// ================================================================

// ================================================================
// Width sweep, batched inputs
// ================================================================

// ================================================================
// Adjacent entry points that share the same hook
// ================================================================

// ================================================================
// Input classes, all at one truncated width
// ================================================================

#[tokio::test]
async fn empty_input_at_a_requested_width() {
    // Doubleword rejects an empty input outright. Recorded because the
    // rejection is the *provider's*, arriving after the width reached the
    // wire — rig adds no input-side guard that would have hidden it.
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/empty_input_at_a_requested_width",
        |client| async move { assert_rejected_input(&client, "").await },
    )
    .await;
    assert_rejected_call(&calls, 512);
}

// No whitespace-only input cell: Doubleword answers that body with 400 or 200
// at random, so a cassette would pin a coin flip.

// ================================================================
// Two widths in one scenario, and the unknown-model escape hatch
// ================================================================

#[tokio::test]
async fn an_unknown_model_still_puts_the_requested_width_on_the_wire() {
    // rig polices only the range it has a table for. For a model it does not
    // know, the caller's width is the only width there is: it goes out
    // unvalidated and Doubleword — not rig — decides. Recorded against a model
    // id Doubleword does not serve, so the width is provably on the wire while
    // the call still fails.
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/an_unknown_model_still_puts_the_requested_width_on_the_wire",
        |client| async move {
            let model = client.embedding("Qwen/Qwen4-Embedding-Unreleased", Some(8_192));
            assert_eq!(model.capabilities().ndims, 8_192);
            let error = model
                .call(vec![PROBE.to_string()])
                .await
                .map(|response| response.embeddings)
                .expect_err("an unserved model should fail");
            assert_eq!(
                error.provider_response_status(),
                Some(http::StatusCode::NOT_FOUND),
                "unserved model should surface the provider's status: {error}"
            );
        },
    )
    .await;

    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].requested_dimensions, Some(8_192));
    assert!(
        calls[0].returned_widths.is_empty(),
        "an error turn returns no vectors"
    );
}
