//! Recorded matrix for the Doubleword embedding-width contract.
//!
//! One invariant runs through every cell: **the vectors Doubleword returns are
//! exactly `EmbeddingModel::ndims()` wide**. That is the number a vector store
//! sizes its index from (`rig-neo4j` validates an existing index against it
//! and creates a new one with it; `rig-sqlite` sizes its table from it), so a
//! width rig reports but never receives is a broken index, not a cosmetic
//! mismatch.
//!
//! Two defects broke the invariant in opposite directions, and the matrix
//! separates them:
//!
//! - **no width at all** — `default_ndims` was unimplemented, so the only
//!   embedding model Doubleword ships reported `ndims() == 0` while returning
//!   4096-wide vectors. The `width_default_*` and `builder_*_default_*` cells
//!   pin this half; their recorded request bodies are byte-identical before
//!   and after the fix, so they fail on `origin/main` as a clean assertion
//!   failure (`4096 != 0`) with no mock miss.
//! - **the wrong width** — a caller-requested `dimensions` was dropped on the
//!   floor, so `embedding_model_with_ndims(model, 512)` reported 512 and
//!   received 4096. Every `width_<n>_*` cell pins this half; the fix changes
//!   what rig *sends*, so on `origin/main` these replay as a mock miss (the
//!   recorded body carries `"dimensions":<n>`, main sends none) *plus* the
//!   width mismatch. That is deliberate, per `tests/README.md`.
//!
//! Widths outside Doubleword's documented 32–4096 range are rejected before
//! the request is built, so they cannot be recorded; they are unit-tested in
//! `crates/rig-core/src/providers/doubleword/embedding.rs` instead. The reason
//! the ceiling needs guarding at all is recorded there too: Doubleword answers
//! an over-wide request `200 OK` with a silently clamped 4096-wide vector, so
//! letting it through would have reintroduced exactly this bug.
//!
//! Each cell asserts against the bytes its own cassette recorded — the width
//! the response actually carried and the `dimensions` the request actually
//! sent — rather than against what the test expected to happen.

use axum::http;
use rig::client::EmbeddingsClient;
use rig::embeddings::{EmbeddingModel, EmbeddingsBuilder};
use rig::providers::doubleword;

use super::super::support::{RecordedEmbeddingCall, with_doubleword_embedding_cassette};

const MODEL: &str = doubleword::QWEN3_EMBEDDING_8B;
/// The width Doubleword returns when a request names none.
const NATIVE_WIDTH: usize = 4_096;

const PROBE: &str = "width probe";
const BATCH: [&str; 3] = ["alpha probe", "bravo probe", "charlie probe"];

/// Asserts the matrix invariant against one cell's recorded bytes.
///
/// `expected_on_the_wire` is what the request should have carried: `None` for
/// "no `dimensions` field at all". Taking it as a parameter rather than
/// deriving it from `width` is the point — the native-width cells are the ones
/// that must *not* send the field, and a derived expectation would agree with
/// whatever the code did.
fn assert_recorded(
    calls: &[RecordedEmbeddingCall],
    expected_on_the_wire: &[Option<usize>],
    expected_width: usize,
    expected_vectors: &[usize],
) {
    assert_eq!(
        calls.len(),
        expected_on_the_wire.len(),
        "cell should have recorded {} round trip(s)",
        expected_on_the_wire.len()
    );

    for (index, call) in calls.iter().enumerate() {
        assert_eq!(
            call.requested_dimensions, expected_on_the_wire[index],
            "round trip {index}: recorded request carried the wrong `dimensions`"
        );
    }

    let widths: Vec<usize> = calls
        .iter()
        .flat_map(|call| call.returned_widths.iter().copied())
        .collect();
    assert_eq!(
        widths.len(),
        expected_vectors.iter().sum::<usize>(),
        "cell should have recorded {expected_vectors:?} vectors"
    );
    for width in &widths {
        assert_eq!(
            *width, expected_width,
            "recorded vector width should be {expected_width}"
        );
    }
}

/// One input at one width: the spine of the matrix.
///
/// Written as a helper because the cassette wrapper's scenario literal has to
/// stay at each `#[tokio::test]` call site for the safety scan to see it — so
/// the bodies are shared and the literals are not.
async fn assert_single_input_width(calls: Vec<RecordedEmbeddingCall>, width: usize) {
    let on_the_wire = (width != NATIVE_WIDTH).then_some(width);
    assert_recorded(&calls, &[on_the_wire], width, &[1]);
}

fn embedding_model(
    client: &doubleword::Client,
    width: Option<usize>,
) -> doubleword::EmbeddingModel {
    match width {
        Some(width) => client.embedding_model_with_ndims(MODEL, width),
        None => client.embedding_model(MODEL),
    }
}

/// Runs one input through the model and proves `ndims()` described the vector
/// that came back. Returns nothing: the cassette bytes are checked by the
/// caller, so a cell that stops exercising the model cannot pass silently.
async fn embed_one(client: &doubleword::Client, width: Option<usize>, input: &str) {
    let model = embedding_model(client, width);
    let embeddings = model
        .embed_texts([input.to_string()])
        .await
        .expect("embedding request should succeed");

    assert_eq!(embeddings.len(), 1);
    assert_eq!(
        embeddings[0].vec.len(),
        model.ndims(),
        "the returned vector must be exactly as wide as `ndims()` reports"
    );
}

async fn embed_batch(client: &doubleword::Client, width: Option<usize>, inputs: &[&str]) {
    let model = embedding_model(client, width);
    let embeddings = model
        .embed_texts(inputs.iter().map(|input| (*input).to_string()))
        .await
        .expect("embedding request should succeed");

    assert_eq!(embeddings.len(), inputs.len());
    for embedding in &embeddings {
        assert_eq!(
            embedding.vec.len(),
            model.ndims(),
            "every returned vector must be exactly as wide as `ndims()` reports"
        );
    }
}

/// An input Doubleword itself refuses, at a width that still had to reach the
/// wire for the refusal to be the provider's rather than rig's.
async fn assert_rejected_input(client: &doubleword::Client, input: &str) {
    let error = client
        .embedding_model_with_ndims(MODEL, 512)
        .embed_texts([input.to_string()])
        .await
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

#[tokio::test]
async fn width_default_single() {
    // The `default_ndims` half of the bug, in its purest form: no caller
    // width, no `dimensions` on the wire before or after the fix, and a
    // 4096-wide vector against an `ndims()` that used to be 0.
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/width_default_single",
        |client| async move { embed_one(&client, None, PROBE).await },
    )
    .await;
    assert_single_input_width(calls, NATIVE_WIDTH).await;
}

#[tokio::test]
async fn width_32_single() {
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/width_32_single",
        |client| async move { embed_one(&client, Some(32), PROBE).await },
    )
    .await;
    assert_single_input_width(calls, 32).await;
}

#[tokio::test]
async fn width_64_single() {
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/width_64_single",
        |client| async move { embed_one(&client, Some(64), PROBE).await },
    )
    .await;
    assert_single_input_width(calls, 64).await;
}

#[tokio::test]
async fn width_128_single() {
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/width_128_single",
        |client| async move { embed_one(&client, Some(128), PROBE).await },
    )
    .await;
    assert_single_input_width(calls, 128).await;
}

#[tokio::test]
async fn width_256_single() {
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/width_256_single",
        |client| async move { embed_one(&client, Some(256), PROBE).await },
    )
    .await;
    assert_single_input_width(calls, 256).await;
}

#[tokio::test]
async fn width_512_single() {
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/width_512_single",
        |client| async move { embed_one(&client, Some(512), PROBE).await },
    )
    .await;
    assert_single_input_width(calls, 512).await;
}

#[tokio::test]
async fn width_1024_single() {
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/width_1024_single",
        |client| async move { embed_one(&client, Some(1_024), PROBE).await },
    )
    .await;
    assert_single_input_width(calls, 1_024).await;
}

#[tokio::test]
async fn width_2048_single() {
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/width_2048_single",
        |client| async move { embed_one(&client, Some(2_048), PROBE).await },
    )
    .await;
    assert_single_input_width(calls, 2_048).await;
}

#[tokio::test]
async fn width_4095_single() {
    // One below the ceiling: still a truncation request, so it must reach the
    // wire — the boundary the "equals the native width" suppression must not
    // swallow.
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/width_4095_single",
        |client| async move { embed_one(&client, Some(4_095), PROBE).await },
    )
    .await;
    assert_single_input_width(calls, 4_095).await;
}

#[tokio::test]
async fn width_4096_single_is_not_sent() {
    // Naming the native width explicitly must produce the *same* request as
    // naming nothing — otherwise every already-recorded caller's request body
    // changes for a field that cannot change the answer.
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/width_4096_single_is_not_sent",
        |client| async move { embed_one(&client, Some(NATIVE_WIDTH), PROBE).await },
    )
    .await;
    assert_single_input_width(calls, NATIVE_WIDTH).await;
}

// ================================================================
// Width sweep, batched inputs
// ================================================================

#[tokio::test]
async fn width_32_batch() {
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/width_32_batch",
        |client| async move { embed_batch(&client, Some(32), &BATCH).await },
    )
    .await;
    assert_recorded(&calls, &[Some(32)], 32, &[BATCH.len()]);
}

#[tokio::test]
async fn width_128_batch() {
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/width_128_batch",
        |client| async move { embed_batch(&client, Some(128), &BATCH).await },
    )
    .await;
    assert_recorded(&calls, &[Some(128)], 128, &[BATCH.len()]);
}

#[tokio::test]
async fn width_512_batch() {
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/width_512_batch",
        |client| async move { embed_batch(&client, Some(512), &BATCH).await },
    )
    .await;
    assert_recorded(&calls, &[Some(512)], 512, &[BATCH.len()]);
}

// ================================================================
// Adjacent entry points that share the same hook
// ================================================================

#[tokio::test]
async fn usage_survives_a_requested_width() {
    // `embed_texts_with_usage` is a second entry point into the same request
    // builder; a width that reached the wire must not cost the usage the
    // caller came for.
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/usage_survives_a_requested_width",
        |client| async move {
            let model = client.embedding_model_with_ndims(MODEL, 256);
            let response = model
                .embed_texts_with_usage([PROBE.to_string()])
                .await
                .expect("embedding request should succeed");

            assert_eq!(response.embeddings.len(), 1);
            assert_eq!(response.embeddings[0].vec.len(), model.ndims());
            assert!(
                response.usage.input_tokens > 0 && response.usage.total_tokens > 0,
                "Doubleword reports embedding usage: {:?}",
                response.usage
            );
        },
    )
    .await;
    assert_recorded(&calls, &[Some(256)], 256, &[1]);
}

#[tokio::test]
async fn usage_at_the_default_width() {
    // The `with_usage` entry point on the other side of the fix: no caller
    // width, so nothing goes on the wire and `ndims()` must come from the
    // model table alone.
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/usage_at_the_default_width",
        |client| async move {
            let model = client.embedding_model(MODEL);
            let response = model
                .embed_texts_with_usage([PROBE.to_string()])
                .await
                .expect("embedding request should succeed");

            assert_eq!(response.embeddings.len(), 1);
            assert_eq!(response.embeddings[0].vec.len(), model.ndims());
            assert!(
                response.usage.input_tokens > 0 && response.usage.total_tokens > 0,
                "Doubleword reports embedding usage: {:?}",
                response.usage
            );
        },
    )
    .await;
    assert_recorded(&calls, &[None], NATIVE_WIDTH, &[1]);
}

#[tokio::test]
async fn builder_documents_at_a_requested_width() {
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/builder_documents_at_a_requested_width",
        |client| async move {
            let model = client.embedding_model_with_ndims(MODEL, 128);
            let documents = EmbeddingsBuilder::new(model.clone())
                .document("first note".to_string())
                .expect("document should embed")
                .document("second note".to_string())
                .expect("document should embed")
                .build()
                .await
                .expect("embeddings builder should succeed");

            assert_eq!(documents.len(), 2);
            for (_, embeddings) in &documents {
                for embedding in embeddings.iter() {
                    assert_eq!(embedding.vec.len(), model.ndims());
                }
            }
        },
    )
    .await;
    assert_recorded(&calls, &[Some(128)], 128, &[2]);
}

#[tokio::test]
async fn builder_documents_at_the_default_width() {
    // The builder path with no caller width: the `ndims() == 0` half of the
    // bug reached vector stores through exactly this route.
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/builder_documents_at_the_default_width",
        |client| async move {
            let model = client.embedding_model(MODEL);
            let documents = EmbeddingsBuilder::new(model.clone())
                .document("first note".to_string())
                .expect("document should embed")
                .document("second note".to_string())
                .expect("document should embed")
                .build()
                .await
                .expect("embeddings builder should succeed");

            assert_eq!(documents.len(), 2);
            for (_, embeddings) in &documents {
                for embedding in embeddings.iter() {
                    assert_eq!(embedding.vec.len(), model.ndims());
                }
            }
        },
    )
    .await;
    assert_recorded(&calls, &[None], NATIVE_WIDTH, &[2]);
}

#[tokio::test]
async fn a_zero_width_is_treated_as_no_width() {
    // The shared path reads 0 as "the caller said nothing", so the hook never
    // sees it. Recorded rather than assumed: the request must go out bare and
    // the model must answer at its native width, even though `ndims()` still
    // reports the 0 the caller asked for.
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/a_zero_width_is_treated_as_no_width",
        |client| async move {
            let model = client.embedding_model_with_ndims(MODEL, 0);
            assert_eq!(model.ndims(), 0);
            let embeddings = model
                .embed_texts([PROBE.to_string()])
                .await
                .expect("embedding request should succeed");
            assert_eq!(embeddings[0].vec.len(), NATIVE_WIDTH);
        },
    )
    .await;
    assert_recorded(&calls, &[None], NATIVE_WIDTH, &[1]);
}

// ================================================================
// Input classes, all at one truncated width
// ================================================================

#[tokio::test]
async fn unicode_input_at_a_requested_width() {
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/unicode_input_at_a_requested_width",
        |client| async move {
            embed_one(&client, Some(512), "こんにちは 🌍 café — naïve résumé").await
        },
    )
    .await;
    assert_single_input_width(calls, 512).await;
}

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

// DROPPED CELL — `whitespace_input_at_a_requested_width` ("   \n\t  " at 512).
// Doubleword answers that exact body non-deterministically: six consecutive
// live requests returned 400, 200, 400, 400, 200, 400, the 200s carrying a
// well-formed 512-wide vector. A cassette can only record one of the two, and
// whichever it recorded the cell would assert a premise the provider does not
// hold, so it is dropped rather than pinned to a coin flip. The neighbouring
// `empty_input_at_a_requested_width` *is* stable (400 on six of six) and
// covers the contentless-input class.

#[tokio::test]
async fn long_input_at_a_requested_width() {
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/long_input_at_a_requested_width",
        |client| async move {
            let long = "Doubleword fronts heterogeneous open-weight backends. ".repeat(64);
            embed_one(&client, Some(512), &long).await
        },
    )
    .await;
    assert_single_input_width(calls, 512).await;
}

#[tokio::test]
async fn repeated_input_at_a_requested_width() {
    // Identical inputs in one batch: the truncation must apply per vector, not
    // once per distinct string.
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/repeated_input_at_a_requested_width",
        |client| async move { embed_batch(&client, Some(512), &[PROBE, PROBE, PROBE]).await },
    )
    .await;
    assert_recorded(&calls, &[Some(512)], 512, &[3]);
}

// ================================================================
// Two widths in one scenario, and the unknown-model escape hatch
// ================================================================

#[tokio::test]
async fn two_widths_in_one_scenario_stay_independent() {
    // Same client, same text, two models at different widths: the width is a
    // property of the model, not of the client or the last request made.
    let calls = with_doubleword_embedding_cassette(
        "embedding_dimensions/two_widths_in_one_scenario_stay_independent",
        |client| async move {
            embed_one(&client, Some(64), PROBE).await;
            embed_one(&client, Some(1_024), PROBE).await;
        },
    )
    .await;

    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].requested_dimensions, Some(64));
    assert_eq!(calls[0].returned_widths, vec![64]);
    assert_eq!(calls[1].requested_dimensions, Some(1_024));
    assert_eq!(calls[1].returned_widths, vec![1_024]);
}

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
            let model = client.embedding_model_with_ndims("Qwen/Qwen4-Embedding-Unreleased", 8_192);
            assert_eq!(model.ndims(), 8_192);
            let error = model
                .embed_texts([PROBE.to_string()])
                .await
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
