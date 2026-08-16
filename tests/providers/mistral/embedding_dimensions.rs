//! Cassette-backed coverage for what Mistral's embedding models declare as
//! their width.
//!
//! `EmbeddingModel::ndims()` is what sizes a vector store, and
//! `codestral-embed` declared **0** — the shared constructor falls back to
//! `default_ndims`, which #2337 populated for `mistral-embed` alone and pinned
//! at `None` for Codestral on the grounds that its width is "configurable".
//! It is configurable *and* defaulted: a request naming no dimension returns
//! 1536-wide vectors.
//!
//! ## Matrix
//!
//! The input space here is small and fully enumerable: {model} × {width the
//! caller asks for}. Mistral serves two embedding models, each under two ids
//! (a bare name and its dated alias), and `embedding_dimensions` sorts the
//! requested width into exactly three classes — none, the model's own default,
//! and some other value. Every combination is below; the ones that need no
//! traffic (a width rejected before the wire) are unit cells next to the
//! provider in `crates/rig-core/src/providers/mistral/embedding.rs`.
//!
//! | # | cell | model | requested width | status |
//! |---|---|---|---|---|
//! | 1 | `codestral_embed_declares_its_real_width` | `codestral-embed` | none | recorded — also pins that the request gains no width field |
//! | 2 | `codestral_embed_honors_an_explicit_output_dimension` | `codestral-embed` | 512 | recorded |
//! | 3 | `codestral_embed_dated_alias_declares_the_same_width` | `codestral-embed-2505` | none | recorded |
//! | 4 | `codestral_embed_batches_at_the_declared_width` | `codestral-embed` | none, 3 inputs | recorded |
//! | 5 | `mistral_embed_still_declares_its_own_width` | `mistral-embed` | none (**control**) | recorded |
//! | 6 | `codestral_embed_declares_its_default_width` | both codestral ids | — | unit |
//! | 7 | `codestral_embed_sends_its_width_as_output_dimension` | `codestral-embed` | 1536 / 512 / 3073 | unit — the default is echoed back as *no* field, and a width over the ceiling is rejected before any request is built |
//! | 8 | `mistral_embed_declares_its_width_without_requesting_it` | `mistral-embed` | 1024 / 512 | unit — ditto for the fixed-width model's `dimensions` rejection |
//!
//! Dropped with reason: a *batch* over Mistral's 256-input cap is already
//! recorded in `capability_edges.rs` and is about chunking rather than width;
//! recording it again here would duplicate a ~5 MB fixture for no new
//! coverage.

use anyhow::Result;
use rig::client::EmbeddingsClient;
use rig::embeddings::EmbeddingModel as _;
use rig::providers::mistral;

use super::support::with_mistral_embedding_cassette;

/// Codestral's documented default output width, and the width a
/// dimension-less request actually returns.
const CODESTRAL_DEFAULT_NDIMS: usize = 1536;

/// The bug: `ndims()` reported 0 while Mistral returned 1536-wide vectors.
#[tokio::test]
async fn codestral_embed_declares_its_real_width() -> Result<()> {
    with_mistral_embedding_cassette(
        "embedding_dimensions/codestral_embed_declares_its_real_width",
        |client| async move {
            let model = client.embedding_model(mistral::embedding::CODESTRAL_EMBED);
            // The claim under test is the *declared* width; the live call is
            // what proves the declaration matches the vectors Mistral returns.
            let declared = model.ndims();
            let embedding = model.embed_text("dimension probe").await?;

            anyhow::ensure!(
                declared != 0,
                "a model that declares 0 dimensions cannot size a vector store"
            );
            anyhow::ensure!(
                declared == embedding.vec.len(),
                "the declared width must match the vector Mistral actually returns: \
                 declared {declared}, returned {}",
                embedding.vec.len()
            );
            anyhow::ensure!(declared == CODESTRAL_DEFAULT_NDIMS, "declared {declared}");

            // Declaring a width must not start *requesting* one: a plain
            // `embedding_model(CODESTRAL_EMBED)` puts the same bytes on the
            // wire it always did.
            if let Some(request) =
                recorded_request("embedding_dimensions/codestral_embed_declares_its_real_width")
            {
                assert_names_no_width(&request)?;
            }
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// The declaration must not cost the caller their own choice: an explicit
/// width still rides Mistral's `output_dimension` spelling and still governs.
#[tokio::test]
async fn codestral_embed_honors_an_explicit_output_dimension() -> Result<()> {
    with_mistral_embedding_cassette(
        "embedding_dimensions/codestral_embed_honors_an_explicit_output_dimension",
        |client| async move {
            let model = client.embedding_model_with_ndims(mistral::embedding::CODESTRAL_EMBED, 512);
            let embedding = model.embed_text("dimension probe").await?;

            anyhow::ensure!(model.ndims() == 512, "declared {}", model.ndims());
            anyhow::ensure!(
                embedding.vec.len() == 512,
                "returned {}",
                embedding.vec.len()
            );

            if let Some(request) = recorded_request(
                "embedding_dimensions/codestral_embed_honors_an_explicit_output_dimension",
            ) {
                let body = request_json(&request)?;
                anyhow::ensure!(
                    body.get("output_dimension") == Some(&serde_json::json!(512)),
                    "Codestral takes its width under Mistral's own spelling: {request}"
                );
                anyhow::ensure!(
                    body.get("dimensions").is_none(),
                    "OpenAI's `dimensions` is not a field Mistral accepts: {request}"
                );
            }
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// The dated alias is the same model and must declare the same width — the
/// gap this fixes was a `matches!` arm that listed one id and not the other.
#[tokio::test]
async fn codestral_embed_dated_alias_declares_the_same_width() -> Result<()> {
    with_mistral_embedding_cassette(
        "embedding_dimensions/codestral_embed_dated_alias_declares_the_same_width",
        |client| async move {
            let model = client.embedding_model("codestral-embed-2505");
            let declared = model.ndims();
            let embedding = model.embed_text("dimension probe").await?;

            anyhow::ensure!(declared == CODESTRAL_DEFAULT_NDIMS, "declared {declared}");
            anyhow::ensure!(
                declared == embedding.vec.len(),
                "the dated alias must declare the width it returns: declared {declared}, \
                 returned {}",
                embedding.vec.len()
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// Several inputs in one request: every returned vector is the declared width,
/// so a store sized from `ndims()` fits the whole batch and not just its first
/// row.
#[tokio::test]
async fn codestral_embed_batches_at_the_declared_width() -> Result<()> {
    with_mistral_embedding_cassette(
        "embedding_dimensions/codestral_embed_batches_at_the_declared_width",
        |client| async move {
            let model = client.embedding_model(mistral::embedding::CODESTRAL_EMBED);
            let declared = model.ndims();
            let embeddings = model
                .embed_texts(["alpha".to_string(), "beta".to_string(), "gamma".to_string()])
                .await?;

            anyhow::ensure!(embeddings.len() == 3, "one vector per input");
            for embedding in &embeddings {
                anyhow::ensure!(
                    embedding.vec.len() == declared,
                    "every vector in the batch must be the declared width, got {}",
                    embedding.vec.len()
                );
            }
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// Control: the fixed-width model is untouched by the Codestral arm, and still
/// declares 1024 without asking for it.
#[tokio::test]
async fn mistral_embed_still_declares_its_own_width() -> Result<()> {
    with_mistral_embedding_cassette(
        "embedding_dimensions/mistral_embed_still_declares_its_own_width",
        |client| async move {
            let model = client.embedding_model(mistral::embedding::MISTRAL_EMBED);
            let declared = model.ndims();
            let embedding = model.embed_text("dimension probe").await?;

            anyhow::ensure!(declared == 1024, "declared {declared}");
            anyhow::ensure!(
                declared == embedding.vec.len(),
                "the fixed-width model must declare the width it returns: declared \
                 {declared}, returned {}",
                embedding.vec.len()
            );

            if let Some(request) =
                recorded_request("embedding_dimensions/mistral_embed_still_declares_its_own_width")
            {
                assert_names_no_width(&request)?;
            }
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// The recorded request body, parsed. Assertions read *keys*, not substrings:
/// the probe text itself contains the word "dimension", so a `contains` check
/// reports a width field that is not there — which is exactly how this cell
/// first failed.
fn request_json(request: &str) -> Result<serde_json::Map<String, serde_json::Value>> {
    let value: serde_json::Value = serde_json::from_str(request)
        .map_err(|error| anyhow::anyhow!("recorded request should be JSON: {error}"))?;
    value
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("recorded request should be a JSON object"))
}

/// A request that named no width must carry neither width field.
fn assert_names_no_width(request: &str) -> Result<()> {
    let body = request_json(request)?;
    anyhow::ensure!(
        body.get("output_dimension").is_none() && body.get("dimensions").is_none(),
        "a request that names no width must not gain one: {request}"
    );
    Ok(())
}

/// The recorded request body for a single-interaction scenario.
///
/// `None` while recording: the fixture is written when the cassette is
/// finished, i.e. *after* the test body returns. Every replay reads the real
/// thing.
fn recorded_request(scenario: &str) -> Option<String> {
    if crate::cassettes::CassetteMode::current() == crate::cassettes::CassetteMode::Record {
        return None;
    }
    let raw = std::fs::read_to_string(crate::cassettes::cassette_path("mistral", scenario))
        .expect("cassette should be readable");
    Some(
        raw.lines()
            .take_while(|line| *line != "then:")
            .find_map(|line| line.strip_prefix("  body: "))
            .map(|body| body.trim_matches('\'').replace("''", "'"))
            .expect("the recorded request must carry a body"),
    )
}
