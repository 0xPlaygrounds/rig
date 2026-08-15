//! Cassette-backed coverage for Mistral's listing and embedding capabilities.
//!
//! Three defects live here, all found by comparing what the live API returns
//! and accepts against what rig models:
//!
//! - `EmbeddingsBuilder` chunked by the shared OpenAI cap of 1024 inputs;
//!   Mistral rejects anything over 256. The cap itself is pinned live here;
//!   that rig now chunks *below* it is a unit assertion, because recording a
//!   256-input success would commit ~5 MB of returned vectors to a fixture.
//! - Every Mistral embedding model reported `ndims() == 0`, because the
//!   default-dimension table consulted is OpenAI's and has no Mistral entry.
//! - Model listing dropped `description` and `max_context_length`, both of
//!   which `Model` has slots for.

use anyhow::Result;
use futures::StreamExt;
use rig::client::{CompletionClient, EmbeddingsClient, ModelListingClient};
use rig::completion::CompletionModel as _;
use rig::embeddings::EmbeddingModel as _;
use rig::providers::mistral;

use super::support::with_mistral_capability_cassette;

/// One more than Mistral's real per-request cap, so a single un-chunked
/// request would be rejected and only correct chunking can succeed.
const OVER_ONE_BATCH: usize = 257;

#[tokio::test]
async fn one_request_over_mistrals_batch_cap_is_rejected() -> Result<()> {
    with_mistral_capability_cassette(
        "capability_edges/one_request_over_mistrals_batch_cap_is_rejected",
        |client| async move {
            let model = client.embedding_model(mistral::embedding::MISTRAL_EMBED);
            // Straight through the model, bypassing the builder's chunking, so
            // the cell pins Mistral's own cap rather than rig's arithmetic.
            let error = model
                .embed_texts((0..OVER_ONE_BATCH).map(|i| format!("document {i}")))
                .await
                // `Vec<Embedding>` is not `Debug`; map the Ok side away.
                .map(|_| ())
                .expect_err("Mistral must reject a batch over its cap");
            assert_batch_cap_rejection(&error);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

#[tokio::test]
async fn mistral_embed_reports_its_real_dimensions() -> Result<()> {
    with_mistral_capability_cassette(
        "capability_edges/mistral_embed_reports_its_real_dimensions",
        |client| async move {
            let model = client.embedding_model(mistral::embedding::MISTRAL_EMBED);
            // The claim under test is the *declared* dimension; the live call
            // is what proves the declaration matches the vectors Mistral
            // actually returns.
            let declared = model.ndims();
            let embedding = model.embed_text("dimension probe").await?;
            assert_declared_matches_returned(declared, embedding.vec.len());
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

#[tokio::test]
async fn list_models_keeps_description_and_context_length() -> Result<()> {
    with_mistral_capability_cassette(
        "capability_edges/list_models_keeps_description_and_context_length",
        |client| async move {
            let models = client.list_models().await?;
            assert_listing_carries_mistrals_fields(&models.data);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

fn assert_batch_cap_rejection(error: &rig::embeddings::EmbeddingError) {
    let rendered = error.to_string();
    assert!(
        rendered.contains("Too many inputs"),
        "the rejection must be Mistral's batch-cap error, not some other failure: {rendered}"
    );
}

fn assert_declared_matches_returned(declared: usize, returned: usize) {
    assert_ne!(
        declared, 0,
        "a model that declares 0 dimensions cannot size a vector store; \
         Mistral's models are absent from OpenAI's dimension table"
    );
    assert_eq!(
        declared, returned,
        "the declared dimension must match the vector Mistral actually returns"
    );
}

fn assert_listing_carries_mistrals_fields(models: &[rig::model::Model]) {
    assert!(!models.is_empty(), "the listing must not be empty");
    assert!(
        models.iter().any(|model| model.description.is_some()),
        "Mistral's listing carries a `description` and `Model` has a slot for it"
    );
    assert!(
        models
            .iter()
            .any(|model| model.context_length.is_some_and(|length| length > 0)),
        "Mistral reports `max_context_length`, which is `Model::context_length`"
    );
}

/// `n > 1` streams as interleaved chunks distinguished only by
/// `choices[].index`. The blocking path answers such a request from candidate
/// 0 alone; the streamed twin used to concatenate every candidate's deltas
/// into one answer that is not any candidate the model produced.
///
/// Recorded against `temperature: 1` so the two candidates genuinely differ —
/// with identical candidates the merge would be invisible.
#[tokio::test]
async fn streaming_with_two_candidates_answers_from_the_first() -> Result<()> {
    with_mistral_capability_cassette(
        "capability_edges/streaming_with_two_candidates_answers_from_the_first",
        |client| async move {
            let model = client.completion_model(mistral::MISTRAL_SMALL);
            let mut stream: rig::streaming::StreamingCompletionResponse = model
                .completion_request("Say one random word.")
                .temperature(1.0)
                .max_tokens(8)
                .additional_params(serde_json::json!({"n": 2}))
                .stream()
                .await?;

            let mut text = String::new();
            while let Some(item) = stream.next().await {
                if let rig::streaming::StreamedAssistantContent::Text(chunk) = item? {
                    text.push_str(&chunk.text);
                }
            }
            assert_answers_from_candidate_zero(
                "capability_edges/streaming_with_two_candidates_answers_from_the_first",
                &text,
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// Assert the streamed text is candidate 0 exactly, derived from the
/// cassette's own bytes rather than hard-coded — the model picks different
/// words on every recording.
///
/// The cell's premise is checked too: if the recorded turn stopped carrying a
/// *second*, different candidate, a merge would be invisible and this cell
/// would pass while covering nothing.
fn assert_answers_from_candidate_zero(scenario: &str, streamed: &str) {
    let (first, second) = recorded_candidates(scenario);
    assert_ne!(
        first, second,
        "the recorded turn must carry two different candidates, or a merged stream would look \
         identical to a correct one"
    );
    assert_eq!(
        streamed.trim(),
        first,
        "the stream must deliver candidate 0 whole; interleaving the two yields neither"
    );
}

/// Concatenated `delta.content` per candidate index, read back out of the
/// recorded SSE body.
fn recorded_candidates(scenario: &str) -> (String, String) {
    let raw = std::fs::read_to_string(crate::cassettes::cassette_path("mistral", scenario))
        .expect("cassette should be readable");
    let mut candidates = [String::new(), String::new()];
    for line in raw.split("data: ").skip(1) {
        let payload = line.trim_start();
        let Some(end) = payload.rfind('}') else {
            continue;
        };
        let Ok(chunk) = serde_json::from_str::<serde_json::Value>(&payload[..=end]) else {
            continue;
        };
        let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) else {
            continue;
        };
        for choice in choices {
            let index = choice.get("index").and_then(serde_json::Value::as_u64);
            let text = choice
                .get("delta")
                .and_then(|delta| delta.get("content"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            if let Some(slot) = index.and_then(|i| candidates.get_mut(i as usize)) {
                slot.push_str(text);
            }
        }
    }
    let [first, second] = candidates;
    (first, second)
}
