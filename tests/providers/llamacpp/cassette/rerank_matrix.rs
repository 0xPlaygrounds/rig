//! The reranking matrix — the capability slot this PR added.
//!
//! **Server**: `--embeddings --pooling rank --reranking` on
//! `gpustack/bge-reranker-v2-m3-GGUF` Q4_K_M, `--seed 42 --temp 0 -c 2048`,
//! `llama-server` b10499-6d05498. A cross-encoder is not optional: a causal LM
//! has no rank pooling head and `llama-server` refuses to start with
//! `--pooling rank` at all, so there is no "rerank with the wrong model"
//! degraded path to record.
//!
//! Rig had a [`RerankModel`](rig::rerank::RerankModel) trait and exactly one
//! implementation of it — Voyage AI's, written against Voyage's own wire. This
//! PR adds the shared Jina-shaped driver llama.cpp speaks
//! (`providers::internal::rerank`), so the next provider on that wire declares
//! a slot instead of copying a request builder.
//!
//! | Cell | Dimension | Pinned |
//! | --- | --- | --- |
//! | [`multiple_documents_come_back_ranked`] | many documents | ordered by score, indices point back into the input |
//! | [`a_single_document_is_still_a_ranking`] | one document | index 0, one result |
//! | [`top_n_beyond_the_document_count_is_clamped`] | `top_n` > len | clamped, not an error |
//! | [`top_n_below_the_document_count_truncates`] | `top_n` < len | the highest-scoring `n` |
//! | [`top_n_zero_returns_an_empty_ranking`] | `top_n` == 0 | an empty list, not an error and not the whole list |
//! | empty document list | 0 documents | `error_matrix.rs` — a 400 from the server |
//! | no reranker loaded | wrong server | `error_matrix.rs` — a 501 |
//!
//! # Scores are logits, not probabilities
//!
//! [`RerankResult::relevance_score`](rig::rerank::RerankResult::relevance_score)
//! is documented as "between 0 and 1". llama.cpp returns the cross-encoder's
//! **raw logit**: measured on b10499-6d05498, ranking three documents against
//! "What is a panda?" gives `0.8225`, `-4.7583` and `-8.3761`. The ordering is
//! meaningful and is what a reranker is for; the magnitude is not a
//! probability and negative values are normal. That mismatch is a defect in
//! the field's *documentation* rather than in any mapping, and this PR
//! corrects the doc comment; [`scores_are_raw_logits_and_may_be_negative`]
//! is what keeps the corrected wording honest.

use rig::client::RerankingClient;
use rig::rerank::RerankModel as _;
use serde_json::Value;

use crate::cassettes::{
    recorded_json_request, recorded_request_paths, recorded_statuses_and_bodies,
};

use super::super::cassette_support::*;

/// Three documents whose relevance to the query is unambiguous, so an
/// assertion on the *ordering* is a real assertion rather than a coin flip.
fn documents() -> Vec<String> {
    vec![
        "hi".to_string(),
        "it is a bear".to_string(),
        "The giant panda (Ailuropoda melanoleuca) is a bear species endemic to China.".to_string(),
    ]
}

const QUERY: &str = "What is a panda?";

fn recorded_results(scenario: &str) -> Vec<Value> {
    let recorded = recorded_statuses_and_bodies("llamacpp", scenario);
    let (status, body) = recorded.last().expect("an interaction");
    assert_eq!(*status, 200, "{scenario}: {body}");
    let response: Value = serde_json::from_str(body).expect("response should be JSON");
    assert_eq!(
        response["object"],
        serde_json::json!("list"),
        "{scenario}: the Jina-shaped envelope: {response}"
    );
    response["results"]
        .as_array()
        .unwrap_or_else(|| panic!("{scenario}: results array: {response}"))
        .clone()
}

#[tokio::test]
async fn multiple_documents_come_back_ranked() {
    with_llamacpp_rerank_cassette("rerank_matrix/multiple_documents", |client| async move {
        let reranked = client
            .rerank_model(CASSETTE_RERANK_MODEL)
            .rerank(QUERY, documents())
            .await
            .expect("a multi-document rerank should succeed");

        assert_eq!(reranked.results.len(), 3, "{:?}", reranked.results);
        assert_eq!(
            reranked.results[0].index, 2,
            "the sentence that actually defines a panda must rank first: {:?}",
            reranked.results
        );
        // Descending by score, which is what "reranked" means.
        for pair in reranked.results.windows(2) {
            assert!(
                pair[0].relevance_score >= pair[1].relevance_score,
                "results must be ordered by score: {:?}",
                reranked.results
            );
        }
        // Every index points back into the input list, since llama.cpp never
        // echoes the document text.
        let mut indices = reranked.results.iter().map(|r| r.index).collect::<Vec<_>>();
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2]);
        assert!(
            reranked.results.iter().all(|r| r.document.is_none()),
            "llama.cpp has no `return_documents` on this path: {:?}",
            reranked.results
        );

        assert!(
            reranked.usage.total_tokens > 0,
            "the server bills the ranking: {:?}",
            reranked.usage
        );
        assert_eq!(reranked.model, CASSETTE_RERANK_MODEL);
    })
    .await;

    assert_eq!(
        recorded_request_paths("llamacpp", "rerank_matrix/multiple_documents"),
        vec!["/v1/rerank".to_string()],
        "the `/v1` prefix is this provider's business"
    );
    let request = recorded_json_request("llamacpp", "rerank_matrix/multiple_documents");
    assert_eq!(request["query"], serde_json::json!(QUERY));
    assert_eq!(
        request["documents"].as_array().map(Vec::len),
        Some(3),
        "the documents must reach the wire: {request}"
    );
    assert!(
        request.get("top_n").is_none(),
        "no top_n unless the caller asked: {request}"
    );
    assert_eq!(
        recorded_results("rerank_matrix/multiple_documents").len(),
        3
    );
}

/// Scores are the cross-encoder's raw logits.
#[tokio::test]
async fn scores_are_raw_logits_and_may_be_negative() {
    with_llamacpp_rerank_cassette("rerank_matrix/negative_scores", |client| async move {
        let reranked = client
            .rerank_model(CASSETTE_RERANK_MODEL)
            .rerank(QUERY, documents())
            .await
            .expect("rerank should succeed");

        assert!(
            reranked
                .results
                .iter()
                .any(|result| result.relevance_score < 0.0),
            "llama.cpp returns logits, so an irrelevant document scores below \
             zero — the trait's `relevance_score` doc says 0..1 and this is why \
             that wording was corrected: {:?}",
            reranked.results
        );
        assert!(
            reranked
                .results
                .iter()
                .any(|result| result.relevance_score > 0.0),
            "and the relevant one scores above it: {:?}",
            reranked.results
        );
    })
    .await;

    let results = recorded_results("rerank_matrix/negative_scores");
    assert!(
        results
            .iter()
            .any(|result| result["relevance_score"].as_f64().unwrap_or_default() < 0.0),
        "the wire itself carries a negative score: {results:?}"
    );
}

#[tokio::test]
async fn a_single_document_is_still_a_ranking() {
    with_llamacpp_rerank_cassette("rerank_matrix/single_document", |client| async move {
        let reranked = client
            .rerank_model(CASSETTE_RERANK_MODEL)
            .rerank(QUERY, vec!["it is a bear".to_string()])
            .await
            .expect("a single-document rerank should succeed");

        assert_eq!(reranked.results.len(), 1, "{:?}", reranked.results);
        assert_eq!(reranked.results[0].index, 0);
    })
    .await;

    assert_eq!(recorded_results("rerank_matrix/single_document").len(), 1);
}

/// `top_n` larger than the document count is clamped rather than refused.
///
/// llama.cpp does `elements.resize(std::min(top_n, elements.size()))`, so an
/// over-large `top_n` is silently the whole list. Worth pinning because the
/// obvious alternative — a 400 — is what several hosted rerankers do, and a
/// caller who guards against that guard is writing dead code here.
#[tokio::test]
async fn top_n_beyond_the_document_count_is_clamped() {
    with_llamacpp_rerank_cassette("rerank_matrix/top_n_beyond_count", |client| async move {
        let reranked = client
            .rerank_model(CASSETTE_RERANK_MODEL)
            .top_n(99)
            .rerank(QUERY, documents())
            .await
            .expect("an over-large top_n is clamped, not refused");

        assert_eq!(
            reranked.results.len(),
            3,
            "clamped to the document count: {:?}",
            reranked.results
        );
    })
    .await;

    let request = recorded_json_request("llamacpp", "rerank_matrix/top_n_beyond_count");
    assert_eq!(
        request["top_n"],
        serde_json::json!(99),
        "the request really did ask for more than it sent: {request}"
    );
    let recorded = recorded_statuses_and_bodies("llamacpp", "rerank_matrix/top_n_beyond_count");
    assert_eq!(recorded[0].0, 200, "not an error");
    assert_eq!(
        recorded_results("rerank_matrix/top_n_beyond_count").len(),
        3
    );
}

#[tokio::test]
async fn top_n_below_the_document_count_truncates() {
    with_llamacpp_rerank_cassette("rerank_matrix/top_n_truncates", |client| async move {
        let reranked = client
            .rerank_model(CASSETTE_RERANK_MODEL)
            .top_n(1)
            .rerank(QUERY, documents())
            .await
            .expect("a truncating top_n should succeed");

        assert_eq!(reranked.results.len(), 1, "{:?}", reranked.results);
        assert_eq!(
            reranked.results[0].index, 2,
            "truncation keeps the *highest scoring* document, not the first: {:?}",
            reranked.results
        );
    })
    .await;

    assert_eq!(
        recorded_json_request("llamacpp", "rerank_matrix/top_n_truncates")["top_n"],
        serde_json::json!(1)
    );
    assert_eq!(recorded_results("rerank_matrix/top_n_truncates").len(), 1);
}

/// `top_n: 0` is an empty ranking, not an error and not "all of them".
///
/// The third arm of the `top_n` dimension, and the one where a clamp
/// implemented as `min(top_n, len)` could plausibly have gone the other way —
/// treating 0 as "unset" and returning everything. It does not:
/// `elements.resize(0)` is exactly what it says. Worth pinning because rig's
/// driver passes the value straight through, so whatever llama.cpp decides is
/// what the caller gets.
#[tokio::test]
async fn top_n_zero_returns_an_empty_ranking() {
    with_llamacpp_rerank_cassette("rerank_matrix/top_n_zero", |client| async move {
        let reranked = client
            .rerank_model(CASSETTE_RERANK_MODEL)
            .top_n(0)
            .rerank(QUERY, documents())
            .await
            .expect("top_n 0 is a valid request, not an error");

        assert!(
            reranked.results.is_empty(),
            "zero means zero — not the whole list: {:?}",
            reranked.results
        );
        assert!(
            reranked.usage.total_tokens > 0,
            "the documents were still scored and still billed: {:?}",
            reranked.usage
        );
    })
    .await;

    let request = recorded_json_request("llamacpp", "rerank_matrix/top_n_zero");
    assert_eq!(
        request["top_n"],
        serde_json::json!(0),
        "0 must reach the wire rather than being dropped as falsy: {request}"
    );
    assert_eq!(
        request["documents"].as_array().map(Vec::len),
        Some(3),
        "and all three documents were sent"
    );
    assert!(
        recorded_results("rerank_matrix/top_n_zero").is_empty(),
        "the wire itself returned nothing"
    );
}
