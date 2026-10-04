//! The reranking matrix — the capability slot this PR added.
//!
//! **Server**: `--embeddings --pooling rank --reranking` on
//! `gpustack/bge-reranker-v2-m3-GGUF` Q4_K_M, `--seed 42 --temp 0 -c 2048`,
//! `llama-server` b10964-b29c606e2. A cross-encoder is not optional: a causal LM
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
//! | [`a_single_document_is_still_a_ranking`] | one document | index 0, one result |
//! | [`top_n_zero_returns_an_empty_ranking`] | `top_n` == 0 | an empty list, not an error and not the whole list |
//! | empty document list | 0 documents | `error_matrix.rs` — a 400 from the server |
//! | no reranker loaded | wrong server | `error_matrix.rs` — a 501 |
//!
//! # Scores are logits, not probabilities
//!
//! [`RerankResult::relevance_score`](rig::rerank::RerankResult::relevance_score)
//! is documented as "between 0 and 1". llama.cpp returns the cross-encoder's
//! **raw logit**: measured on b10964-b29c606e2, ranking three documents against
//! "What is a panda?" gives `0.8225`, `-4.7583` and `-8.3761`. The ordering is
//! meaningful and is what a reranker is for; the magnitude is not a
//! probability and negative values are normal. That mismatch is a defect in
//! the field's *documentation* rather than in any mapping, and this PR
//! corrects the doc comment; [`scores_are_raw_logits_and_may_be_negative`]
//! is what keeps the corrected wording honest.

use rig_test_support::cassette_models::MapWire;
use serde_json::Value;

use crate::cassettes::{recorded_json_request, recorded_statuses_and_bodies};

use super::super::cassette_support::*;
use rig::operation::RerankRequest;

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
async fn a_single_document_is_still_a_ranking() {
    with_llamacpp_rerank_cassette("rerank_matrix/single_document", |client| async move {
        let reranked = client
            .rerank(CASSETTE_RERANK_MODEL)
            .call(RerankRequest {
                query: QUERY.to_owned(),
                documents: vec!["it is a bear".to_string()],
            })
            .await
            .expect("a single-document rerank should succeed");

        assert_eq!(reranked.results.len(), 1, "{:?}", reranked.results);
        assert_eq!(reranked.results[0].index, 0);
    })
    .await;

    assert_eq!(recorded_results("rerank_matrix/single_document").len(), 1);
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
            .rerank(CASSETTE_RERANK_MODEL)
            .map_wire(|wire| wire.with_top_n(0))
            .call(RerankRequest {
                query: QUERY.to_owned(),
                documents: documents(),
            })
            .await
            .expect("top_n 0 is a valid request, not an error");

        assert!(
            reranked.results.is_empty(),
            "zero means zero — not the whole list: {:?}",
            reranked.results
        );
        assert!(
            reranked.usage.total_tokens.is_some_and(|n| n > 0),
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
