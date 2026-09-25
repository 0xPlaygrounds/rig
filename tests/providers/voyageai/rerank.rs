//! VoyageAI reranking smoke test.

use rig::operation::RerankRequest;
use rig::providers::voyageai::{self, wire::VoyageAi};
use rig_test_support::endpoint::Endpoint;

#[tokio::test]
#[ignore = "requires VOYAGE_API_KEY"]
async fn rerank_smoke() {
    let provider = Endpoint::new(
        VoyageAi::from_env().expect("config should build from VOYAGE_API_KEY env var"),
        rig::rig_reqwest::bundled().expect("transport should build"),
    );
    let model = provider.rerank(voyageai::RERANK_2_5);

    let response = model
        .call(
            RerankRequest {
                query: "capital of France".to_owned(),
                documents: vec![
                    "Paris is the capital of France.".to_string(),
                    "Madrid is the capital of Spain.".to_string(),
                ],
            },
            None,
        )
        .await
        .expect("rerank request should succeed");

    assert!(
        !response.results.is_empty(),
        "should have at least one result"
    );
    assert!(
        response.results[0].relevance_score > 0.0,
        "top result should have positive relevance"
    );
    assert!(
        response.results[0].index == 0,
        "Paris should be the top result"
    );
    assert!(
        response.usage.total_tokens.is_some_and(|n| n > 0),
        "usage should be positive"
    );
    assert!(
        response
            .model
            .as_deref()
            .is_some_and(|model| !model.is_empty()),
        "model name should be present"
    );
    assert_eq!(response.provider, "voyageai");
}
