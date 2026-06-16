//! VoyageAI reranking smoke test.

use rig::client::{ProviderClient, RerankingClient};
use rig::providers::voyageai;
use rig::rerank::RerankModel;

#[tokio::test]
#[ignore = "requires VOYAGE_API_KEY"]
async fn rerank_smoke() {
    let client =
        voyageai::Client::from_env().expect("client should build from VOYAGE_API_KEY env var");
    let model = client.rerank_model(voyageai::RERANK_2_5);

    let response = model
        .rerank(
            "capital of France",
            vec![
                "Paris is the capital of France.".to_string(),
                "Madrid is the capital of Spain.".to_string(),
            ],
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
    assert!(response.usage.total_tokens > 0, "usage should be positive");
    assert!(!response.model.is_empty(), "model name should be present");
}
