//! VoyageAI reranking smoke test.

use rig::http_runtime::HttpRuntime;
use rig::providers::voyageai;

#[tokio::test]
#[ignore = "requires VOYAGE_API_KEY"]
async fn rerank_smoke() {
    let cfg = voyageai::functions::RerankConfig::from_env(voyageai::RERANK_2_5)
        .expect("config should build from VOYAGE_API_KEY env var");
    let rt = HttpRuntime::new();

    let response = voyageai::functions::rerank(
        &cfg,
        &rt,
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
