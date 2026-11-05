use super::{HelixDB, HelixDBFilter, HelixDBVectorStore};
use rig_core::test_utils::MockEmbeddings;
use rig_core::vector_store::VectorStoreIndex;
use rig_core::vector_store::request::{SearchFilter, VectorSearchRequest};
use serde_json::json;

/// Both searches must keep the same candidates for one request: the metadata
/// filter applies to ids too, and no threshold means no similarity floor.
#[tokio::test]
async fn id_search_keeps_the_same_candidates_as_document_search() -> anyhow::Result<()> {
    // HelixDB scores are cosine distances; 1.5 is a similarity of -0.5.
    let hit = |id: &str, status: &str, distance: f64| {
        json!({
            "id": id,
            "score": distance,
            "doc": id,
            "json_payload": json!({ "status": status }).to_string(),
        })
    };
    let server = httpmock::MockServer::start_async().await;
    server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/VectorSearch");
            then.status(200).json_body(json!({ "vec_docs": [
                hit("near", "active", 0.1),
                hit("archived", "archived", 0.1),
                hit("opposite", "active", 1.5),
            ] }));
        })
        .await;
    let client = reqwest::Client::builder().no_proxy().build()?;
    let store = HelixDBVectorStore::new(
        HelixDB::with_client(Some(&server.base_url()), None, None, client),
        MockEmbeddings::model(),
    );
    let req = VectorSearchRequest::<HelixDBFilter>::builder()
        .query("document")
        .samples(3)
        .filter(HelixDBFilter::eq("status", json!("active")))
        .build();

    let documents = store.top_n::<serde_json::Value>(req.clone()).await?;
    let ids = store.top_n_ids(req).await?;
    let document_ids: Vec<_> = documents.iter().map(|hit| hit.id.as_str()).collect();
    let result_ids: Vec<_> = ids.iter().map(|hit| hit.id.as_str()).collect();
    anyhow::ensure!(document_ids == ["near", "opposite"], "{document_ids:?}");
    anyhow::ensure!(result_ids == document_ids, "{result_ids:?}");
    Ok(())
}
