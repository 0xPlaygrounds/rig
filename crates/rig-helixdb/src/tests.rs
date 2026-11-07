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

/// A redirect to the same origin keeps the API key; one to another origin
/// (here another port) must not carry it there.
#[tokio::test]
async fn api_key_stays_on_the_redirecting_origin() -> anyhow::Result<()> {
    let origin = httpmock::MockServer::start_async().await;
    let other = httpmock::MockServer::start_async().await;
    origin
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/VectorSearch");
            then.status(307).header("location", "/hop");
        })
        .await;
    let hop = origin
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/hop")
                .header("x-api-key", "secret");
            then.status(307)
                .header("location", other.url("/VectorSearch"));
        })
        .await;
    let leaked = other
        .mock_async(|when, then| {
            when.header_exists("x-api-key");
            then.status(200).json_body(json!({}));
        })
        .await;

    let client = HelixDB::new(Some(&origin.base_url()), None, Some("secret"));
    let _ = client
        .query::<_, serde_json::Value>("VectorSearch", &json!({}))
        .await;

    anyhow::ensure!(
        hop.calls_async().await == 1,
        "the same-origin redirect lost the API key"
    );
    anyhow::ensure!(
        leaked.calls_async().await == 0,
        "x-api-key reached another origin"
    );
    Ok(())
}
