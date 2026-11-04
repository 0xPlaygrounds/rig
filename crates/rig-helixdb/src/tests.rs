use super::{HelixDBFilter, QueryResult};
use rig_core::vector_store::request::{SearchFilter, VectorSearchRequest};
use serde_json::json;

/// The fixture isolates client-side filtering of the same HTTP candidates;
/// it does not claim to record HelixDB's ranking behavior.
#[tokio::test]
async fn public_searches_apply_the_same_metadata_filter() -> anyhow::Result<()> {
    use super::{HelixDB, HelixDBVectorStore};
    use rig_core::{test_utils::MockEmbeddings, vector_store::VectorStoreIndex};

    let server = httpmock::MockServer::start_async().await;
    let candidates = [hit("active", 0.1), hit("archived", 0.1)];
    let mock = server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/VectorSearch");
            then.status(200).json_body(json!({"vec_docs": candidates}));
        })
        .await;
    let client = reqwest::Client::builder().no_proxy().build()?;
    let store = HelixDBVectorStore::new(
        HelixDB::with_client(Some(&server.base_url()), None, None, client),
        MockEmbeddings::model(),
    );
    let request = VectorSearchRequest::<HelixDBFilter>::builder()
        .query("document")
        .samples(2)
        .threshold(0.5);
    let unfiltered = store.top_n_ids(request.clone().build()).await?;
    anyhow::ensure!(unfiltered.len() == 2);
    let filtered = request
        .filter(HelixDBFilter::eq("status", json!("active")))
        .build();
    let documents = store.top_n::<serde_json::Value>(filtered.clone()).await?;
    let ids = store.top_n_ids(filtered).await?;
    let document_ids: Vec<_> = documents.iter().map(|hit| hit.id.as_str()).collect();
    let result_ids: Vec<_> = ids.iter().map(|hit| hit.id.as_str()).collect();
    anyhow::ensure!(document_ids == ["active"]);
    anyhow::ensure!(result_ids == document_ids);
    anyhow::ensure!(mock.calls_async().await == 3);
    Ok(())
}

fn hit(status: &str, distance: f64) -> QueryResult {
    QueryResult {
        id: status.to_owned(),
        score: distance,
        doc: "document".to_owned(),
        json_payload: json!({"status": status}).to_string(),
    }
}

#[test]
fn candidates_must_satisfy_both_metadata_and_similarity() -> anyhow::Result<()> {
    let req = VectorSearchRequest::<HelixDBFilter>::builder()
        .query("document")
        .samples(3)
        .threshold(0.5)
        .filter(HelixDBFilter::eq("status", json!("active")))
        .build();
    anyhow::ensure!(hit("active", 0.1).matches(&req)?);
    anyhow::ensure!(!hit("archived", 0.1).matches(&req)?);
    anyhow::ensure!(!hit("active", 0.6).matches(&req)?);
    Ok(())
}

#[test]
fn compound_metadata_filters_are_applied() -> anyhow::Result<()> {
    let req = VectorSearchRequest::<HelixDBFilter>::builder()
        .query("document")
        .samples(3)
        .filter(
            HelixDBFilter::eq("status", json!("active"))
                .or(HelixDBFilter::eq("status", json!("pending"))),
        )
        .build();
    anyhow::ensure!(hit("active", 0.1).matches(&req)?);
    anyhow::ensure!(hit("pending", 0.1).matches(&req)?);
    anyhow::ensure!(!hit("archived", 0.1).matches(&req)?);
    Ok(())
}

#[test]
fn filtering_cannot_admit_a_payload_it_cannot_decode() -> anyhow::Result<()> {
    let mut candidate = hit("active", 0.1);
    candidate.json_payload = "invalid JSON".to_owned();
    let request = VectorSearchRequest::<HelixDBFilter>::builder()
        .query("document")
        .samples(1);
    anyhow::ensure!(candidate.matches(&request.clone().build())?);
    let filtered = request
        .filter(HelixDBFilter::eq("status", json!("active")))
        .build();
    anyhow::ensure!(candidate.matches(&filtered).is_err());
    Ok(())
}

#[test]
fn absent_threshold_does_not_exclude_negative_similarity() -> anyhow::Result<()> {
    let request = VectorSearchRequest::<HelixDBFilter>::builder()
        .query("document")
        .samples(1);
    anyhow::ensure!(hit("active", 1.1).matches(&request.clone().build())?);
    anyhow::ensure!(!hit("active", 1.1).matches(&request.threshold(0.0).build())?);
    Ok(())
}
