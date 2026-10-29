use std::sync::Arc;

use arrow_array::{ArrayRef, FixedSizeListArray, RecordBatch, StringArray, types::Float64Type};
use rig_core::{
    driver::{Exchange, Local, Model, Opened, Opening, Step, Transport},
    embeddings::{Embedding, EmbeddingResponse},
    error::ProviderError,
    operation,
    wire::Capabilities,
};

use super::*;

#[derive(Clone)]
struct FixedEmbedding;

impl Transport<Local<operation::Embedding>> for FixedEmbedding {
    fn send(&self, texts: Vec<String>, _exchange: Exchange) -> Opening<Step<operation::Embedding>> {
        let response = EmbeddingResponse::new(
            texts
                .into_iter()
                .map(|document| Embedding {
                    document,
                    vec: vec![0.0, 0.0],
                })
                .collect(),
        );
        Opening::ready(Opened::new(futures::stream::iter([
            Ok::<_, ProviderError>(Step::End(response)),
        ])))
    }
}

#[tokio::test]
async fn top_n_ids_preserves_distance() -> anyhow::Result<()> {
    let path = std::env::temp_dir().join(format!(
        "rig-lancedb-distance-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    let db = lancedb::connect(path.to_string_lossy().as_ref())
        .execute()
        .await?;
    let batch = RecordBatch::try_from_iter(vec![
        ("id", Arc::new(StringArray::from(vec!["point"])) as ArrayRef),
        (
            "embedding",
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float64Type, _, _>(
                    vec![Some(vec![Some(3.0), Some(4.0)])],
                    2,
                ),
            ) as ArrayRef,
        ),
    ])?;
    let table = db.create_table("points", vec![batch]).execute().await?;
    let model = Model::new(
        Local::<operation::Embedding>::new("fixed")
            .with_capabilities(Capabilities::embedding(1, 2)),
        FixedEmbedding,
    );
    let index = LanceDbVectorIndex::new(table, model, "id", SearchParams::default()).await?;
    let request = || {
        VectorSearchRequest::builder()
            .query("query")
            .samples(1)
            .build()
    };
    let documents = index.top_n::<Value>(request()).await?;
    let ids = index.top_n_ids(request()).await?;
    drop(index);
    drop(db);
    std::fs::remove_dir_all(path)?;

    anyhow::ensure!(documents.len() == 1 && ids.len() == 1, "expected one match");
    let document = documents
        .first()
        .ok_or_else(|| anyhow::anyhow!("missing document"))?;
    let id = ids.first().ok_or_else(|| anyhow::anyhow!("missing id"))?;
    anyhow::ensure!(document.id == "point" && id.id == "point", "id changed");
    anyhow::ensure!(document.score > 0.0, "expected nonzero L2 distance");
    anyhow::ensure!(
        document.document.get("_distance").and_then(Value::as_f64) == Some(document.score),
        "top_n must expose the LanceDB _distance"
    );
    anyhow::ensure!(id.score == document.score, "top_n_ids lost the distance");
    Ok(())
}
