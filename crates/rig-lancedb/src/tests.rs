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

/// Embeds every text to the origin.
#[derive(Clone)]
struct OriginEmbedding;

impl Transport<Local<operation::Embedding>> for OriginEmbedding {
    fn send(&self, texts: Vec<String>, _exchange: Exchange) -> Opening<Step<operation::Embedding>> {
        let embeddings = texts
            .into_iter()
            .map(|document| Embedding {
                document,
                vec: vec![0.0, 0.0],
            })
            .collect();
        Opening::ready(Opened::new(futures::stream::iter([
            Ok::<_, ProviderError>(Step::End(EmbeddingResponse::new(embeddings))),
        ])))
    }
}

#[tokio::test]
async fn top_n_ids_returns_the_same_distance_as_top_n() -> anyhow::Result<()> {
    let db = lancedb::connect("memory://").execute().await?;
    let batch = RecordBatch::try_from_iter([
        ("id", Arc::new(StringArray::from(vec!["point"])) as ArrayRef),
        (
            "embedding",
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float64Type, _, _>(
                    [Some([Some(3.0), Some(4.0)])],
                    2,
                ),
            ) as ArrayRef,
        ),
    ])?;
    let table = db.create_table("points", vec![batch]).execute().await?;
    let model = Model::new(
        Local::<operation::Embedding>::new("origin")
            .with_capabilities(Capabilities::embedding(1, 2)),
        OriginEmbedding,
    );
    let index = LanceDbVectorIndex::new(table, model, "id", SearchParams::default()).await?;
    let request = || VectorSearchRequest::builder().query("q").samples(1).build();

    let documents = index.top_n::<Value>(request()).await?;
    let ids = index.top_n_ids(request()).await?;
    let document_scores = documents.iter().map(|r| r.score).collect::<Vec<_>>();
    let id_scores = ids.iter().map(|r| r.score).collect::<Vec<_>>();
    anyhow::ensure!(
        document_scores.len() == 1 && document_scores.iter().all(|score| *score > 0.0),
        "top_n should return one nonzero distance, got {document_scores:?}"
    );
    anyhow::ensure!(
        id_scores == document_scores,
        "top_n_ids distances {id_scores:?} differ from top_n's {document_scores:?}"
    );
    Ok(())
}
