use std::sync::Arc;

use anyhow::Context;
use arrow_array::{ArrayRef, FixedSizeListArray, RecordBatch, StringArray, types::Float64Type};
use assert_fs::TempDir;
use lancedb::arrow::arrow_schema::{DataType, Field, Schema};
use rig_core::{
    driver::{Exchange, Local, Model, Opened, Opening, Step, Transport},
    embeddings::{Embedding, EmbeddingResponse},
    error::ProviderError,
    operation,
    vector_store::InsertDocuments,
    wire::Capabilities,
};
use serde_json::json;

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

/// An index over an empty on-disk table whose `embedding` column holds two `item`
/// floats, with optional number and binary columns that documents may leave out.
async fn index_on_disk(
    dir: &TempDir,
    item: DataType,
    search_params: SearchParams,
) -> anyhow::Result<LanceDbVectorIndex> {
    let path = dir.path().to_str().context("temp dir path is not UTF-8")?;
    let db = lancedb::connect(path).execute().await?;
    let list = Arc::new(Field::new_list_field(item, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, true),
        Field::new("definition", DataType::Utf8, true),
        Field::new("embedding", DataType::FixedSizeList(list, 2), false),
        Field::new("count", DataType::Int64, true),
        Field::new("score", DataType::Float32, true),
        Field::new("blob", DataType::FixedSizeBinary(2), true),
        Field::new_list("blobs", Field::new_list_field(DataType::Binary, true), true),
    ]));
    let table = db.create_empty_table("words", schema).execute().await?;
    let model = Model::new(
        Local::<operation::Embedding>::new("origin")
            .with_capabilities(Capabilities::embedding(1, 2)),
        OriginEmbedding,
    );
    Ok(LanceDbVectorIndex::new(table, model, "id", search_params).await?)
}

fn embedding(vec: &[f64]) -> Embedding {
    Embedding {
        document: "text".to_string(),
        vec: vec.to_vec(),
    }
}

#[tokio::test]
async fn inserted_documents_are_found_by_search() -> anyhow::Result<()> {
    for item in [DataType::Float32, DataType::Float64] {
        let dir = TempDir::new()?;
        let index = index_on_disk(&dir, item, SearchParams::default()).await?;
        index
            .insert_documents(vec![
                (
                    json!({"id": "far", "definition": "far away"}),
                    vec![embedding(&[3.0, 4.0])],
                ),
                (
                    json!({"id": "near", "definition": "close by", "blob": "0a0b"}),
                    vec![embedding(&[1.0, 0.0]), embedding(&[0.0, 2.0])],
                ),
            ])
            .await?;

        let request = || VectorSearchRequest::builder().query("q").samples(3).build();
        let found = index.top_n::<Value>(request()).await?;
        let found_ids = found.iter().map(|r| r.id.as_str()).collect::<Vec<_>>();
        anyhow::ensure!(
            found_ids == ["near", "near", "far"],
            "top_n found {found_ids:?}"
        );
        let definition = found.first().and_then(|r| r.document.get("definition"));
        anyhow::ensure!(
            definition == Some(&json!("close by")),
            "top_n read back {definition:?}"
        );
        let ids = index.top_n_ids(request()).await?;
        let ids = ids.iter().map(|r| r.id.as_str()).collect::<Vec<_>>();
        anyhow::ensure!(ids == found_ids, "top_n_ids found {ids:?}");
    }
    Ok(())
}

#[tokio::test]
async fn insert_refuses_documents_the_table_cannot_hold() -> anyhow::Result<()> {
    let dir = TempDir::new()?;
    let index = index_on_disk(&dir, DataType::Float32, SearchParams::default()).await?;
    let fits = || vec![embedding(&[1.0, 0.0])];
    let not_finite = "\"a\" has a value that is not finite";
    let refused = [
        (json!({"definition": "no id"}), fits(), "`id`"),
        (json!({"id": null}), fits(), "`id`"),
        (json!("text"), fits(), "not a JSON object"),
        (json!({"id": "a", "embedding": "mine"}), fits(), "embedding"),
        (json!({"id": "a"}), vec![], "no embeddings"),
        (json!({"id": "a", "blob": [1, 2]}), fits(), "`blob`"),
        (json!({"id": "a", "blobs": [[1]]}), fits(), "`blobs`"),
        (
            json!({"id": "a", "count": 1.5}),
            fits(),
            "\"a\" field `count`",
        ),
        (
            json!({"id": "a", "score": 1e300}),
            fits(),
            "\"a\" field `score`",
        ),
        (
            json!({"id": "a", "blob": "abc"}),
            fits(),
            "\"a\" field `blob`",
        ),
        (
            json!({"id": "a", "blobs": ["abc"]}),
            fits(),
            "\"a\" field `blobs`",
        ),
        (
            json!({"id": "a"}),
            vec![embedding(&[f64::NAN, 0.0])],
            not_finite,
        ),
        (
            json!({"id": "a"}),
            vec![embedding(&[f64::INFINITY, 0.0])],
            not_finite,
        ),
        (
            json!({"id": "a"}),
            vec![embedding(&[1e300, 0.0])],
            not_finite,
        ),
    ];
    for (document, embeddings, reason) in refused {
        let outcome = index
            .insert_documents(vec![
                (json!({"id": "fine"}), fits()),
                (document.clone(), embeddings),
            ])
            .await;
        let message = outcome.err().map(|e| e.to_string()).unwrap_or_default();
        anyhow::ensure!(
            message.contains(reason),
            "{document} should be refused for {reason}, got {message:?}"
        );
    }
    let rows = index.table.count_rows(None).await?;
    anyhow::ensure!(rows == 0, "refused inserts wrote {rows} rows");

    let dir = TempDir::new()?;
    let params = SearchParams::default().column("definition");
    let index = index_on_disk(&dir, DataType::Float32, params).await?;
    let outcome = index
        .insert_documents(vec![(json!({"id": "a"}), fits())])
        .await;
    let message = outcome.err().map(|e| e.to_string()).unwrap_or_default();
    anyhow::ensure!(
        message.contains("definition"),
        "a non-vector search column should be refused, got {message:?}"
    );

    // 1e300 overflows only a Float32 column.
    let dir = TempDir::new()?;
    let index = index_on_disk(&dir, DataType::Float64, SearchParams::default()).await?;
    index
        .insert_documents(vec![(json!({"id": "a"}), vec![embedding(&[1e300, 0.0])])])
        .await?;
    Ok(())
}
