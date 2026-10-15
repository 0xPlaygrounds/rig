//! Loads a CSV file one document per row, inserts the embedded rows into a new
//! LanceDB table, and searches it. Needs `OPENAI_API_KEY`.

use std::sync::Arc;

use lancedb::{
    arrow::arrow_schema::{DataType, Field, Schema},
    database::CreateTableMode,
};
use rig_core::{
    Embed,
    embeddings::EmbeddingsBuilder,
    loaders::CsvFileLoader,
    providers::openai::{self, OpenAI},
    vector_store::{InsertDocuments, VectorStoreIndex, request::VectorSearchRequest},
};
use rig_lancedb::{LanceDbVectorIndex, SearchParams};
use serde::{Deserialize, Serialize};

/// One CSV row, rendered as `header: value` lines.
#[derive(Embed, Serialize, Deserialize, Debug)]
struct Row {
    id: String,
    #[embed]
    text: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let csv = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/fixtures/definitions.csv"
    );
    let rows = CsvFileLoader::with_glob(csv)?
        .rows()
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(i, text)| Row {
            id: format!("row{i}"),
            text,
        });

    let model = OpenAI::from_env()?
        .embedding(openai::TEXT_EMBEDDING_ADA_002, None)
        .erase();
    let documents = EmbeddingsBuilder::new(model.clone())
        .documents(rows)?
        .build()
        .await?;

    let item = Arc::new(Field::new("item", DataType::Float64, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(item, model.capabilities().ndims as i32),
            false,
        ),
    ]));

    let db = lancedb::connect("data/lancedb-store").execute().await?;
    let table = db
        .create_empty_table("csv_definitions", schema)
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await?;

    let index = LanceDbVectorIndex::new(table, model, "id", SearchParams::default()).await?;
    index.insert_documents(documents).await?;

    let req = VectorSearchRequest::builder()
        .query("My boss says I zindle too much, what does that mean?")
        .samples(1)
        .build();
    for result in index.top_n::<Row>(req).await? {
        println!(
            "{:.4} {}\n{}",
            result.score, result.id, result.document.text
        );
    }

    Ok(())
}
