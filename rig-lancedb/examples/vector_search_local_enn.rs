use std::sync::Arc;

use arrow_array::RecordBatchIterator;
use fixture::{as_record_batch, schema, words};
use rig::{
    embeddings::{EmbeddingModel, EmbeddingsBuilder},
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::VectorStoreIndexDyn,
};
use rig_lancedb::{LanceDbVectorIndex, SearchParams};

#[path = "./fixtures/lib.rs"]
mod fixture;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize OpenAI client. Use this to generate embeddings (and generate test data for RAG demo).
    let openai_client = Client::from_env();

    // Select the embedding model and generate our embeddings
    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Generate embeddings for the test data.
    let embeddings = EmbeddingsBuilder::new(model.clone())
        .documents(words())?
        .build()
        .await?;

    // Define search_params params that will be used by the vector store to perform the vector search.
    let search_params = SearchParams::default();

    // Initialize LanceDB locally.
    let db = lancedb::connect("data/lancedb-store").execute().await?;

    let table = if db
        .table_names()
        .execute()
        .await?
        .contains(&"definitions".to_string())
    {
        db.open_table("definitions").execute().await?
    } else {
        db.create_table(
            "definitions",
            RecordBatchIterator::new(
                vec![as_record_batch(embeddings, model.ndims())],
                Arc::new(schema(model.ndims())),
            ),
        )
        .execute()
        .await?
    };

    let vector_store = LanceDbVectorIndex::new(table, model, "id", search_params).await?;

    // Query the index
    let results = vector_store
        .top_n_ids("My boss says I zindle too much, what does that mean?", 1)
        .await?;

    println!("Results: {:?}", results);

    Ok(())
}
