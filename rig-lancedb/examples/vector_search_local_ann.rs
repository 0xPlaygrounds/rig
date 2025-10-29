use std::sync::Arc;

use arrow_array::RecordBatchIterator;
use fixture::{Word, as_record_batch, schema, words};
use lancedb::index::vector::IvfPqIndexBuilder;
use rig::client::EmbeddingsClient;
use rig::vector_store::request::VectorSearchRequest;
use rig::{
    embeddings::{EmbeddingModel, EmbeddingsBuilder},
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::VectorStoreIndex,
};
use rig_lancedb::{LanceDbVectorIndex, SearchParams};

#[path = "./fixtures/lib.rs"]
mod fixture;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize OpenAI client. Use this to generate embeddings (and generate test data for RAG demo).
    let openai_client = Client::from_env();

    // Select an embedding model.
    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Initialize LanceDB locally.
    let db = lancedb::connect("data/lancedb-store").execute().await?;

    // Generate embeddings for the test data.
    let embeddings = EmbeddingsBuilder::new(model.clone())
        .documents(words())?
        // Note: need at least 256 rows in order to create an index so copy the definition 256 times for testing purposes.
        .documents(
            (0..256)
                .map(|i| Word {
                    id: format!("doc{i}"),
                    definition: "Definition of *flumbuzzle (noun)*: A sudden, inexplicable urge to rearrange or reorganize small objects, such as desk items or books, for no apparent reason.".to_string()
                })
        )?
        .build()
        .await?;

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

    // See [LanceDB indexing](https://lancedb.github.io/lancedb/concepts/index_ivfpq/#product-quantization) for more information
    if table.index_stats("embedding").await?.is_none() {
        table
            .create_index(
                &["embedding"],
                lancedb::index::Index::IvfPq(IvfPqIndexBuilder::default()),
            )
            .execute()
            .await?;
    }

    // Define search_params params that will be used by the vector store to perform the vector search.
    let search_params = SearchParams::default();
    let vector_store_index = LanceDbVectorIndex::new(table, model, "id", search_params).await?;

    let query = "My boss says I zindle too much, what does that mean?";
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(1)
        .build()?;

    // Query the index
    let results = vector_store_index.top_n::<Word>(req).await?;

    println!("Results: {results:?}");

    Ok(())
}
