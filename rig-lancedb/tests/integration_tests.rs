use serde_json::json;

use arrow_array::RecordBatchIterator;
use fixture::{as_record_batch, schema, words, Word};
use lancedb::index::vector::IvfPqIndexBuilder;
use rig::{
    embeddings::{EmbeddingModel, EmbeddingsBuilder},
    providers::openai::{self, Client},
    vector_store::VectorStoreIndex,
};
use rig_lancedb::{LanceDbVectorIndex, SearchParams};
use std::sync::Arc;

#[path = "./fixtures/lib.rs"]
mod fixture;

#[tokio::test]
async fn vector_search_test() {
    // Initialize OpenAI client. Use this to generate embeddings (and generate test data for RAG demo).
    let openai_client = Client::from_env();

    // Select an embedding model.
    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    // Initialize LanceDB locally.
    let db = lancedb::connect("data/lancedb-store")
        .execute()
        .await
        .unwrap();

    // Generate embeddings for the test data.
    let embeddings = EmbeddingsBuilder::new(model.clone())
        .documents(words()).unwrap()
        // Note: need at least 256 rows in order to create an index so copy the definition 256 times for testing purposes.
        .documents(
            (0..256)
                .map(|i| Word {
                    id: format!("doc{}", i),
                    definition: "Definition of *flumbuzzle (noun)*: A sudden, inexplicable urge to rearrange or reorganize small objects, such as desk items or books, for no apparent reason.".to_string()
                })
        ).unwrap()
        .build()
        .await.unwrap();

    let table = db
        .create_table(
            "words",
            RecordBatchIterator::new(
                vec![as_record_batch(embeddings, model.ndims())],
                Arc::new(schema(model.ndims())),
            ),
        )
        .execute()
        .await
        .unwrap();

    // See [LanceDB indexing](https://lancedb.github.io/lancedb/concepts/index_ivfpq/#product-quantization) for more information
    table
        .create_index(
            &["embedding"],
            lancedb::index::Index::IvfPq(IvfPqIndexBuilder::default()),
        )
        .execute()
        .await
        .unwrap();

    // Define search_params params that will be used by the vector store to perform the vector search.
    let search_params = SearchParams::default();
    let vector_store_index = LanceDbVectorIndex::new(table, model, "id", search_params)
        .await
        .unwrap();

    // Query the index
    let results = vector_store_index
        .top_n::<serde_json::Value>(
            "My boss says I zindle too much, what does that mean.unwrap()",
            1,
        )
        .await
        .unwrap();

    let (distance, _, value) = &results.first().unwrap();

    assert_eq!(
        *value,
        json!({
            "_distance": distance,
            "definition": "Definition of *zindle (verb)*: to pretend to be working on something important while actually doing something completely unrelated or unproductive.",
            "id": "doc1"
        })
    );

    db.drop_db().await.unwrap();
}
