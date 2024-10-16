use std::{env, sync::Arc};

use arrow_array::RecordBatchIterator;
use fixture::{as_record_batch, schema};
use lancedb::{index::vector::IvfPqIndexBuilder, DistanceType};
use rig::{
    embeddings::{builder::EmbeddingsBuilder, embedding::EmbeddingModel},
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::VectorStoreIndex,
};
use rig_lancedb::{LanceDbVectorStore, SearchParams};
use serde::Deserialize;

#[path = "./fixtures/lib.rs"]
mod fixture;

#[derive(Deserialize, Debug)]
pub struct VectorSearchResult {
    pub id: String,
    pub content: String,
}

// Note: see docs to deploy LanceDB on other cloud providers such as google and azure.
// https://lancedb.github.io/lancedb/guides/storage/
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize OpenAI client. Use this to generate embeddings (and generate test data for RAG demo).
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    // Select the embedding model and generate our embeddings
    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Initialize LanceDB on S3.
    // Note: see below docs for more options and IAM permission required to read/write to S3.
    // https://lancedb.github.io/lancedb/guides/storage/#aws-s3
    let db = lancedb::connect("s3://lancedb-test-829666124233")
        .execute()
        .await?;

    // Set up test data for RAG demo
    let definition = "Definition of *flumbuzzle (verb)*: to bewilder or confuse someone completely, often by using nonsensical or overly complex explanations or instructions.".to_string();

    // Note: need at least 256 rows in order to create an index so copy the definition 256 times for testing purposes.
    let definitions = vec![definition; 256];

    // Generate embeddings for the test data.
    let embeddings = EmbeddingsBuilder::new(model.clone())
        .simple_document("doc0", "Definition of *flumbrel (noun)*: a small, seemingly insignificant item that you constantly lose or misplace, such as a pen, hair tie, or remote control.")
        .simple_document("doc1", "Definition of *zindle (verb)*: to pretend to be working on something important while actually doing something completely unrelated or unproductive")
        .simple_document("doc2", "Definition of *glimber (adjective)*: describing a state of excitement mixed with nervousness, often experienced before an important event or decision.")
        .simple_documents(definitions.clone().into_iter().enumerate().map(|(i, def)| (format!("doc{}", i+3), def)).collect())
        .build()
        .await?;

    // Create table with embeddings.
    let record_batch = as_record_batch(embeddings, model.ndims());
    let table = db
        .create_table(
            "definitions",
            RecordBatchIterator::new(vec![record_batch], Arc::new(schema(model.ndims()))),
        )
        .execute()
        .await?;

    // See [LanceDB indexing](https://lancedb.github.io/lancedb/concepts/index_ivfpq/#product-quantization) for more information
    table
        .create_index(
            &["embedding"],
            lancedb::index::Index::IvfPq(
                IvfPqIndexBuilder::default()
                    // This overrides the default distance type of L2.
                    // Needs to be the same distance type as the one used in search params.
                    .distance_type(DistanceType::Cosine),
            ),
        )
        .execute()
        .await?;

    // Define search_params params that will be used by the vector store to perform the vector search.
    let search_params = SearchParams::default().distance_type(DistanceType::Cosine);

    let vector_store = LanceDbVectorStore::new(table, model, "id", search_params).await?;

    // Query the index
    let results = vector_store
        .top_n::<VectorSearchResult>("I'm always looking for my phone, I always seem to forget it in the most counterintuitive places. What's the word for this feeling?", 1)
        .await?;

    println!("Results: {:?}", results);

    Ok(())
}
