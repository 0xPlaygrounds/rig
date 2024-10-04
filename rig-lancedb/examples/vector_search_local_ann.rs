use std::{env, sync::Arc};

use arrow_array::RecordBatchIterator;
use fixture::{as_record_batch, schema};
use lancedb::{index::vector::IvfPqIndexBuilder, DistanceType};
use rig::vector_store::VectorStoreIndex;
use rig::{
    completion::Prompt,
    embeddings::{EmbeddingModel, EmbeddingsBuilder},
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
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

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize OpenAI client. Use this to generate embeddings (and generate test data for RAG demo).
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    // Select an embedding model.
    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Generate test data for RAG demo
    let agent = openai_client
        .agent("gpt-4o")
        .preamble("Return the answer as JSON containing a list of strings in the form: `Definition of {generated_word}: {generated definition}`. Return ONLY the JSON string generated, nothing else.")
        .build();
    let response = agent
        .prompt("Invent 100 words and their definitions")
        .await?;
    let mut definitions: Vec<String> = serde_json::from_str(&response)?;

    // Note: need at least 256 rows in order to create an index on a table but OpenAI limits the output size
    // so we triplicate the vector for testing purposes.
    definitions.extend(definitions.clone());
    definitions.extend(definitions.clone());

    // Generate embeddings for the test data.
    let embeddings = EmbeddingsBuilder::new(model.clone())
        .simple_document("doc0", "Definition of *flumbrel (noun)*: a small, seemingly insignificant item that you constantly lose or misplace, such as a pen, hair tie, or remote control.")
        .simple_document("doc1", "Definition of *zindle (verb)*: to pretend to be working on something important while actually doing something completely unrelated or unproductive")
        .simple_document("doc2", "Definition of *glimber (adjective)*: describing a state of excitement mixed with nervousness, often experienced before an important event or decision.")
        .simple_documents(definitions.clone().into_iter().enumerate().map(|(i, def)| (format!("doc{}", i+3), def)).collect())
        .build()
        .await?;

    // Define search_params params that will be used by the vector store to perform the vector search.
    let search_params = SearchParams::default().distance_type(DistanceType::Cosine);

    // Initialize LanceDB locally.
    let db = lancedb::connect("data/lancedb-store").execute().await?;

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

    let vector_store = LanceDbVectorStore::new(table, model, "id", search_params).await?;

    // Query the index
    let results = vector_store
        .top_n::<VectorSearchResult>("My boss says I zindle too much, what does that mean?", 1)
        .await?;

    println!("Results: {:?}", results);

    Ok(())
}
