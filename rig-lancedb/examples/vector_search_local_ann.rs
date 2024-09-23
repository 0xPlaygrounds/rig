use std::env;

use lancedb::{index::vector::IvfPqIndexBuilder, DistanceType};
use rig::{
    completion::Prompt,
    embeddings::EmbeddingsBuilder,
    providers::openai::{Client, OpenAIEmbeddingModel},
    vector_store::{VectorStore, VectorStoreIndexDyn},
};
use rig_lancedb::{LanceDbVectorStore, SearchParams};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize OpenAI client. Use this to generate embeddings (and generate test data for RAG demo).
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    // Select the embedding model and generate our embeddings
    let model = openai_client.embedding_model(&OpenAIEmbeddingModel::TextEmbeddingAda002);

    // Initialize LanceDB locally.
    let db = lancedb::connect("data/lancedb-store").execute().await?;
    let mut vector_store = LanceDbVectorStore::new(&db, &model).await?;

    // Generate test data for RAG demo
    let agent = openai_client
        .agent("gpt-4o")
        .preamble("Return the answer as JSON containing a list of strings in the form: `Definition of {generated_word}: {generated definition}`. Return ONLY the JSON string generated, nothing else.")
        .build();
    let response = agent
        .prompt("Invent at least 100 words and their definitions")
        .await?;
    let mut definitions: Vec<String> = serde_json::from_str(&response)?;

    // Note: need at least 256 rows in order to create an index on a table but OpenAI limits the output size
    // so we triplicate the vector for testing purposes.
    definitions.extend(definitions.clone());
    definitions.extend(definitions.clone());

    let embeddings = EmbeddingsBuilder::new(model.clone())
        .simple_document("doc0", "Definition of *flumbrel (noun)*: a small, seemingly insignificant item that you constantly lose or misplace, such as a pen, hair tie, or remote control.")
        .simple_document("doc1", "Definition of *zindle (verb)*: to pretend to be working on something important while actually doing something completely unrelated or unproductive")
        .simple_document("doc2", "Definition of *glimber (adjective)*: describing a state of excitement mixed with nervousness, often experienced before an important event or decision.")
        .simple_documents(definitions.clone().into_iter().enumerate().map(|(i, def)| (format!("doc{}", i+3), def)).collect())
        .build()
        .await?;

    // Add embeddings to vector store
    vector_store.add_documents(embeddings).await?;

    // See [LanceDB indexing](https://lancedb.github.io/lancedb/concepts/index_ivfpq/#product-quantization) for more information
    vector_store
        .create_index(lancedb::index::Index::IvfPq(
            IvfPqIndexBuilder::default()
                // This overrides the default distance type of L2
                .distance_type(DistanceType::Cosine),
        ))
        .await?;

    // Query the index
    let results = vector_store
        .top_n_from_query(
            "My boss says I zindle too much, what does that mean?",
            1,
            &serde_json::to_string(&SearchParams::new(
                Some(DistanceType::Cosine),
                None,
                None,
                None,
                None,
            ))?,
        )
        .await?
        .into_iter()
        .map(|(score, doc)| (score, doc.id, doc.document))
        .collect::<Vec<_>>();

    println!("Results: {:?}", results);

    Ok(())
}
