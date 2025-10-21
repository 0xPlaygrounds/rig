//! Simple end-to-end example of the vector search capabilities of the `rig-neo4j` crate.
//! This example expects a running Neo4j instance running.
//! It:
//! 1. Generates embeddings for a set of 3 "documents"
//! 2. Adds the documents to the Neo4j DB
//! 3. Creates a vector index on the embeddings
//! 4. Queries the vector index
//! 5. Returns the results
use std::env;

use futures::{StreamExt, TryStreamExt};
use rig::client::EmbeddingsClient;
use rig::vector_store::request::VectorSearchRequest;
use rig::{
    Embed,
    embeddings::EmbeddingsBuilder,
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::VectorStoreIndex as _,
};
use rig_neo4j::{Neo4jClient, ToBoltType};

#[derive(Embed, Clone, Debug)]
pub struct Word {
    pub id: String,
    #[embed]
    pub definition: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    // Initialize Neo4j client
    let neo4j_uri = env::var("NEO4J_URI").expect("NEO4J_URI not set");
    let neo4j_username = env::var("NEO4J_USERNAME").expect("NEO4J_USERNAME not set");
    let neo4j_password = env::var("NEO4J_PASSWORD").expect("NEO4J_PASSWORD not set");

    let neo4j_client = Neo4jClient::connect(&neo4j_uri, &neo4j_username, &neo4j_password).await?;

    // Select the embedding model and generate our embeddings
    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    let embeddings = EmbeddingsBuilder::new(model.clone())
        .document(Word {
            id: "doc0".to_string(),
            definition: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
        })?
        .document(Word {
            id: "doc1".to_string(),
            definition: "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
        })?
        .document(Word {
            id: "doc2".to_string(),
            definition: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
        })?
        .build()
        .await?;

    futures::stream::iter(embeddings)
        .map(|(doc, embeddings)| {
            neo4j_client.graph.run(
                neo4rs::query(
                    "
                        CREATE
                            (document:DocumentEmbeddings {
                                id: $id,
                                document: $document,
                                embedding: $embedding})
                        RETURN document",
                )
                .param("id", doc.id)
                // Here we use the first embedding but we could use any of them.
                // Neo4j only takes primitive types or arrays as properties.
                .param("embedding", embeddings.first().vec.clone())
                .param("document", doc.definition.to_bolt_type()),
            )
        })
        .buffer_unordered(3)
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    // Create a vector index on our vector store
    println!("Creating vector index...");
    neo4j_client
        .graph
        .run(neo4rs::query(
            "CREATE VECTOR INDEX vector_index IF NOT EXISTS
            FOR (m:DocumentEmbeddings)
            ON m.embedding
            OPTIONS { indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
                }}",
        ))
        .await?;

    // ℹ️ The index name must be unique among both indexes and constraints.
    // A newly created index is not immediately available but is created in the background.

    // Check if the index exists with db.awaitIndex(), the call timeouts if the index is not ready
    let index_exists = neo4j_client
        .graph
        .run(neo4rs::query("CALL db.awaitIndex('vector_index')"))
        .await;
    if index_exists.is_err() {
        println!("Index not ready, waiting for index...");
        std::thread::sleep(std::time::Duration::from_secs(5));
    }

    println!("Index exists: {index_exists:?}");

    // Create a vector index on our vector store
    // IMPORTANT: Reuse the same model that was used to generate the embeddings
    let index = neo4j_client.get_index(model, "vector_index").await?;

    // The struct that will represent a node in the database. Used to deserialize the results of the query (passed to the `top_n` methods)
    // ❗IMPORTANT: The field names must match the property names in the database
    #[derive(serde::Deserialize)]
    struct Document {
        #[allow(dead_code)]
        id: String,
        document: String,
    }

    let query1 = "What is a glarb?";
    let query2 = "What is a linglingdong?";

    let req = VectorSearchRequest::builder()
        .query(query1)
        .samples(1)
        .build()?;

    // Query the index
    let results = index
        .top_n::<Document>(req)
        .await?
        .into_iter()
        .map(|(score, id, doc)| (score, id, doc.document))
        .collect::<Vec<_>>();

    println!("Results: {results:?}");

    let req = VectorSearchRequest::builder()
        .query(query2)
        .samples(1)
        .build()?;

    let id_results = index.top_n_ids(req).await?.into_iter().collect::<Vec<_>>();

    println!("ID results: {id_results:?}");

    Ok(())
}
