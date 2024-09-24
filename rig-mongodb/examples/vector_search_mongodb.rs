use mongodb::{options::ClientOptions, Client as MongoClient, Collection};
use std::env;

use rig::{
    embeddings::{DocumentEmbeddings, EmbeddingsBuilder},
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::{VectorStore, VectorStoreIndex},
};
use rig_mongodb::{MongoDbVectorStore, SearchParams};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    // Initialize MongoDB client
    let mongodb_connection_string =
        env::var("MONGODB_CONNECTION_STRING").expect("MONGODB_CONNECTION_STRING not set");
    let options = ClientOptions::parse(mongodb_connection_string)
        .await
        .expect("MongoDB connection string should be valid");

    let mongodb_client =
        MongoClient::with_options(options).expect("MongoDB client options should be valid");

    // Initialize MongoDB vector store
    let collection: Collection<DocumentEmbeddings> = mongodb_client
        .database("knowledgebase")
        .collection("context");

    let mut vector_store = MongoDbVectorStore::new(collection);

    // Select the embedding model and generate our embeddings
    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    let embeddings = EmbeddingsBuilder::new(model.clone())
        .simple_document("doc0", "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")
        .simple_document("doc1", "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
        .simple_document("doc2", "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.")
        .build()
        .await?;

    // Add embeddings to vector store
    match vector_store.add_documents(embeddings).await {
        Ok(_) => println!("Documents added successfully"),
        Err(e) => println!("Error adding documents: {:?}", e),
    }

    // Create a vector index on our vector store
    // IMPORTANT: Reuse the same model that was used to generate the embeddings
    let index = vector_store.index(model, "context_vector_index", SearchParams::new());

    // Query the index
    let results = index
        .top_n_from_query("What is a linglingdong?", 1)
        .await?
        .into_iter()
        .map(|(score, doc)| (score, doc.id, doc.document))
        .collect::<Vec<_>>();

    println!("Results: {:?}", results);

    Ok(())
}
