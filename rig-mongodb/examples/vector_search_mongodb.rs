use mongodb::{
    bson::{self, doc},
    options::ClientOptions,
    Client as MongoClient, Collection,
};
use rig::providers::openai::TEXT_EMBEDDING_ADA_002;
use serde::Deserialize;
use std::env;

use rig::{
    embeddings::EmbeddingsBuilder, providers::openai::Client, vector_store::VectorStoreIndex, Embed,
};
use rig_mongodb::{MongoDbVectorIndex, SearchParams};

// Shape of data that needs to be RAG'ed.
// The definition field will be used to generate embeddings.
#[derive(Embed, Clone, Deserialize, Debug)]
struct Word {
    #[serde(rename = "_id")]
    id: String,
    #[embed]
    definition: String,
}

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
    let collection: Collection<bson::Document> = mongodb_client
        .database("knowledgebase")
        .collection("context");

    // Select the embedding model and generate our embeddings
    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    let words = vec![
        Word {
            id: "doc0".to_string(),
            definition: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
        },
        Word {
            id: "doc1".to_string(),
            definition: "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
        },
        Word {
            id: "doc2".to_string(),
            definition: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
        }
    ];

    let embeddings = EmbeddingsBuilder::new(model.clone())
        .documents(words)?
        .build()
        .await?;

    let mongo_documents = embeddings
        .iter()
        .map(|(Word { id, definition, .. }, embedding)| {
            doc! {
                "id": id.clone(),
                "definition": definition.clone(),
                "embedding": embedding.first().vec.clone(),
            }
        })
        .collect::<Vec<_>>();

    match collection.insert_many(mongo_documents).await {
        Ok(_) => println!("Documents added successfully"),
        Err(e) => println!("Error adding documents: {:?}", e),
    };

    // Create a vector index on our vector store.
    // Note: a vector index called "vector_index" must exist on the MongoDB collection you are querying.
    // IMPORTANT: Reuse the same model that was used to generate the embeddings
    let index =
        MongoDbVectorIndex::new(collection, model, "vector_index", SearchParams::new()).await?;

    // Query the index
    let results = index.top_n::<Word>("What is a linglingdong?", 1).await?;

    println!("Results: {:?}", results);

    let id_results = index
        .top_n_ids("What is a linglingdong?", 1)
        .await?
        .into_iter()
        .collect::<Vec<_>>();

    println!("ID results: {:?}", id_results);

    Ok(())
}
