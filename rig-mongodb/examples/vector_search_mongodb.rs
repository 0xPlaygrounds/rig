use mongodb::{bson::doc, options::ClientOptions, Client as MongoClient, Collection};
use rig::providers::openai::TEXT_EMBEDDING_ADA_002;
use serde::{Deserialize, Serialize};
use std::env;

use rig::Embeddable;
use rig::{
    embeddings::builder::EmbeddingsBuilder, providers::openai::Client,
    vector_store::VectorStoreIndex,
};
use rig_mongodb::{MongoDbVectorStore, SearchParams};

// Shape of data that needs to be RAG'ed.
// The definition field will be used to generate embeddings.
#[derive(Embeddable, Clone, Deserialize, Debug)]
struct FakeDefinition {
    id: String,
    #[embed]
    definition: String,
}

#[derive(Clone, Deserialize, Debug, Serialize)]
struct Link {
    word: String,
    link: String,
}

// Shape of the document to be stored in MongoDB, with embeddings.
#[derive(Serialize, Debug)]
struct Document {
    #[serde(rename = "_id")]
    id: String,
    definition: String,
    embedding: Vec<f64>,
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
    let collection: Collection<Document> = mongodb_client
        .database("knowledgebase")
        .collection("context");

    // Select the embedding model and generate our embeddings
    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    let fake_definitions = vec![
        FakeDefinition {
            id: "doc0".to_string(),
            definition: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
        },
        FakeDefinition {
            id: "doc1".to_string(),
            definition: "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
        },
        FakeDefinition {
            id: "doc2".to_string(),
            definition: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
        }
    ];

    let embeddings = EmbeddingsBuilder::new(model.clone())
        .documents(fake_definitions)?
        .build()
        .await?;

    let mongo_documents = embeddings
        .iter()
        .map(
            |(FakeDefinition { id, definition, .. }, embedding)| Document {
                id: id.clone(),
                definition: definition.clone(),
                embedding: embedding.vec.clone(),
            },
        )
        .collect::<Vec<_>>();

    match collection.insert_many(mongo_documents, None).await {
        Ok(_) => println!("Documents added successfully"),
        Err(e) => println!("Error adding documents: {:?}", e),
    };

    // Create a vector index on our vector store
    // IMPORTANT: Reuse the same model that was used to generate the embeddings
    let index = MongoDbVectorStore::new(collection).index(
        model,
        "definitions_vector_index",
        SearchParams::new("embedding"),
    );

    // Query the index
    let results = index
        .top_n::<FakeDefinition>("What is a linglingdong?", 1)
        .await?;

    println!("Results: {:?}", results);

    let id_results = index
        .top_n_ids("What is a linglingdong?", 1)
        .await?
        .into_iter()
        .map(|(score, id)| (score, id))
        .collect::<Vec<_>>();

    println!("ID results: {:?}", id_results);

    Ok(())
}
