use mongodb::{
    Client as MongoClient, Collection,
    bson::{self, doc},
    options::ClientOptions,
};
use rig::{client::ProviderClient, providers::openai, vector_store::request::VectorSearchRequest};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use std::env;

use rig::client::EmbeddingsClient;
use rig::{
    Embed, embeddings::EmbeddingsBuilder, providers::openai::Client, vector_store::VectorStoreIndex,
};
use rig_mongodb::{MongoDbVectorIndex, SearchParams};

// Shape of data that needs to be RAG'ed.
// The definition field will be used to generate embeddings.
#[derive(Embed, Clone, Deserialize, Debug)]
struct Word {
    #[serde(rename = "_id", deserialize_with = "deserialize_object_id")]
    id: String,
    #[embed]
    definition: String,
}

fn deserialize_object_id<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::String(s) => Ok(s),
        Value::Object(map) => {
            if let Some(Value::String(oid)) = map.get("$oid") {
                Ok(oid.to_string())
            } else {
                Err(serde::de::Error::custom(
                    "Expected $oid field with string value",
                ))
            }
        }
        _ => Err(serde::de::Error::custom(
            "Expected string or object with $oid field",
        )),
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize OpenAI client
    let openai_client = Client::from_env();

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
    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

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
        Err(e) => println!("Error adding documents: {e:?}"),
    };

    // Create a vector index on our vector store.
    // Note: a vector index called "vector_index" must exist on the MongoDB collection you are querying.
    // IMPORTANT: Reuse the same model that was used to generate the embeddings
    let index =
        MongoDbVectorIndex::new(collection, model, "vector_index", SearchParams::new()).await?;

    let query = "What is a linglingdong?";
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(1)
        .build()
        .expect("VectorSearchRequest should not fail to build here");

    // Query the index
    let results = index.top_n::<Word>(req.clone()).await?;

    println!("Results: {results:?}");

    let id_results = index.top_n_ids(req).await?.into_iter().collect::<Vec<_>>();

    println!("ID results: {id_results:?}");

    Ok(())
}
