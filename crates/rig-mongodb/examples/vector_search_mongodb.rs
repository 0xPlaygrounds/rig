use mongodb::{
    Client as MongoClient, Collection,
    bson::{self, doc},
    options::ClientOptions,
};
use rig_core::OneOrMany;
use rig_core::{providers::openai, vector_store::request::VectorSearchRequest};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use std::env;

use rig_core::Embed;
use rig_core::embeddings::{default_concurrency, embed_documents};
use rig_core::http_runtime::HttpRuntime;
use rig_mongodb::{MongoDbSearchFilter, MongoDbVectorIndex, SearchParams};

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
    // Initialize MongoDB client
    let mongodb_connection_string = env::var("MONGODB_CONNECTION_STRING")?;
    let options = ClientOptions::parse(mongodb_connection_string).await?;

    let mongodb_client = MongoClient::with_options(options)?;

    // Initialize MongoDB vector store
    let collection: Collection<bson::Document> = mongodb_client
        .database("knowledgebase")
        .collection("context");

    // Embedding configuration is plain data plus a shared HTTP runtime — no
    // client object, no model handle.
    let embed_cfg = openai::functions::EmbeddingConfig::from_env(openai::TEXT_EMBEDDING_ADA_002)?;
    let rt = HttpRuntime::new();
    let max_documents = openai::functions::DESCRIPTOR
        .max_embedding_documents
        .unwrap_or(usize::MAX);

    let words = vec![
        Word {
            id: "doc0".to_string(),
            definition: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
        },
        Word {
            id: "doc1".to_string(),
            definition: "Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
        },
        Word {
            id: "doc2".to_string(),
            definition: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
        }
    ];

    let embeddings = embed_documents(
        words,
        max_documents,
        default_concurrency(max_documents),
        |texts| openai::functions::embed(&embed_cfg, &rt, texts),
    )
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
    let index = MongoDbVectorIndex::new(collection, "vector_index", SearchParams::new()).await?;

    // Embed the query outside the store (reuse the same configuration that
    // generated the document embeddings), then search with the pre-embedded query.
    let query =
        openai::functions::embed(&embed_cfg, &rt, vec!["What is a linglingdong?".to_string()])
            .await?
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no embedding returned for the query"))?;
    let req = VectorSearchRequest::<MongoDbSearchFilter>::new(OneOrMany::one(query), 1);

    // Query the index
    let results = index.top_n_as::<Word>(req.clone()).await?;

    println!("Results: {results:?}");

    let id_results = index.top_n_ids(req).await?.into_iter().collect::<Vec<_>>();

    println!("ID results: {id_results:?}");

    Ok(())
}
