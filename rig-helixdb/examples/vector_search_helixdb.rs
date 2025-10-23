use helix_rs::{HelixDB, HelixDBClient};
use rig::{
    Embed,
    client::EmbeddingsClient,
    embeddings::EmbeddingsBuilder,
    vector_store::{InsertDocuments, VectorSearchRequest, VectorStoreIndex},
};
use rig_helixdb::HelixDBVectorStore;
use serde::{Deserialize, Serialize};

// A vector search needs to be performed on the `definitions` field, so we derive the `Embed` trait for `WordDefinition`
// and tag that field with `#[embed]`.
// We are not going to store the definitions on our database so we skip the `Serialize` trait
#[derive(Embed, Serialize, Deserialize, Clone, Debug, Eq, PartialEq, Default)]
struct WordDefinition {
    word: String,
    #[serde(skip)] // we don't want to serialize this field, we use only to create embeddings
    #[embed]
    definition: String,
}

impl std::fmt::Display for WordDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.word)
    }
}

#[tokio::main]
async fn main() {
    let openai_model =
        rig::providers::openai::Client::from_env().embedding_model("text-embedding-ada-002");

    let helixdb_client = HelixDB::new(None, Some(6969), None); // Uses default port 6969
    let vector_store = HelixDBVectorStore::new(helixdb_client, openai_model.clone());

    let words = vec![
        WordDefinition {
            word: "flurbo".to_string(),
            definition: "1. *flurbo* (name): A fictional digital currency that originated in the animated series Rick and Morty.".to_string()
        },
        WordDefinition {
            word: "glarb-glarb".to_string(),
            definition: "1. *glarb-glarb* (noun): A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.".to_string()
        },
        WordDefinition {
            word: "linglingdong".to_string(),
            definition: "1. *linglingdong* (noun): A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
        }];

    let documents = EmbeddingsBuilder::new(openai_model)
        .documents(words)
        .unwrap()
        .build()
        .await
        .expect("Failed to create embeddings");

    vector_store.insert_documents(documents).await.unwrap();

    let query = "What is a flurbo?";
    let vector_req = VectorSearchRequest::builder()
        .query(query)
        .samples(5)
        .build()
        .unwrap();

    let docs = vector_store
        .top_n::<WordDefinition>(vector_req)
        .await
        .unwrap();

    for doc in docs {
        println!(
            "Vector found with id: {id} and score: {score} and word def: {doc}",
            id = doc.1,
            score = doc.0,
            doc = doc.2
        )
    }
}
