use helix_rs::{HelixDB, HelixDBClient};
use rig::{
    Embed,
    client::{EmbeddingsClient, ProviderClient},
    embeddings::EmbeddingModel,
    vector_store::{VectorSearchRequest, VectorStoreIndex},
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

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct QueryInput {
    vector: Vec<f64>,
    doc: String,
    json_payload: String,
}

#[tokio::main]
async fn main() {
    let openai_model =
        rig::providers::openai::Client::from_env().embedding_model("text-embedding-ada-002");

    let helixdb_client = HelixDB::new(None, Some(6969), None); // Uses default port 6969

    let doc = "Hello world!".to_string();
    let vector = openai_model.embed_text(&doc).await.unwrap().vec;

    let queryinput = QueryInput {
        vector,
        doc: doc.clone(),
        json_payload: "todo".to_string(),
    };

    #[derive(Deserialize)]
    struct Thing {
        doc: String,
    }

    helixdb_client
        .query::<QueryInput, Thing>("InsertVector", &queryinput)
        .await
        .unwrap();

    let thing = HelixDBVectorStore::new(helixdb_client, openai_model);

    let vector_req = VectorSearchRequest::builder()
        .query(doc)
        .samples(5)
        .build()
        .unwrap();

    let docs = thing.top_n::<String>(vector_req).await.unwrap();
}
