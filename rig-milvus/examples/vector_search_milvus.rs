use rig::vector_store::InsertDocuments;
use rig::vector_store::request::VectorSearchRequest;
use rig::{
    Embed, client::EmbeddingsClient, embeddings::EmbeddingsBuilder, vector_store::VectorStoreIndex,
};
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
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_client = rig::providers::openai::Client::from_env();
    let model = openai_client.embedding_model(rig::providers::openai::TEXT_EMBEDDING_3_SMALL);

    let base_url = std::env::var("MILVUS_BASE_URL").expect("the MILVUS_BASE_URL env var to exist");
    let collection_name = std::env::var("MILVUS_COLLECTION_NAME")
        .expect("the MILVUS_COLLECTION_NAME env var to exist");
    let database_name =
        std::env::var("MILVUS_DATABASE_NAME").expect("the MILVUS_DATABASE_NAME env var to exist");
    let milvus_user =
        std::env::var("MILVUS_USERNAME").expect("the MILVUS_USERNAME env var to exist");
    let milvus_password =
        std::env::var("MILVUS_PASSWORD").expect("the MILVUS_PASSWORD env var to exist");

    let vector_store =
        rig_milvus::MilvusVectorStore::new(model.clone(), base_url, database_name, collection_name)
            .auth(milvus_user, milvus_password);

    // create test documents with mocked embeddings
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

    let documents = EmbeddingsBuilder::new(model.clone())
        .documents(words)
        .unwrap()
        .build()
        .await
        .expect("Failed to create embeddings");

    vector_store.insert_documents(documents).await?;

    // query vector
    let query = "What does \"glarb-glarb\" mean?";

    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(2)
        .build()
        .expect("VectorSearchRequest should not fail to build here");

    let results = vector_store.top_n::<WordDefinition>(req).await?;

    println!("#{} results for query: {}", results.len(), query);
    for (distance, _id, doc) in results.iter() {
        println!("Result distance {distance} for word: {doc}");

        // expected output
        // Result distance 0.693218142100547 for word: glarb-glarb
        // Result distance 0.2529120980283861 for word: linglingdong
    }

    Ok(())
}
