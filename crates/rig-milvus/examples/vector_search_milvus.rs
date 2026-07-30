use rig_core::OneOrMany;
use rig_core::client::ProviderClient;
use rig_core::vector_store::request::VectorSearchRequest;
use rig_core::{
    Embed,
    client::EmbeddingsClient,
    embeddings::{EmbeddingModel, EmbeddingsBuilder},
    vector_store::StoreRecord,
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
    let openai_client = rig_core::providers::openai::Client::from_env()?;
    let model = openai_client.embedding_model(rig_core::providers::openai::TEXT_EMBEDDING_3_SMALL);

    let base_url = std::env::var("MILVUS_BASE_URL")?;
    let collection_name = std::env::var("MILVUS_COLLECTION_NAME")?;
    let database_name = std::env::var("MILVUS_DATABASE_NAME")?;
    let milvus_user = std::env::var("MILVUS_USERNAME")?;
    let milvus_password = std::env::var("MILVUS_PASSWORD")?;

    // The store holds no embedding model: embedding happens outside the store,
    // and both records and queries arrive pre-embedded.
    let vector_store = rig_milvus::MilvusVectorStore::new(base_url, database_name, collection_name)
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
        .documents(words)?
        .build()
        .await?;

    let records = documents
        .into_iter()
        .enumerate()
        .map(|(i, (doc, embeddings))| StoreRecord::new(format!("doc{i}"), &doc, embeddings))
        .collect::<Result<Vec<_>, _>>()?;

    vector_store.insert(records).await?;

    // query vector
    let query = "What does \"glarb-glarb\" mean?";
    let query_embedding = model.embed_text(query).await?;

    let req = VectorSearchRequest::new(OneOrMany::one(query_embedding), 2);

    let results = vector_store.top_n_as::<WordDefinition>(req).await?;

    println!("#{} results for query: {}", results.len(), query);
    for (distance, _id, doc) in results.iter() {
        println!("Result distance {distance} for word: {doc}");

        // expected output
        // Result distance 0.693218142100547 for word: glarb-glarb
        // Result distance 0.2529120980283861 for word: linglingdong
    }

    Ok(())
}
