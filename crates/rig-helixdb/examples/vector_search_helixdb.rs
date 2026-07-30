use rig_core::OneOrMany;
use rig_core::embeddings::{default_concurrency, embed_documents};
use rig_core::http_runtime::HttpRuntime;
use rig_core::{
    Embed,
    providers::openai,
    vector_store::{StoreRecord, VectorSearchRequest},
};
use rig_helixdb::{HelixDB, HelixDBVectorStore};
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
    // Embedding configuration is plain data plus a shared HTTP runtime.
    let embed_cfg = openai::functions::EmbeddingConfig::from_env(openai::TEXT_EMBEDDING_ADA_002)?;
    let rt = HttpRuntime::new();
    let max_documents = openai::functions::DESCRIPTOR
        .max_embedding_documents
        .unwrap_or(usize::MAX);

    let helixdb_client = HelixDB::new(None, Some(6969), None); // Uses default port 6969
    // The store holds no embedding model: embedding happens outside the store,
    // and both records and queries arrive pre-embedded.
    let vector_store = HelixDBVectorStore::new(helixdb_client);

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

    let documents = embed_documents(
        words,
        max_documents,
        default_concurrency(max_documents),
        |texts| openai::functions::embed(&embed_cfg, &rt, texts),
    )
    .await?;

    let records = documents
        .into_iter()
        .enumerate()
        .map(|(i, (doc, embeddings))| StoreRecord::new(format!("doc{i}"), &doc, embeddings))
        .collect::<Result<Vec<_>, _>>()?;

    vector_store.insert(records).await?;

    let query = "What is a flurbo?";
    let query_embedding = openai::functions::embed(&embed_cfg, &rt, vec![query.to_string()])
        .await?
        .embeddings
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no embedding returned for the query"))?;
    let vector_req = VectorSearchRequest::new(OneOrMany::one(query_embedding), 5);

    let docs = vector_store.top_n_as::<WordDefinition>(vector_req).await?;

    for doc in docs {
        println!(
            "Vector found with id: {id} and score: {score} and word def: {doc}",
            id = doc.1,
            score = doc.0,
            doc = doc.2
        )
    }

    Ok(())
}
