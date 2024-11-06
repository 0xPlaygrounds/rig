use std::env;

use rig::{
    embeddings::{embed::EmbedError, Embedding, EmbeddingError, EmbeddingModel, EmbeddingsBuilder},
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::{in_memory_store::InMemoryVectorStore, VectorStoreIndex},
    Embed, OneOrMany,
};
use serde::{Deserialize, Serialize};

// Shape of data that needs to be RAG'ed.
// The definition field will be used to generate embeddings.
// #[derive(ExtractEmbeddingFields, Clone, Deserialize, Debug, Serialize, Eq, PartialEq, Default)]
struct FakeDefinition {
    id: String,
    word: String,
    // #[embed]
    definitions: Vec<String>,
}

impl Embed for FakeDefinition {
    fn embed(&self, embedder: &mut rig::embeddings::Embedder) -> Result<(), EmbedError> {
        for doc in &self.definitions {
            embedder.embed(doc.clone());
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    let docs = vec![
        FakeDefinition {
            id: "doc0".to_string(),
            word: "flurbo".to_string(),
            definitions: vec![
                "A green alien that lives on cold planets.".to_string(),
                "A fictional digital currency that originated in the animated series Rick and Morty.".to_string()
            ]
        },
        FakeDefinition {
            id: "doc1".to_string(),
            word: "glarb-glarb".to_string(),
            definitions: vec![
                "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
                "A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.".to_string()
            ]
        },
        FakeDefinition {
            id: "doc2".to_string(),
            word: "linglingdong".to_string(),
            definitions: vec![
                "A term used by inhabitants of the sombrero galaxy to describe humans.".to_string(),
                "A rare, mystical instrument crafted by the ancient monks of the Nebulon Mountain Ranges on the planet Quarm.".to_string()
            ]
        },
    ];

    let embeddings = model.embed_many(docs).await?;

    let data = vec![
        "What is a flurbo?",
        "What is a glarb-glarb?",
        "What is a linglingdong?",
    ];

    let embeddings = model.embed_many(data).await?;

    Ok(())
}
