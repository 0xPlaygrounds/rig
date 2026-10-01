//! Demonstrates vector search against a local Ollama embedding model.
//! Requires a local Ollama server and the `derive` feature.
//! Run it to compare semantic search results and matching document ids.

use rig::providers::ollama::OllamaConfig;
use rig::vector_store::request::VectorSearchRequest;
use rig::{
    Embed,
    embeddings::EmbeddingsBuilder,
    vector_store::{
        VectorSearchIdResult, VectorSearchResult, VectorStoreIndex,
        in_memory_store::InMemoryVectorStore,
    },
};

use serde::{Deserialize, Serialize};

// Shape of data that needs to be RAG'ed.
// The definition field will be used to generate embeddings.
#[derive(Embed, Clone, Deserialize, Debug, Serialize, Eq, PartialEq, Default)]
struct WordDefinition {
    id: String,
    word: String,
    #[embed]
    definitions: Vec<String>,
}

fn sample_documents() -> Vec<WordDefinition> {
    vec![
        WordDefinition {
            id: "doc0".to_string(),
            word: "flurbo".to_string(),
            definitions: vec![
                "A green alien that lives on cold planets.".to_string(),
                "A fictional digital currency that originated in the animated series Rick and Morty.".to_string(),
            ],
        },
        WordDefinition {
            id: "doc1".to_string(),
            word: "glarb-glarb".to_string(),
            definitions: vec![
                "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
                "A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.".to_string(),
            ],
        },
        WordDefinition {
            id: "doc2".to_string(),
            word: "linglingdong".to_string(),
            definitions: vec![
                "A term used by inhabitants of the sombrero galaxy to describe humans.".to_string(),
                "A rare, mystical instrument crafted by the ancient monks of the Nebulon Mountain Ranges on the planet Quarm.".to_string(),
            ],
        },
    ]
}

fn print_matches(label: &str, matches: &[VectorSearchResult<WordDefinition>]) {
    println!("{label}:");
    for result in matches {
        println!(
            "  score={:.4} id={} word={}",
            result.score, result.id, result.document.word
        );
    }
}

fn print_id_matches(label: &str, matches: &[VectorSearchIdResult]) {
    println!("{label}:");
    for result in matches {
        println!("  score={:.4} id={}", result.score, result.id);
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = OllamaConfig::new()
        .with_base_url("http://localhost:11434")
        .client();

    let embedding_model = client.embedding("nomic-embed-text", None).erase();

    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(sample_documents())?
        .build()
        .await?;

    let vector_store =
        InMemoryVectorStore::from_documents_with_id_f(embeddings, |doc| doc.id.clone());

    let index = vector_store.index(embedding_model);

    let query =
        "I need to buy something in a fictional universe. What type of money can I use for this?";
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(1)
        .build();

    let results = index.top_n::<WordDefinition>(req.clone()).await?;

    let id_results = index.top_n_ids(req).await?;

    print_matches("Top document matches", &results);
    print_id_matches("Top document ids", &id_results);

    Ok(())
}
