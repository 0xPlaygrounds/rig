//! Demonstrates embedding documents and querying an in-memory vector index with OpenAI.
//! Requires `OPENAI_API_KEY` and the `derive` feature.
//! Run it to compare `top_n` results with `top_n_ids`.
//!
//! Embedding is plain data plus free functions: an
//! `openai::functions::EmbeddingConfig` (which names the embedding model) and
//! an [`HttpRuntime`]. `EmbeddingsBuilder` is gone — [`embed_documents`]
//! takes the documents, the provider's per-request document limit, a
//! concurrency bound, and a closure over the provider's free `embed`.

use rig::embeddings::default_concurrency;
use rig::prelude::*;
use rig::providers::openai;
use serde::{Deserialize, Serialize};

type SearchMatch = (f64, String, String);

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

fn print_matches(label: &str, matches: &[SearchMatch]) {
    println!("{label}:");
    for (score, id, word) in matches {
        println!("  score={score:.4} id={id} word={word}");
    }
}

fn print_id_matches(label: &str, matches: &[(f64, String)]) {
    println!("{label}:");
    for (score, id) in matches {
        println!("  score={score:.4} id={id}");
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let ecfg = openai::functions::EmbeddingConfig::from_env(openai::TEXT_EMBEDDING_ADA_002)?;
    let rt = HttpRuntime::new();

    // The provider descriptor carries the per-request document limit; the
    // batching helper chunks and parallelizes to match it.
    let max_documents = openai::functions::DESCRIPTOR
        .max_embedding_documents
        .unwrap_or(usize::MAX);
    let embeddings = embed_documents(
        sample_documents(),
        max_documents,
        default_concurrency(max_documents),
        |texts| openai::functions::embed(&ecfg, &rt, texts),
    )
    .await?;

    let vector_store =
        InMemoryVectorStore::from_documents_with_id_f(embeddings, |doc| doc.id.clone())?;

    let query =
        "I need to buy something in a fictional universe. What type of money can I use for this?";
    // Queries are embedded up front; the store only sees pre-embedded requests.
    let query_embedding = openai::functions::embed(&ecfg, &rt, vec![query.to_string()])
        .await?
        .embeddings
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no embedding returned for the query"))?;
    let req = VectorSearchRequest::new(OneOrMany::one(query_embedding), 1);

    let results = vector_store
        .top_n_as::<WordDefinition>(req.clone())
        .await?
        .into_iter()
        .map(|(score, id, doc)| (score, id, doc.word))
        .collect::<Vec<SearchMatch>>();

    let id_results = vector_store
        .top_n_ids(req)
        .await?
        .into_iter()
        .collect::<Vec<_>>();

    print_matches("Top document matches", &results);
    print_id_matches("Top document ids", &id_results);

    Ok(())
}
