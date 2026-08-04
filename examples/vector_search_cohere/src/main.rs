//! Demonstrates vector search with separate Cohere document and query embeddings.
//! Requires `COHERE_API_KEY` and the `derive` feature.
//! Run it to see a semantic query retrieve the closest matching document.
//!
//! Cohere distinguishes document embeddings from query embeddings through its
//! `input_type`. That used to be a constructor argument on the embedding
//! model; it is now a field on `cohere::functions::EmbeddingConfig`, so the
//! two roles are simply two configs (`with_input_type`) over the same model.

use rig::embeddings::EmbeddingJob;
use rig::prelude::*;
use rig::providers::cohere;
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

fn print_matches(matches: &[SearchMatch]) {
    println!("Top document matches:");
    for (score, id, word) in matches {
        println!("  score={score:.4} id={id} word={word}");
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let document_config = cohere::functions::EmbeddingConfig::from_env(cohere::EMBED_ENGLISH_V3)?
        .with_input_type("search_document");
    let search_config = cohere::functions::EmbeddingConfig::from_env(cohere::EMBED_ENGLISH_V3)?
        .with_input_type("search_query");
    let rt = HttpRuntime::new();

    let embeddings = EmbeddingJob::new()
        .documents(sample_documents())
        .for_provider(&cohere::functions::DESCRIPTOR)
        .run(|texts| cohere::functions::embed(&document_config, &rt, texts))
        .await?;

    let vector_store =
        InMemoryVectorStore::from_documents_with_id_f(embeddings, |doc| doc.id.clone())?;

    let query = "Which instrument is found in the Nebulon Mountain Ranges?";
    // Embed the query with the `search_query` config; the store receives it
    // pre-embedded.
    let query_embedding = cohere::functions::embed(&search_config, &rt, vec![query.to_string()])
        .await?
        .embeddings
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no embedding returned for the query"))?;
    let req = VectorSearchRequest::new(OneOrMany::one(query_embedding), 1);

    let results = vector_store
        .top_n_as::<WordDefinition>(req)
        .await?
        .into_iter()
        .map(|(score, id, doc)| (score, id, doc.word))
        .collect::<Vec<SearchMatch>>();

    print_matches(&results);

    Ok(())
}
