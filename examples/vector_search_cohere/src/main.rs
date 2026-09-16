//! Demonstrates vector search with separate Cohere document and query embeddings.
//! Requires `COHERE_API_KEY` and the `derive` feature.
//! Run it to see a semantic query retrieve the closest matching document.

use rig::{
    Embed,
    embeddings::EmbeddingsBuilder,
    prelude::*,
    providers::cohere::{self, Cohere},
    vector_store::{
        VectorStoreIndex, in_memory_store::InMemoryVectorStore, request::VectorSearchRequest,
    },
};
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
    let cohere_client = Cohere::from_env()?.bound()?;
    // Cohere scores a document and a query differently, so the two wires
    // differ only in the `input_type` they send.
    let document_model = cohere_client
        .embedding(cohere::EMBED_ENGLISH_V3, None)
        .map_wire(|wire| wire.with_input_type("search_document"));
    let search_model = cohere_client
        .embedding(cohere::EMBED_ENGLISH_V3, None)
        .map_wire(|wire| wire.with_input_type("search_query"));
    let embeddings = EmbeddingsBuilder::new(document_model.clone())
        .documents(sample_documents())?
        .build()
        .await?;

    let vector_store =
        InMemoryVectorStore::from_documents_with_id_f(embeddings, |doc| doc.id.clone());

    let query = "Which instrument is found in the Nebulon Mountain Ranges?";
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(1)
        .build();

    let index = vector_store.index(search_model);
    let results = index
        .top_n::<WordDefinition>(req)
        .await?
        .into_iter()
        .map(|(score, id, doc)| (score, id, doc.word))
        .collect::<Vec<SearchMatch>>();

    print_matches(&results);

    Ok(())
}
