use std::env;

use rig::{
    embeddings::builder::EmbeddingsBuilder,
    providers::cohere::{Client, EMBED_ENGLISH_V3},
    vector_store::{in_memory_store::InMemoryVectorStore, VectorStoreIndex},
    Embeddable,
};
use serde::{Deserialize, Serialize};

// Shape of data that needs to be RAG'ed.
// The definition field will be used to generate embeddings.
#[derive(Embeddable, Clone, Deserialize, Debug, Serialize, Eq, PartialEq, Default)]
struct FakeDefinition {
    id: String,
    word: String,
    #[embed]
    definitions: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create Cohere client
    let cohere_api_key = env::var("COHERE_API_KEY").expect("COHERE_API_KEY not set");
    let cohere_client = Client::new(&cohere_api_key);

    let document_model = cohere_client.embedding_model(EMBED_ENGLISH_V3, "search_document");
    let search_model = cohere_client.embedding_model(EMBED_ENGLISH_V3, "search_query");

    let embeddings = EmbeddingsBuilder::new(document_model.clone())
        .documents(vec![
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
        ])?
        .build()
        .await?;

    let index = InMemoryVectorStore::default()
        .add_documents(
            embeddings
                .into_iter()
                .map(|(fake_definition, embedding_vec)| {
                    (fake_definition.id.clone(), fake_definition, embedding_vec)
                })
                .collect(),
        )?
        .index(search_model);

    let results = index
        .top_n::<FakeDefinition>(
            "Which instrument is found in the Nebulon Mountain Ranges?",
            1,
        )
        .await?
        .into_iter()
        .map(|(score, id, doc)| (score, id, doc.word))
        .collect::<Vec<_>>();

    println!("Results: {:?}", results);

    Ok(())
}
