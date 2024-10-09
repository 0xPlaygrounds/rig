use std::env;

use rig::{
    embeddings::EmbeddingsBuilder,
    providers::cohere::{Client, EMBED_ENGLISH_V3},
    vector_store::{in_memory_store::InMemoryVectorStore, VectorStore, VectorStoreIndex},
};
use rig_derive::Embed;
use serde::{Deserialize, Serialize};

#[derive(Embed, Clone, Serialize, Default, Eq, PartialEq, Deserialize, Debug)]
struct FakeDefinition {
    id: String,
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

    let fake_definitions = vec![
        FakeDefinition {
            id: "doc0".to_string(),
            definitions: vec![
                "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
                "Definition of a *flurbo*: A unit of currency used in a bizarre or fantastical world, often associated with eccentric societies or sci-fi settings.".to_string()
            ]
        },
        FakeDefinition {
            id: "doc1".to_string(),
            definitions: vec![
                "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
                "Definition of a *glarb-glarb*: A mysterious, bubbling substance often found in swamps, alien planets, or under mysterious circumstances.".to_string()
            ]
        },
        FakeDefinition {
            id: "doc2".to_string(),
            definitions: vec![
                "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string()
            ]
        }
    ];

    let embeddings = EmbeddingsBuilder::new(document_model)
        .documents(fake_definitions)
        .build()
        .await?;

    let mut store = InMemoryVectorStore::default();
    store
        .add_documents(
            embeddings
                .into_iter()
                .enumerate()
                .map(|(i, (fake_definition, embeddings))| {
                    (format!("doc{i}"), fake_definition, embeddings)
                })
                .collect(),
        )
        .await?;

    let index = store.index(search_model);

    let results = index
        .top_n::<FakeDefinition>("What is a linglingdong?", 1)
        .await?
        .into_iter()
        .map(|(score, id, doc)| (score, id, doc))
        .collect::<Vec<_>>();

    println!("Results: {:?}", results);

    Ok(())
}
