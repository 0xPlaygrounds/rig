// Run a local Weaviate, then the example with an OpenAI key:
// docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.40.1
// OPENAI_API_KEY=... cargo run -p rig-weaviate --example vector_search_weaviate
use rig_core::{
    Embed,
    embeddings::EmbeddingsBuilder,
    providers::openai::{self, OpenAI},
    vector_store::{
        InsertDocuments, VectorStoreIndex,
        request::{SearchFilter, VectorSearchRequest},
    },
};
use rig_weaviate::{WeaviateFilter, WeaviateVectorStore};
use serde::{Deserialize, Serialize};

#[derive(Embed, Serialize, Deserialize, Clone, Debug)]
struct WordDefinition {
    word: String,
    #[embed]
    definition: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = OpenAI::from_env()?.embedding(openai::TEXT_EMBEDDING_3_SMALL, None);
    let url = std::env::var("WEAVIATE_URL").unwrap_or_else(|_| "http://localhost:8080".into());
    let mut store = WeaviateVectorStore::new(model.clone(), url, "WordDefinition")?;
    if let Ok(key) = std::env::var("WEAVIATE_API_KEY") {
        store = store.api_key(key);
    }

    let words = [
        ("flurbo", "A green alien that lives on cold planets."),
        (
            "glarb-glarb",
            "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
        ),
        (
            "linglingdong",
            "A term used by inhabitants of the far side of the moon to describe humans.",
        ),
    ]
    .map(|(word, definition)| WordDefinition {
        word: word.into(),
        definition: format!("Definition of a *{word}*: {definition}"),
    });
    let documents = EmbeddingsBuilder::new(model)
        .documents(words)?
        .build()
        .await?;
    store.insert_documents(documents).await?;

    let query = "What does \"glarb-glarb\" mean?";
    let request = VectorSearchRequest::builder()
        .query(query)
        .samples(2)
        .build();
    for result in store.top_n::<WordDefinition>(request).await? {
        println!(
            "{:.3} {}: {}",
            result.score, result.id, result.document.word
        );
    }

    // Each object stores its document as JSON text in `document`, which a
    // `Like` pattern can match.
    let request = VectorSearchRequest::builder()
        .query(query)
        .samples(2)
        .filter(
            WeaviateFilter::like("document", "*moon*")
                .or(WeaviateFilter::eq("document", "missing".into())),
        )
        .build();
    for result in store.top_n_ids(request).await? {
        println!("filtered {:.3} {}", result.score, result.id);
    }

    Ok(())
}
