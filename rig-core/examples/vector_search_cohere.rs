use std::env;

use rig::{
    embeddings::{DocumentEmbeddings, EmbeddingsBuilder},
    providers::cohere::{Client, EMBED_ENGLISH_V3},
    vector_store::{in_memory_store::InMemoryVectorStore, VectorStore, VectorStoreIndex},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create Cohere client
    let cohere_api_key = env::var("COHERE_API_KEY").expect("COHERE_API_KEY not set");
    let cohere_client = Client::new(&cohere_api_key);

    let document_model = cohere_client.embedding_model(EMBED_ENGLISH_V3, "search_document");
    let search_model = cohere_client.embedding_model(EMBED_ENGLISH_V3, "search_query");

    let mut vector_store = InMemoryVectorStore::default();

    let embeddings = EmbeddingsBuilder::new(document_model)
        .simple_document("doc0", "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")
        .simple_document("doc1", "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
        .simple_document("doc2", "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.")
        .build()
        .await?;

    vector_store.add_documents(embeddings).await?;

    let index = vector_store.index(search_model);

    let results = index
        .top_n::<DocumentEmbeddings>("What is a linglingdong?", 1)
        .await?
        .into_iter()
        .map(|(score, id, doc)| (score, id, doc.document))
        .collect::<Vec<_>>();

    println!("Results: {:?}", results);

    Ok(())
}
