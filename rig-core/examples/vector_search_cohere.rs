use std::env;

use rig::{
    embeddings::{builder::DocumentEmbeddings, builder::EmbeddingsBuilder},
    providers::cohere::{Client, EMBED_ENGLISH_V3},
    vector_store::{in_memory_store::InMemoryVectorStore, VectorStoreIndex},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create Cohere client
    let cohere_api_key = env::var("COHERE_API_KEY").expect("COHERE_API_KEY not set");
    let cohere_client = Client::new(&cohere_api_key);

    let document_model = cohere_client.embedding_model(EMBED_ENGLISH_V3, "search_document");
    let search_model = cohere_client.embedding_model(EMBED_ENGLISH_V3, "search_query");

    let embeddings = EmbeddingsBuilder::new(document_model)
        .simple_document("doc0", "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")
        .simple_document("doc1", "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
        .simple_document("doc2", "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.")
        .build()
        .await?;

    let index = InMemoryVectorStore::default()
        .add_documents(
            embeddings
                .into_iter()
                .map(
                    |DocumentEmbeddings {
                         id,
                         document,
                         embeddings,
                     }| { (id, document, embeddings) },
                )
                .collect(),
        )?
        .index(search_model);

    let results = index
        .top_n::<String>("What is a linglingdong?", 1)
        .await?
        .into_iter()
        .map(|(score, id, doc)| (score, id, doc))
        .collect::<Vec<_>>();

    println!("Results: {:?}", results);

    Ok(())
}
