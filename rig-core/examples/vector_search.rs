use std::env;

use rig::{
    embeddings::{DocumentEmbeddings, EmbeddingsBuilder},
    providers::openai::Client,
    vector_store::{in_memory_store::InMemoryVectorIndex, VectorStoreIndex},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    let model = openai_client.embedding_model("text-embedding-ada-002");

    let embeddings = EmbeddingsBuilder::new(model.clone())
        .simple_document("doc0", "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")
        .simple_document("doc1", "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
        .simple_document("doc2", "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.")
        .build()
        .await?;

    let index = InMemoryVectorIndex::from_embeddings(model, embeddings).await?;

    let results = index
        .top_n::<DocumentEmbeddings>("What is a linglingdong?", 1)
        .await?
        .into_iter()
        .map(|(score, id, doc)| (score, id, doc.document))
        .collect::<Vec<_>>();

    println!("Results: {:?}", results);

    let id_results = index
        .top_n_ids("What is a linglingdong?", 1)
        .await?
        .into_iter()
        .map(|(score, id)| (score, id))
        .collect::<Vec<_>>();

    println!("ID results: {:?}", id_results);

    Ok(())
}
