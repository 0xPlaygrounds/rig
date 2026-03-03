// To run this example:
//
// 1. Create a Vectorize index:
//    wrangler vectorize create rig-example --dimensions=1536 --metric=cosine
//
// 2. Set environment variables:
//    export OPENAI_API_KEY=<your-openai-api-key>
//    export CLOUDFLARE_ACCOUNT_ID=<your-account-id>
//    export CLOUDFLARE_API_TOKEN=<your-api-token>
//
// 3. Run the example:
//    cargo run --release --example vectorize_vector_search

use rig::{
    Embed,
    client::{EmbeddingsClient, ProviderClient},
    embeddings::EmbeddingsBuilder,
    providers::openai::{self, Client},
    vector_store::request::VectorSearchRequest,
    vector_store::{InsertDocuments, VectorStoreIndex},
};
use rig_vectorize::VectorizeVectorStore;

#[derive(Embed, serde::Deserialize, serde::Serialize, Debug)]
struct Word {
    id: String,
    #[embed]
    definition: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let openai_client = Client::from_env();
    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_3_SMALL);

    let vector_store = VectorizeVectorStore::new(
        model.clone(),
        std::env::var("CLOUDFLARE_ACCOUNT_ID")?,
        "rig-example",
        std::env::var("CLOUDFLARE_API_TOKEN")?,
    );

    let documents = EmbeddingsBuilder::new(model)
        .document(Word {
            id: "doc-1".to_string(),
            definition: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
        })?
        .document(Word {
            id: "doc-2".to_string(),
            definition: "Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
        })?
        .document(Word {
            id: "doc-3".to_string(),
            definition: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
        })?
        .build()
        .await?;

    vector_store.insert_documents(documents).await?;
    println!("Documents inserted successfully!");

    // Note: Vectorize has eventual consistency, so newly inserted documents
    // may take a few seconds to become queryable
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    let query = "What is a linglingdong?";
    println!("\nSearching for: {}", query);

    let request = VectorSearchRequest::builder()
        .query(query)
        .samples(3)
        .build()?;

    let results = vector_store.top_n::<Word>(request).await?;

    println!("\nResults:");
    for (score, id, word) in results {
        println!("  Score: {:.4}, ID: {}", score, id);
        println!("    Definition: {}", word.definition);
    }

    Ok(())
}
