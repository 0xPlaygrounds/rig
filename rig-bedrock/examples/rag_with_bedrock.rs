use rig::{
    completion::Prompt, embeddings::EmbeddingsBuilder,
    vector_store::in_memory_store::InMemoryVectorStore,
};
use rig_bedrock::{
    client::ClientBuilder, completion::AMAZON_NOVA_LITE, embedding::AMAZON_TITAN_EMBED_TEXT_V2_0,
};
use rig_shared::fixtures::word_definition::WordDefinition;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let client = ClientBuilder::new().build().await;
    let embedding_model = client.embedding_model(AMAZON_TITAN_EMBED_TEXT_V2_0, 256);

    // Generate embeddings for the definitions of all the documents using the specified embedding model.
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(WordDefinition::sample())?
        .build()
        .await?;

    // Create vector store with the embeddings
    let vector_store = InMemoryVectorStore::from_documents(embeddings);

    // Create vector store index
    let index = vector_store.index(embedding_model);

    let rag_agent = client.agent(AMAZON_NOVA_LITE)
        .preamble("
            You are a dictionary assistant here to assist the user in understanding the meaning of words.
            You will find additional non-standard word definitions that could be useful below.
        ")
        .dynamic_context(1, index)
        .build();

    // Prompt the agent and print the response
    let response = rag_agent.prompt("What does \"glarb-glarb\" mean?").await?;

    info!("{}", response);

    Ok(())
}
