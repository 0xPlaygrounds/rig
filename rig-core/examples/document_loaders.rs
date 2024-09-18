// examples/document_loaders.rs

use rig::{
    providers::openai::Client,
    embeddings::EmbeddingsBuilder,
    document_loaders::PdfLoader,
    completion::Prompt,
    vector_store::{in_memory_store::InMemoryVectorStore, VectorStore},
};
use std::path::{Path, PathBuf};
use std::env;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Print current working directory
    println!("Current working directory: {:?}", env::current_dir()?);

    // Path to the PDF file
    let pdf_path = PathBuf::from("rig-core/examples/sample_data/Moores_law_for_everything.pdf");
    
    // Print absolute path
    println!("Attempting to access file at: {:?}", pdf_path.canonicalize()?);

    // Check if the file exists
    if !pdf_path.exists() {
        eprintln!("Error: The file {} does not exist.", pdf_path.display());
        return Ok(());
    }

    println!("File found successfully!");

    // Initialize OpenAI client
    let openai = Client::from_env();
    let embedding_model = openai.embedding_model("text-embedding-ada-002");

    // Initialize vector store
    let mut vector_store = InMemoryVectorStore::default();

    // Build embeddings
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .add_loader(PdfLoader::new(pdf_path.to_str().unwrap()))
        .build()
        .await?;

    println!("Embeddings created successfully");

    // Add documents to vector store
    vector_store.add_documents(embeddings).await?;

    println!("Documents added to vector store");

    // Create vector store index
    let index = vector_store.index(embedding_model);

    // Create RAG agent
    let rag_agent = openai.agent("gpt-4")
        .preamble("You are a helpful assistant with access to a document about Moore's Law. Use this information to answer questions about the topic.")
        .dynamic_context(5, index)
        .build();

    // Prompt the agent
    let response = rag_agent.prompt("Summarize the key points about Moore's Law based on the document.").await?;
    println!("Agent Response:\n{}", response);

    Ok(())
}