use anyhow::{Context, Result};
use rig::integrations::cli_chatbot::ChatBotBuilder;
use rig::prelude::*;
use rig::{
    Embed, embeddings::EmbeddingsBuilder, loaders::PdfFileLoader, providers::openai,
    vector_store::in_memory_store::InMemoryVectorStore,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Embed, Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
struct Document {
    id: String,
    #[embed]
    content: String,
}

fn load_pdf(path: PathBuf) -> Result<Vec<String>> {
    const CHUNK_SIZE: usize = 2000;
    let content_chunks = PdfFileLoader::with_glob(path.to_str().context("Invalid path")?)?
        .read()
        .into_iter()
        .filter_map(|result| {
            result
                .map_err(|e| {
                    eprintln!("Error reading PDF content: {e}");
                    e
                })
                .ok()
        })
        .flat_map(|content| {
            let mut chunks = Vec::new();
            let mut current = String::new();
            for word in content.split_whitespace() {
                if current.len() + word.len() + 1 > CHUNK_SIZE && !current.is_empty() {
                    chunks.push(std::mem::take(&mut current).trim().to_string());
                }
                current.push_str(word);
                current.push(' ');
            }
            if !current.is_empty() {
                chunks.push(current.trim().to_string());
            }
            chunks
        })
        .collect::<Vec<_>>();
    if content_chunks.is_empty() {
        anyhow::bail!("No content found in PDF file: {}", path.display());
    }
    Ok(content_chunks)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize Ollama client
    let client = openai::Client::builder("ollama")
        .base_url("http://localhost:11434/v1")
        .build();

    // Load PDFs using Rig's built-in PDF loader
    let documents_dir = std::env::current_dir()?.join("rig-core/examples/documents");
    let pdf_chunks =
        load_pdf(documents_dir.join("deepseek_r1.pdf")).context("Failed to load pdf documents")?;
    println!("Successfully loaded and chunked PDF documents");

    // Create embedding model
    let model = client.embedding_model("bge-m3");

    // Create embeddings builder
    let mut builder = EmbeddingsBuilder::new(model.clone());

    // Add chunks from pdf documents
    for (i, chunk) in pdf_chunks.into_iter().enumerate() {
        builder = builder.document(Document {
            id: format!("pdf_document_{i}"),
            content: chunk,
        })?;
    }

    // Build embeddings
    let embeddings = builder.build().await?;
    println!("Successfully generated embeddings");

    // Create vector store and index
    let vector_store = InMemoryVectorStore::from_documents(embeddings);
    let index = vector_store.index(model);
    println!("Successfully created vector store and index");

    // Create RAG agent
    let rag_agent = client
        .agent("deepseek-r1")
        .preamble("You are a helpful assistant that answers questions based on the provided document context. When answering questions, try to synthesize information from multiple chunks if they're related.")
        .dynamic_context(1, index)
        .build();

    println!("Starting CLI chatbot...");

    // Start interactive CLI
    let chatbot = ChatBotBuilder::new()
        .agent(rag_agent)
        .multi_turn_depth(10)
        .build();

    chatbot.run().await?;

    Ok(())
}
