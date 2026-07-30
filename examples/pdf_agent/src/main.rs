use anyhow::{Context, Result};
use rig::OneOrMany;
use rig::agent::{CompletionCallAction, RequestPatch};
use rig::client::Nothing;
use rig::hooks::{HookDecision, HookEntry, HookEvent};
use rig::integrations::cli_chatbot::ChatBotBuilder;
use rig::prelude::*;
use rig::providers::ollama;
use rig::{
    Embed, embeddings::EmbeddingsBuilder, loaders::PdfFileLoader,
    vector_store::VectorSearchRequest, vector_store::in_memory_store::InMemoryVectorStore,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Embed, Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
struct Document {
    id: String,
    #[embed]
    content: String,
}

/// Passive RAG as a hook entry: on every model call, embed the latest user
/// text, search the PDF-chunk store, and inject the hits as per-turn context.
///
/// Hooks are attach-and-forget records — a named `HookEntry` wrapping a
/// closure over owned `HookEvent`s that returns a `HookDecision`; the
/// embedding model, the store, and the sample count are captured behind an
/// `Arc` so the returned future stays `'static + Send + Sync`.
fn pdf_rag_hook(
    embedding_model: ollama::EmbeddingModel,
    store: InMemoryVectorStore,
    samples: u64,
) -> HookEntry {
    let state = std::sync::Arc::new((embedding_model, store, samples));
    HookEntry::new("pdf-rag", move |event| {
        let state = state.clone();
        Box::pin(async move {
            let HookEvent::BeforeModelCall {
                prompt, history, ..
            } = event
            else {
                return HookDecision::Continue;
            };
            let (embedding_model, store, samples) = state.as_ref();
            let query = prompt
                .rag_text()
                .or_else(|| history.iter().rev().find_map(|message| message.rag_text()));
            let Some(query) = query else {
                return HookDecision::CompletionCall(CompletionCallAction::continue_run());
            };

            let embedded = match embedding_model.embed_text(&query).await {
                Ok(embedding) => embedding,
                Err(error) => {
                    return HookDecision::CompletionCall(CompletionCallAction::stop(
                        error.to_string(),
                    ));
                }
            };
            let request = VectorSearchRequest::new(OneOrMany::one(embedded), *samples);
            match store.top_n(request).await {
                Ok(hits) => HookDecision::CompletionCall(CompletionCallAction::patch(
                    RequestPatch::new().extra_context(hits.into_iter().map(|hit| {
                        rig::completion::Document {
                            id: hit.id,
                            text: hit.payload.to_string(),
                            additional_props: Default::default(),
                        }
                    })),
                )),
                Err(error) => {
                    HookDecision::CompletionCall(CompletionCallAction::stop(error.to_string()))
                }
            }
        })
    })
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
    // because Ollama is local and does not require an api key, we pass in `Nothing`
    let client = ollama::Client::builder()
        .api_key(Nothing)
        .base_url("http://localhost:11434/v1")
        .build()?;

    // Load PDFs using Rig's built-in PDF loader
    let documents_dir = std::env::current_dir()?.join("examples/documents");
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

    // Create vector store
    let vector_store = InMemoryVectorStore::from_documents(embeddings)?;
    println!("Successfully created vector store");

    // Create RAG agent with the passive-RAG hook
    let rag_agent = client
        .agent("deepseek-r1")
        .preamble("You are a helpful assistant that answers questions based on the provided document context. When answering questions, try to synthesize information from multiple chunks if they're related.")
        .add_hook(pdf_rag_hook(model, vector_store, 1))
        .build();

    println!("Starting CLI chatbot...");

    // Start interactive CLI
    let chatbot = ChatBotBuilder::new(rag_agent).max_turns(10).build();

    chatbot.run().await?;

    Ok(())
}
