//! Passive-RAG chatbot over a PDF, embedded and served by a local Ollama.
//!
//! Both faces of the provider are plain data: an
//! `ollama::functions::EmbeddingConfig` for the chunk/query embeddings and an
//! `ollama::functions::Config` for the chat model, each paired with a shared
//! `HttpRuntime`. `EmbeddingsBuilder` is gone — `embed_documents` batches the
//! PDF chunks through the provider's free `embed` function.

use anyhow::{Context, Result};
use rig::OneOrMany;
use rig::agent::{CompletionCallAction, RequestPatch};
use rig::embeddings::EmbeddingJob;
use rig::hooks::{HookDecision, HookEntry, HookEvent};
use rig::integrations::cli_chatbot::ChatBotBuilder;
use rig::prelude::*;
use rig::providers::ollama;
use rig::{
    Embed, loaders::PdfFileLoader, vector_store::VectorSearchRequest,
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

/// Passive RAG as a hook entry: on every model call, embed the latest user
/// text, search the PDF-chunk store, and inject the hits as per-turn context.
///
/// Hooks are attach-and-forget records — a named `HookEntry` wrapping a
/// closure over owned `HookEvent`s that returns a `HookDecision`; the
/// embedding config, the transport, the store, and the sample count are owned
/// by the hook record and borrowed by each inline invocation future.
fn pdf_rag_hook(
    ecfg: ollama::functions::EmbeddingConfig,
    rt: HttpRuntime,
    store: InMemoryVectorStore,
    samples: u64,
) -> HookEntry {
    HookEntry::with_state("pdf-rag", (ecfg, rt, store, samples), |state, event| {
        Box::pin(async move {
            let HookEvent::BeforeModelCall {
                prompt, history, ..
            } = event
            else {
                return HookDecision::Continue;
            };
            let (ecfg, rt, store, samples) = state;
            let query = prompt
                .rag_text()
                .or_else(|| history.iter().rev().find_map(|message| message.rag_text()));
            let Some(query) = query else {
                return HookDecision::CompletionCall(CompletionCallAction::continue_run());
            };

            // The store only sees pre-embedded requests, so embed the query
            // through the provider's free `embed` function first.
            let embedded = match ollama::functions::embed(ecfg, rt, vec![query]).await {
                Ok(response) => match response.embeddings.into_iter().next() {
                    Some(embedding) => embedding,
                    None => {
                        return HookDecision::CompletionCall(CompletionCallAction::stop(
                            "no embedding returned for the query".to_string(),
                        ));
                    }
                },
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
    let client = ollama::Client::from_env()?;
    let rt = client.http();

    // Load PDFs using Rig's built-in PDF loader
    let documents_dir = std::env::current_dir()?.join("examples/documents");
    let pdf_chunks =
        load_pdf(documents_dir.join("deepseek_r1.pdf")).context("Failed to load pdf documents")?;
    println!("Successfully loaded and chunked PDF documents");

    let ecfg = client.embedding_config("bge-m3");

    let documents: Vec<Document> = pdf_chunks
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| Document {
            id: format!("pdf_document_{i}"),
            content: chunk,
        })
        .collect();

    // `EmbeddingsBuilder` is gone: `embed_documents` chunks to the provider's
    // `max_embedding_documents` and re-associates each document with its
    // embeddings.
    let embeddings = EmbeddingJob::new()
        .documents(documents)
        .for_provider(&ollama::functions::DESCRIPTOR)
        .run(|texts| ollama::functions::embed(&ecfg, &rt, texts))
        .await?;
    println!("Successfully generated embeddings");

    // Create vector store
    let vector_store = InMemoryVectorStore::from_documents(embeddings)?;
    println!("Successfully created vector store");

    // Create RAG agent with the passive-RAG hook
    let rag_agent = client
        .agent("deepseek-r1")
        .preamble("You are a helpful assistant that answers questions based on the provided document context. When answering questions, try to synthesize information from multiple chunks if they're related.")
        .add_hook(pdf_rag_hook(ecfg, rt, vector_store, 1))
        .build();

    println!("Starting CLI chatbot...");

    // Start interactive CLI
    let chatbot = ChatBotBuilder::new(rag_agent).max_turns(10).build();

    chatbot.run().await?;

    Ok(())
}
