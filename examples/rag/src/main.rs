//! Passive RAG over an in-memory vector store, wired in as a hook.
//!
//! Both halves of this example are plain data now. Embedding is an
//! `openai::functions::EmbeddingConfig` plus an [`HttpRuntime`], driven
//! through [`embed_documents`] (the replacement for `EmbeddingsBuilder`);
//! the agent is an `openai::functions::Config` wrapped in
//! [`ProviderConfig`]. The hook captures the embedding config and the
//! transport instead of an embedding model.
use rig::agent::{CompletionCallAction, RequestPatch};
use rig::completion::Document;
use rig::embeddings::default_concurrency;
use rig::hooks::{HookDecision, HookEntry, HookEvent};
use rig::prelude::*;
use rig::providers::openai;
use serde::Serialize;
use std::vec;

// Data to be RAGged.
// A vector search needs to be performed on the `definitions` field, so we derive the `Embed` trait for `WordDefinition`
// and tag that field with `#[embed]`.
#[derive(Embed, Serialize, Clone, Debug, Eq, PartialEq, Default)]
struct WordDefinition {
    id: String,
    word: String,
    #[embed]
    definitions: Vec<String>,
}

/// Passive RAG as a hook entry: on every model call, embed the prompt, search
/// the vector store, and inject the best-matching documents as per-turn
/// context.
///
/// Hooks are attach-and-forget records — a named `HookEntry` wrapping a
/// closure that receives an owned `HookEvent` and returns a `HookDecision`.
/// Anything the closure needs (here the embedding config, the transport, the
/// store, and the sample count) is captured, shared through an `Arc` so the
/// future stays `'static + Send + Sync`.
fn rag_hook(
    embedding_config: openai::functions::EmbeddingConfig,
    rt: HttpRuntime,
    store: InMemoryVectorStore,
    samples: u64,
) -> HookEntry {
    let state = std::sync::Arc::new((embedding_config, rt, store, samples));
    HookEntry::new("rag", move |event| {
        let state = state.clone();
        Box::pin(async move {
            // Only the pre-model-call event is interesting; everything else
            // falls through untouched.
            let HookEvent::BeforeModelCall {
                prompt, history, ..
            } = event
            else {
                return HookDecision::Continue;
            };
            let (embedding_config, rt, store, samples) = state.as_ref();

            // Search with the prompt's text, falling back to the latest
            // textual history message.
            let query = prompt
                .rag_text()
                .or_else(|| history.iter().rev().find_map(|message| message.rag_text()));
            let Some(query) = query else {
                return HookDecision::CompletionCall(CompletionCallAction::continue_run());
            };

            // Embed the query, then run a pre-embedded similarity search.
            let embedded = match openai::functions::embed(embedding_config, rt, vec![query]).await {
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
                    RequestPatch::new().extra_context(hits.into_iter().map(|hit| Document {
                        id: hit.id,
                        text: hit.payload.to_string(),
                        additional_props: Default::default(),
                    })),
                )),
                Err(error) => {
                    HookDecision::CompletionCall(CompletionCallAction::stop(error.to_string()))
                }
            }
        })
    })
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    // Providers are data: one embedding config, one completion config, and a
    // shared HTTP transport.
    let embedding_config =
        openai::functions::EmbeddingConfig::from_env(openai::TEXT_EMBEDDING_ADA_002)?;
    let rt = HttpRuntime::new();
    let max_documents = openai::functions::DESCRIPTOR
        .max_embedding_documents
        .unwrap_or(usize::MAX);

    // Generate embeddings for the definitions of all the documents using the specified embedding config.
    let embeddings = embed_documents(
        vec![
            WordDefinition {
                id: "doc0".to_string(),
                word: "flurbo".to_string(),
                definitions: vec![
                    "1. *flurbo* (name): A flurbo is a green alien that lives on cold planets.".to_string(),
                    "2. *flurbo* (name): A fictional digital currency that originated in the animated series Rick and Morty.".to_string()
                ]
            },
            WordDefinition {
                id: "doc1".to_string(),
                word: "glarb-glarb".to_string(),
                definitions: vec![
                    "1. *glarb-glarb* (noun): A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
                    "2. *glarb-glarb* (noun): A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.".to_string()
                ]
            },
            WordDefinition {
                id: "doc2".to_string(),
                word: "linglingdong".to_string(),
                definitions: vec![
                    "1. *linglingdong* (noun): A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
                    "2. *linglingdong* (noun): A rare, mystical instrument crafted by the ancient monks of the Nebulon Mountain Ranges on the planet Quarm.".to_string()
                ]
            },
        ],
        max_documents,
        default_concurrency(max_documents),
        |texts| openai::functions::embed(&embedding_config, &rt, texts),
    )
    .await?;

    // Create vector store with the embeddings
    let vector_store = InMemoryVectorStore::from_documents(embeddings)?;

    let cfg = openai::functions::Config::from_env(openai::GPT_4O)?;
    let rag_agent = AgentBuilder::new(ProviderConfig::OpenAi(cfg))
        .preamble("
            You are a dictionary assistant here to assist the user in understanding the meaning of words.
            You will find additional non-standard word definitions that could be useful below.
        ")
        // Passive RAG: retrieve one document per model call through the hook.
        .add_hook(rag_hook(embedding_config, rt, vector_store, 1))
        .build();

    // Prompt the agent and print the response
    let response = rag_agent.prompt("What does \"glarb-glarb\" mean?").await?;

    println!("{response}");

    Ok(())
}
