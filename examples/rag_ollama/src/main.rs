use rig::OneOrMany;
use rig::agent::{CompletionCallAction, RequestPatch};
use rig::client::Nothing;
use rig::completion::Document;
use rig::hooks::{HookDecision, HookEntry, HookEvent};
use rig::prelude::*;
use rig::providers::ollama;
use rig::{
    Embed, embeddings::EmbeddingsBuilder, providers::ollama::Client,
    vector_store::VectorSearchRequest, vector_store::in_memory_store::InMemoryVectorStore,
};
use serde::Serialize;

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
/// Anything the closure needs (here the embedding model, the store, and the
/// sample count) is captured, shared through an `Arc` so the future stays
/// `'static + Send + Sync`.
fn rag_hook(
    embedding_model: ollama::EmbeddingModel,
    store: InMemoryVectorStore,
    samples: u64,
) -> HookEntry {
    let state = std::sync::Arc::new((embedding_model, store, samples));
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
            let (embedding_model, store, samples) = state.as_ref();

            // Search with the prompt's text, falling back to the latest
            // textual history message.
            let query = prompt
                .rag_text()
                .or_else(|| history.iter().rev().find_map(|message| message.rag_text()));
            let Some(query) = query else {
                return HookDecision::CompletionCall(CompletionCallAction::continue_run());
            };

            // Embed the query, then run a pre-embedded similarity search.
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

    // Create ollama client
    let ollama_client = Client::from_val(Nothing.into())?;
    let embedding_model = ollama_client.embedding_model("nomic-embed-text");

    // Generate embeddings for the definitions of all the documents using the specified embedding model.
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(vec![
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
        ])?
        .build()
        .await?;

    // Create vector store with the embeddings
    let vector_store = InMemoryVectorStore::from_documents(embeddings)?;

    let rag_agent = ollama_client.agent("qwen2.5:14b")
        .preamble("
            You are a dictionary assistant here to assist the user in understanding the meaning of words.
            You will find additional non-standard word definitions that could be useful below.
        ")
        // Passive RAG: retrieve one document per model call through the hook.
        .add_hook(rag_hook(embedding_model, vector_store, 1))
        .build();

    // Prompt the agent and print the response
    let response = rag_agent.prompt("What does \"glarb-glarb\" mean?").await?;

    println!("{response}");

    Ok(())
}
