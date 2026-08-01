//! RAG over Bedrock: embed a corpus once, then inject vector-search hits as
//! per-turn context from a hook.
//!
//! `EmbeddingsBuilder` and the embedding-model trait are gone; the corpus is
//! embedded with `rig_core::embeddings::embed_documents` over
//! `rig_bedrock::functions::embed`, and the hook re-uses the same AWS client
//! to embed each query.
use rig_core::OneOrMany;
use std::vec;

use rig_agent::agent::hook::{CompletionCallAction, RequestPatch};
use rig_agent::client::AgentClientExt;
use rig_agent::hooks::{HookDecision, HookEntry, HookEvent};
use rig_bedrock::embedding::AMAZON_TITAN_EMBED_TEXT_V2_0;
use rig_bedrock::functions;
use rig_bedrock::{aws_sdk_bedrockruntime, completion::AMAZON_NOVA_LITE};
use rig_core::completion::Document;
use rig_core::embeddings::EmbeddingJob;
use rig_core::vector_store::VectorSearchRequest;
use rig_core::vector_store::in_memory_store::InMemoryVectorStore;
use serde::Serialize;
use tracing::info;

// Data to be RAG-ed.
// A vector search needs to be performed on the `definitions` field, so we derive the `Embed` trait for `WordDefinition`
// and tag that field with `#[embed]`.
#[derive(rig_derive::Embed, Serialize, Clone, Debug, Eq, PartialEq, Default)]
struct WordDefinition {
    id: String,
    word: String,
    #[embed]
    definitions: Vec<String>,
}

/// Passive RAG as a hook entry: on every model call, embed the prompt, search
/// the vector store, and inject the best matches as per-turn context.
///
/// Hooks are attach-and-forget records — a named `HookEntry` wrapping a
/// closure over owned `HookEvent`s that returns a `HookDecision`; the
/// AWS client, the embedding config, the store, and the sample count are
/// captured behind an `Arc` so the returned future stays
/// `'static + Send + Sync`.
fn rag_hook(
    client: aws_sdk_bedrockruntime::Client,
    embedding_config: functions::EmbeddingConfig,
    store: InMemoryVectorStore,
    samples: u64,
) -> HookEntry {
    let state = std::sync::Arc::new((client, embedding_config, store, samples));
    HookEntry::new("rag", move |event| {
        let state = state.clone();
        Box::pin(async move {
            let HookEvent::BeforeModelCall {
                prompt, history, ..
            } = event
            else {
                return HookDecision::Continue;
            };
            let (client, embedding_config, store, samples) = state.as_ref();
            let query = prompt
                .rag_text()
                .or_else(|| history.iter().rev().find_map(|message| message.rag_text()));
            let Some(query) = query else {
                return HookDecision::CompletionCall(CompletionCallAction::continue_run());
            };
            let embedded = match functions::embed(
                client,
                &embedding_config.model,
                embedding_config.ndims,
                vec![query],
            )
            .await
            {
                Ok(response) => match response.embeddings.into_iter().next() {
                    Some(embedding) => embedding,
                    None => {
                        return HookDecision::CompletionCall(CompletionCallAction::stop(
                            "bedrock returned no embedding for the RAG query",
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
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let client = rig_bedrock::Client::builder()
        .default_region()
        .build()
        .await;
    let embedding_config = client
        .embedding_config(AMAZON_TITAN_EMBED_TEXT_V2_0)
        .with_ndims(256);
    let aws_client = client.get_inner().await;

    // Generate embeddings for the definitions of all the documents using the specified embedding model.
    let embeddings = EmbeddingJob::new()
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
        ])
        .for_provider(&functions::DESCRIPTOR)
        .run(|texts| functions::embed(&aws_client, &embedding_config.model, embedding_config.ndims, texts))
    .await?;

    // Create vector store with the embeddings
    let vector_store = InMemoryVectorStore::from_documents(embeddings)?;

    let rag_agent = client
        .agent(AMAZON_NOVA_LITE)
        .preamble("
            You are a dictionary assistant here to assist the user in understanding the meaning of words.
            You will find additional non-standard word definitions that could be useful below.
        ")
        .add_hook(rag_hook(
            aws_client.clone(),
            embedding_config,
            vector_store,
            1,
        ))
        .build();

    // Prompt the agent and print the response
    let response = rag_agent.prompt("What does \"glarb-glarb\" mean?").await?;

    info!("{}", response);

    Ok(())
}
