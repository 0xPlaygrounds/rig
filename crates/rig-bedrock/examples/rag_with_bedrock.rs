use std::vec;

use rig_agent::agent::hook::{
    AgentHook, CompletionCall as CompletionCallEvent, CompletionCallAction, HookContext,
    RequestPatch,
};
use rig_agent::{agent::AgentBuilder, prelude::*, provider::ProviderConfig};
use rig_bedrock::client::Client;
use rig_bedrock::completion::AMAZON_NOVA_LITE;
use rig_bedrock::embedding::AMAZON_TITAN_EMBED_TEXT_V2_0;
use rig_core::client::{EmbeddingsClient, ProviderClient};
use rig_core::completion::Document;
use rig_core::vector_store::VectorSearchRequest;
use rig_core::{embeddings::EmbeddingsBuilder, vector_store::in_memory_store::InMemoryVectorStore};
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

/// Passive RAG as a hook: on every model call, embed the prompt, search the
/// vector store, and inject the best matches as per-turn context.
struct RagHook {
    embedding_model: rig_bedrock::embedding::EmbeddingModel,
    store: InMemoryVectorStore,
    samples: u64,
}

impl AgentHook for RagHook {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let query = event.prompt.rag_text().or_else(|| {
            event
                .history
                .iter()
                .rev()
                .find_map(|message| message.rag_text())
        });
        let Some(query) = query else {
            return CompletionCallAction::continue_run();
        };
        let embedded = match self.embedding_model.embed_text(&query).await {
            Ok(embedding) => embedding,
            Err(error) => return CompletionCallAction::stop(error.to_string()),
        };
        let request = VectorSearchRequest::builder()
            .query(embedded)
            .samples(self.samples)
            .build();
        match self.store.top_n(request).await {
            Ok(hits) => CompletionCallAction::patch(RequestPatch::new().extra_context(
                hits.into_iter().map(|hit| Document {
                    id: hit.id,
                    text: hit.payload.to_string(),
                    additional_props: Default::default(),
                }),
            )),
            Err(error) => CompletionCallAction::stop(error.to_string()),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let client = Client::from_env()?;
    let embedding_model = client.embedding_model_with_ndims(AMAZON_TITAN_EMBED_TEXT_V2_0, 256);

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

    // The classic bedrock client is not expressible as portable provider
    // configuration; build the agent from a bedrock provider config directly
    // (default AWS credential chain and region, like `Client::from_env`).
    let rag_agent = AgentBuilder::new(ProviderConfig::Bedrock(
        rig_bedrock::functions::Config::new(AMAZON_NOVA_LITE),
    ))
        .preamble("
            You are a dictionary assistant here to assist the user in understanding the meaning of words.
            You will find additional non-standard word definitions that could be useful below.
        ")
        .add_hook(RagHook {
            embedding_model,
            store: vector_store,
            samples: 1,
        })
        .build();

    // Prompt the agent and print the response
    let response = rag_agent.prompt("What does \"glarb-glarb\" mean?").await?;

    info!("{}", response);

    Ok(())
}
