//! Dynamic (RAG) tools re-expressed as the hook recipe: `ToolEmbedding`
//! toolsets are embedded into a vector store per prompt, and a
//! completion-call hook embeds the query, retrieves the best-matching tool
//! names, and narrows the advertised set for the turn with
//! `RequestPatch::active_tools`. These cassettes were recorded against the
//! removed index-backed builder plumbing; the hook recipe replays the exact
//! same request bytes (toolset embedding at build time, one query embedding
//! per model call, and per-turn tool declarations), so they remain the
//! contract this migration satisfies.

use rig::agent::{AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, RequestPatch};
use rig::completion::{Chat, Message};
use rig::embeddings::EmbeddingsBuilder;
use rig::prelude::*;
use rig::providers::gemini;
use rig::tool::ToolSet;
use rig::vector_store::VectorSearchRequest;
use rig::vector_store::in_memory_store::InMemoryVectorStore;

use super::super::agent_run_support::{history_has_assistant_tool_call, tool_result_texts};
use super::super::support::with_gemini_cassette;
use super::super::tools_support::{
    CountingAdd, EmbedAdd, EmbedMultiply, EmbedSubtract, FORCE_TOOLS_PREAMBLE,
};
use crate::support::assert_mentions_expected_number;

/// Build an in-memory store over the toolset's embeddable schemas, keyed by
/// tool name.
async fn build_tool_store(client: &gemini::Client, toolset: &ToolSet) -> InMemoryVectorStore {
    let embedding_model = client.embedding_model(gemini::embedding::EMBEDDING_001);
    // ToolSet::schemas() returns registration order, so the recorded
    // embedding batch replays deterministically.
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(toolset.schemas().expect("tool schemas should build"))
        .expect("documents should be added")
        .build()
        .await
        .expect("tool schema embeddings should succeed");

    InMemoryVectorStore::from_documents_with_id_f(embeddings, |tool| tool.name.clone())
        .expect("tool schema store should build")
}

/// The dynamic-tools hook recipe: on every completion call, embed the query,
/// retrieve the best-matching tool names from the store, and advertise those
/// names (plus the always-exposed static ones) for the turn.
struct ToolRetrievalHook {
    embedding_model: gemini::embedding::EmbeddingModel,
    store: InMemoryVectorStore,
    samples: u64,
    always_exposed: Vec<String>,
}

impl AgentHook for ToolRetrievalHook {
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
        match self.store.top_n_ids(request).await {
            Ok(hits) => CompletionCallAction::patch(
                RequestPatch::new().active_tools(
                    hits.into_iter()
                        .map(|(_score, name)| name)
                        .chain(self.always_exposed.iter().cloned()),
                ),
            ),
            Err(error) => CompletionCallAction::stop(error.to_string()),
        }
    }
}

#[tokio::test]
async fn dynamic_tool_retrieved_and_merged_with_static() {
    let add = CountingAdd::default();
    let subtract = EmbedSubtract::default();
    let subtract_counter = subtract.counter.clone();

    with_gemini_cassette(
        "dynamic_tools/dynamic_tool_retrieved_and_merged_with_static",
        |client| async move {
            let toolset = ToolSet::builder()
                .retrieved_tool(EmbedSubtract::default())
                .retrieved_tool(EmbedMultiply::default())
                .build();
            let store = build_tool_store(&client, &toolset).await;
            let embedding_model = client.embedding_model(gemini::embedding::EMBEDDING_001);

            // Retrieval candidates register before the static tool so the
            // per-turn declarations keep the recorded retrieved-first order.
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(subtract)
                .tool(EmbedMultiply::default())
                .tool(add)
                .add_hook(ToolRetrievalHook {
                    embedding_model,
                    store,
                    samples: 1,
                    always_exposed: vec![CountingAdd::NAME.to_string()],
                })
                .default_max_turns(3)
                .build();

            let mut history = Vec::<Message>::new();
            let response = agent
                .chat("Subtract 8 from 50 to get their difference.", &mut history)
                .await
                .expect("dynamic tool prompt should succeed");

            assert_mentions_expected_number(&response, 42);
            assert!(
                history_has_assistant_tool_call(&history, "subtract"),
                "the retrieved dynamic tool should be called: {history:?}"
            );
            assert_eq!(
                subtract_counter.count(),
                1,
                "the retrieved dynamic tool should execute exactly once"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn dynamic_only_agent_retrieves_tool_per_prompt() {
    let add = EmbedAdd::default();
    let add_counter = add.counter.clone();

    with_gemini_cassette(
        "dynamic_tools/dynamic_only_agent_retrieves_tool_per_prompt",
        |client| async move {
            let toolset = ToolSet::builder()
                .retrieved_tool(EmbedAdd::default())
                .retrieved_tool(EmbedSubtract::default())
                .build();
            let store = build_tool_store(&client, &toolset).await;
            let embedding_model = client.embedding_model(gemini::embedding::EMBEDDING_001);

            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .tool(EmbedSubtract::default())
                .add_hook(ToolRetrievalHook {
                    embedding_model,
                    store,
                    samples: 1,
                    always_exposed: Vec::new(),
                })
                .default_max_turns(3)
                .build();

            let mut history = Vec::<Message>::new();
            let response = agent
                .chat("Add 19 and 23 together to get their sum.", &mut history)
                .await
                .expect("dynamic-only tool prompt should succeed");

            assert_mentions_expected_number(&response, 42);
            let texts: Vec<String> = history.iter().flat_map(tool_result_texts).collect();
            assert_eq!(texts, vec!["42".to_string()]);
            assert_eq!(
                add_counter.count(),
                1,
                "the retrieved dynamic tool should execute exactly once"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn sample_caps_retrieved_definitions() {
    with_gemini_cassette(
        "dynamic_tools/sample_caps_retrieved_definitions",
        |client| async move {
            let toolset = ToolSet::builder()
                .retrieved_tool(EmbedAdd::default())
                .retrieved_tool(EmbedSubtract::default())
                .retrieved_tool(EmbedMultiply::default())
                .build();
            let store = build_tool_store(&client, &toolset).await;
            let embedding_model = client.embedding_model(gemini::embedding::EMBEDDING_001);

            let query = embedding_model
                .embed_text("Multiply two numbers together to get their product.")
                .await
                .expect("query embedding should succeed");
            let request = VectorSearchRequest::builder()
                .query(query)
                .samples(2)
                .build();
            let hits = store
                .top_n_ids(request)
                .await
                .expect("dynamic retrieval should resolve");

            assert_eq!(
                hits.len(),
                2,
                "the sample size should cap how many dynamic tools are retrieved: {:?}",
                hits.iter()
                    .map(|(_, name)| name.as_str())
                    .collect::<Vec<_>>()
            );
            assert!(
                hits.iter().any(|(_, name)| name == "multiply"),
                "the best-matching tool should be retrieved: {:?}",
                hits.iter()
                    .map(|(_, name)| name.as_str())
                    .collect::<Vec<_>>()
            );
        },
    )
    .await;
}
