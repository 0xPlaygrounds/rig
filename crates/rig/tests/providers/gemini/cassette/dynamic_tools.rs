//! Dynamic (RAG) tools: `ToolEmbedding` toolsets sampled from a vector store
//! per prompt and merged with static tools. This capability has no rmcp
//! equivalent today, so these cassettes are the contract any migration has to
//! consciously satisfy or supersede.
//!
//! Each cassette records the Gemini embedding calls (toolset embedding at
//! build time, query embedding at prompt time) alongside the completion
//! turns.

use rig::client::{CompletionClient, EmbeddingsClient};
use rig::completion::{Chat, Message};
use rig::embeddings::EmbeddingsBuilder;
use rig::providers::gemini;
use rig::tool::ToolSet;
use rig::vector_store::in_memory_store::InMemoryVectorStore;

use super::super::agent_run_support::{history_has_assistant_tool_call, tool_result_texts};
use super::super::support::with_gemini_cassette;
use super::super::tools_support::{
    CountingAdd, EmbedAdd, EmbedMultiply, EmbedSubtract, FORCE_TOOLS_PREAMBLE,
};
use crate::support::assert_mentions_expected_number;

/// Build an in-memory index over the toolset's embeddable schemas.
async fn build_tool_index(
    client: &gemini::Client,
    toolset: &ToolSet,
) -> rig::vector_store::in_memory_store::InMemoryVectorIndex<
    gemini::embedding::EmbeddingModel,
    rig::embeddings::ToolSchema,
> {
    let embedding_model = client.embedding_model(gemini::embedding::EMBEDDING_001);
    // ToolSet::schemas() returns registration order, so the recorded
    // embedding batch replays deterministically.
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(toolset.schemas().expect("tool schemas should build"))
        .expect("documents should be added")
        .build()
        .await
        .expect("tool schema embeddings should succeed");

    let vector_store =
        InMemoryVectorStore::from_documents_with_id_f(embeddings, |tool| tool.name.clone());
    vector_store.index(embedding_model)
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
                .dynamic_tool(subtract)
                .dynamic_tool(EmbedMultiply::default())
                .build();
            let index = build_tool_index(&client, &toolset).await;

            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .dynamic_tools(1, index, toolset)
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
                .dynamic_tool(add)
                .dynamic_tool(EmbedSubtract::default())
                .build();
            let index = build_tool_index(&client, &toolset).await;

            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .dynamic_tools(1, index, toolset)
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
                .dynamic_tool(EmbedAdd::default())
                .dynamic_tool(EmbedSubtract::default())
                .dynamic_tool(EmbedMultiply::default())
                .build();
            let index = build_tool_index(&client, &toolset).await;

            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .dynamic_tools(2, index, toolset)
                .build();

            let defs = agent
                .tool_server_handle
                .get_tool_defs(Some(
                    "Multiply two numbers together to get their product.".to_string(),
                ))
                .await
                .expect("dynamic definitions should resolve");

            assert_eq!(
                defs.len(),
                2,
                "the sample size should cap how many dynamic definitions are returned: {:?}",
                defs.iter().map(|def| def.name.as_str()).collect::<Vec<_>>()
            );
            assert!(
                defs.iter().any(|def| def.name == "multiply"),
                "the best-matching tool should be retrieved: {:?}",
                defs.iter().map(|def| def.name.as_str()).collect::<Vec<_>>()
            );
        },
    )
    .await;
}
