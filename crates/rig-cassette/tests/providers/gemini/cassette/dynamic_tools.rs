//! Dynamic (RAG) tools: `ToolEmbedding` toolsets sampled from a vector store
//! per prompt and merged with static tools. This capability has no rmcp
//! equivalent today, so these cassettes are the contract any migration has to
//! consciously satisfy or supersede.
//!
//! Each cassette records the Gemini embedding calls (toolset embedding at
//! build time, query embedding at prompt time) alongside the completion
//! turns.

use rig::embeddings::EmbeddingsBuilder;
use rig::providers::gemini::{self};
use rig::tool::ToolSet;
use rig::vector_store::in_memory_store::InMemoryVectorStore;
use rig_test_support::cassette_models::GeminiModels;

use super::super::support::with_gemini_cassette;
use super::super::tools_support::{EmbedAdd, EmbedMultiply, EmbedSubtract, FORCE_TOOLS_PREAMBLE};

/// Build an in-memory index over the toolset's embeddable schemas.
async fn build_tool_index(
    client: &GeminiModels,
    toolset: &ToolSet,
) -> rig::vector_store::in_memory_store::InMemoryVectorIndex<rig::embeddings::ToolSchema> {
    let embedding_model = client.embedding(gemini::embedding::EMBEDDING_001, None);
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
async fn sample_caps_retrieved_definitions() {
    with_gemini_cassette(
        "dynamic_tools/sample_caps_retrieved_definitions",
        |client| async move {
            let mut toolset = ToolSet::default();
            toolset
                .add_retrieved_tool(EmbedAdd::default())
                .expect("the tool context serializes");
            toolset
                .add_retrieved_tool(EmbedSubtract::default())
                .expect("the tool context serializes");
            toolset
                .add_retrieved_tool(EmbedMultiply::default())
                .expect("the tool context serializes");
            let index = build_tool_index(&client, &toolset).await;

            let agent =
                rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_2_5_FLASH))
                    .preamble(FORCE_TOOLS_PREAMBLE)
                    .temperature(0.0)
                    .retrieved_tools(2, index, toolset)
                    .build();

            let defs = agent
                .tool_definitions(Some(
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
