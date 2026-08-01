//! Dynamic (RAG) tools re-expressed as the hook recipe: `ToolSchema`
//! discovery records are embedded into a vector store per prompt, and a
//! completion-call hook embeds the query, retrieves the best-matching tool
//! names, and narrows the advertised set for the turn with
//! `RequestPatch::active_tools`. These cassettes were recorded against the
//! removed index-backed builder plumbing; the hook recipe replays the exact
//! same request bytes (toolset embedding at build time, one query embedding
//! per model call, and per-turn tool declarations), so they remain the
//! contract this migration satisfies.

use rig::OneOrMany;
use rig::agent::{CompletionCallAction, RequestPatch};
use rig::completion::Message;
use rig::embeddings::ToolSchema;
use rig::hooks::{HookDecision, HookEntry, HookEvent};
use rig::http_runtime::HttpRuntime;
use rig::prelude::*;
use rig::providers::gemini;
use rig::vector_store::VectorSearchRequest;
use rig::vector_store::in_memory_store::InMemoryVectorStore;

use super::super::agent_run_support::{history_has_assistant_tool_call, tool_result_texts};
use super::super::support::with_gemini_cassette;
use super::super::tools_support::{
    CountingAdd, EmbedAdd, EmbedMultiply, EmbedSubtract, FORCE_TOOLS_PREAMBLE,
};
use crate::support::assert_mentions_expected_number;

/// Build an in-memory store over the embeddable tool schemas, keyed by
/// tool name.
async fn build_tool_store(
    client: &gemini::Client,
    schemas: Vec<ToolSchema>,
) -> InMemoryVectorStore {
    let cfg = client.embedding_config(gemini::embedding::EMBEDDING_001);
    let rt = client.http();
    // The schemas are supplied in the classic registration order, so the
    // recorded embedding batch replays deterministically.
    let embeddings = rig::embeddings::embed_documents(
        schemas,
        gemini::functions::DESCRIPTOR
            .max_embedding_documents
            .expect("gemini declares a batch limit"),
        1,
        |texts| gemini::functions::embed(&cfg, &rt, texts),
    )
    .await
    .expect("tool schema embeddings should succeed");

    InMemoryVectorStore::from_documents_with_id_f(embeddings, |tool| tool.name.clone())
        .expect("tool schema store should build")
}

/// The dynamic-tools hook recipe: on every completion call, embed the query,
/// retrieve the best-matching tool names from the store, and advertise those
/// names (plus the always-exposed static ones) for the turn.
fn tool_retrieval_hook(
    embedding_model: (gemini::functions::EmbeddingConfig, HttpRuntime),
    store: InMemoryVectorStore,
    samples: u64,
    always_exposed: Vec<String>,
) -> HookEntry {
    let state = (embedding_model, store, samples, always_exposed);
    HookEntry::with_state("tool-retrieval", state, |state, event| {
        Box::pin(async move {
            let HookEvent::BeforeModelCall {
                prompt, history, ..
            } = event
            else {
                return HookDecision::Continue;
            };
            let (embedding_model, store, samples, always_exposed) = state;
            let query = prompt
                .rag_text()
                .or_else(|| history.iter().rev().find_map(|message| message.rag_text()));
            let Some(query) = query else {
                return HookDecision::CompletionCall(CompletionCallAction::continue_run());
            };

            let (embedding_cfg, embedding_rt) = embedding_model;
            let embedded =
                match gemini::functions::embed(embedding_cfg, embedding_rt, vec![query.clone()])
                    .await
                    .map(|response| response.embeddings)
                {
                    Ok(mut embeddings) if !embeddings.is_empty() => embeddings.remove(0),
                    Ok(_) => {
                        return HookDecision::CompletionCall(CompletionCallAction::stop(
                            "gemini returned no embedding for the retrieval query".to_string(),
                        ));
                    }
                    Err(error) => {
                        return HookDecision::CompletionCall(CompletionCallAction::stop(
                            error.to_string(),
                        ));
                    }
                };
            let request = VectorSearchRequest::new(OneOrMany::one(embedded), *samples);
            match store.top_n_ids(request).await {
                Ok(hits) => HookDecision::CompletionCall(CompletionCallAction::patch(
                    RequestPatch::new().active_tools(
                        hits.into_iter()
                            .map(|(_score, name)| name)
                            .chain(always_exposed.iter().cloned()),
                    ),
                )),
                Err(error) => {
                    HookDecision::CompletionCall(CompletionCallAction::stop(error.to_string()))
                }
            }
        })
    })
}

#[tokio::test]
async fn dynamic_tool_retrieved_and_merged_with_static() {
    let add = CountingAdd::default();
    let subtract = EmbedSubtract::default();
    let subtract_counter = subtract.counter.clone();

    with_gemini_cassette(
        "dynamic_tools/dynamic_tool_retrieved_and_merged_with_static",
        |client| async move {
            let schemas = vec![EmbedSubtract::discovery(), EmbedMultiply::discovery()];
            let store = build_tool_store(&client, schemas).await;
            let embedding_model = (
                client.embedding_config(gemini::embedding::EMBEDDING_001),
                client.http(),
            );

            // Retrieval candidates register before the static tool so the
            // per-turn declarations keep the recorded retrieved-first order.
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(subtract)
                .tool(EmbedMultiply::default())
                .tool(add)
                .add_hook(tool_retrieval_hook(
                    embedding_model,
                    store,
                    1,
                    vec![CountingAdd::NAME.to_string()],
                ))
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
            let schemas = vec![EmbedAdd::discovery(), EmbedSubtract::discovery()];
            let store = build_tool_store(&client, schemas).await;
            let embedding_model = (
                client.embedding_config(gemini::embedding::EMBEDDING_001),
                client.http(),
            );

            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .tool(EmbedSubtract::default())
                .add_hook(tool_retrieval_hook(embedding_model, store, 1, Vec::new()))
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
            let schemas = vec![
                EmbedAdd::discovery(),
                EmbedSubtract::discovery(),
                EmbedMultiply::discovery(),
            ];
            let store = build_tool_store(&client, schemas).await;
            let embedding_model = (
                client.embedding_config(gemini::embedding::EMBEDDING_001),
                client.http(),
            );

            let query = gemini::functions::embed(
                &embedding_model.0,
                &embedding_model.1,
                vec!["Multiply two numbers together to get their product.".to_string()],
            )
            .await
            .expect("query embedding should succeed")
            .embeddings
            .remove(0);
            let request = VectorSearchRequest::new(OneOrMany::one(query), 2);
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
