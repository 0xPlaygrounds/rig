//! Reasoning across a session boundary on OpenRouter: see
//! `rig_test_support::history_survival::sessions`.

use super::super::support::{BoundOpenRouter, with_openrouter_cassette};
use rig::completion::CompletionModel;

use crate::history_survival::sessions::{self, Cell};

fn params() -> Option<serde_json::Value> {
    Some(serde_json::json!({ "reasoning": { "max_tokens": 1024 }, "include_reasoning": true }))
}

const CELL: Cell = Cell {
    provider: "openrouter",
    params,
    max_tokens: 4096,
    expect: &["signature"],
};

fn models(
    client: BoundOpenRouter,
) -> (
    impl CompletionModel + Clone + 'static,
    impl CompletionModel + Clone + 'static,
    impl CompletionModel + Clone + 'static,
) {
    (
        client.completion("anthropic/claude-haiku-4.5"),
        client.completion("anthropic/claude-haiku-4.5"),
        client.completion("anthropic/claude-sonnet-4.6"),
    )
}

#[tokio::test]
async fn same_model() {
    const SCENARIO: &str = "session_matrix/same_model";
    with_openrouter_cassette("session_matrix/same_model", |client| async move {
        let (first, second, _) = models(client);
        sessions::run(first, second, CELL).await;
    })
    .await;
    sessions::assert_recorded(CELL, SCENARIO);
}

/// The loaded history continues on another model of the same provider.
#[tokio::test]
async fn other_model() {
    const SCENARIO: &str = "session_matrix/other_model";
    with_openrouter_cassette("session_matrix/other_model", |client| async move {
        let (first, _, other) = models(client);
        sessions::run(first, other, CELL).await;
    })
    .await;
    sessions::assert_recorded(CELL, SCENARIO);
}

/// The continuation is checkpointed, restored into a fresh world, and sent
/// by the restored world's handler for another model of the same provider.
#[tokio::test]
async fn checkpoint_other_model() {
    const SCENARIO: &str = "session_matrix/checkpoint_other_model";
    with_openrouter_cassette(
        "session_matrix/checkpoint_other_model",
        |client| async move {
            let (first, _, other) = models(client);
            sessions::run_checkpoint(first, other, CELL).await;
        },
    )
    .await;
    sessions::assert_recorded(CELL, SCENARIO);
}

/// The agent's conversation memory carries the reasoning into a second
/// prompt.
#[tokio::test]
async fn memory_unary() {
    const SCENARIO: &str = "session_matrix/memory_unary";
    with_openrouter_cassette("session_matrix/memory_unary", |client| async move {
        let (first, _, _) = models(client);
        sessions::run_memory(first, CELL, false).await;
    })
    .await;
    sessions::assert_memory_recorded(CELL, SCENARIO);
}

/// The agent's conversation memory carries the reasoning into a second
/// prompt (streamed).
#[tokio::test]
async fn memory_streamed() {
    const SCENARIO: &str = "session_matrix/memory_streamed";
    with_openrouter_cassette("session_matrix/memory_streamed", |client| async move {
        let (first, _, _) = models(client);
        sessions::run_memory(first, CELL, true).await;
    })
    .await;
    sessions::assert_memory_recorded(CELL, SCENARIO);
}
