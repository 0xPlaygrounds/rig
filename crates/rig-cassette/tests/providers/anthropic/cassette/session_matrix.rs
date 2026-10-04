//! Reasoning across a session boundary on Anthropic: see
//! `rig_test_support::history_survival::sessions`.

use super::super::support::with_anthropic_cassette;
use rig_test_support::cassette_models::AnthropicModels;

use crate::history_survival::sessions::{self, Cell};

fn params() -> Option<serde_json::Value> {
    Some(serde_json::json!({ "thinking": { "type": "enabled", "budget_tokens": 1024 } }))
}

const CELL: Cell = Cell {
    provider: "anthropic",
    params,
    max_tokens: 4096,
    expect: &["signature"],
};

fn models(
    client: AnthropicModels,
) -> (
    rig::Model<rig::providers::anthropic::wire::Messages>,
    rig::Model<rig::providers::anthropic::wire::Messages>,
    rig::Model<rig::providers::anthropic::wire::Messages>,
) {
    (
        client.completion("claude-haiku-4-5"),
        client.completion("claude-haiku-4-5"),
        client.completion("claude-sonnet-4-6"),
    )
}

/// The agent's conversation memory carries the reasoning into a second
/// prompt.
#[tokio::test]
async fn memory_unary() {
    const SCENARIO: &str = "session_matrix/memory_unary";
    with_anthropic_cassette("session_matrix/memory_unary", |client| async move {
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
    with_anthropic_cassette("session_matrix/memory_streamed", |client| async move {
        let (first, _, _) = models(client);
        sessions::run_memory(first, CELL, true).await;
    })
    .await;
    sessions::assert_memory_recorded(CELL, SCENARIO);
}
