//! Reasoning across a session boundary on Gemini: see
//! `rig_test_support::history_survival::sessions`.

use super::super::support::with_gemini_cassette;
use rig_test_support::cassette_models::GeminiModels;

use crate::history_survival::sessions::{self, Cell};

fn params() -> Option<serde_json::Value> {
    Some(
        serde_json::json!({ "generationConfig": { "thinkingConfig": { "thinkingBudget": 1024, "includeThoughts": true } } }),
    )
}

const CELL: Cell = Cell {
    provider: "gemini",
    params,
    max_tokens: 4096,
    expect: &["thought_signature"],
};

fn models(
    client: GeminiModels,
) -> (
    rig::Model<
        rig::providers::gemini::completion::GenerateContent,
        rig::http_client::DynHttpClient,
    >,
    rig::Model<
        rig::providers::gemini::completion::GenerateContent,
        rig::http_client::DynHttpClient,
    >,
    rig::Model<
        rig::providers::gemini::completion::GenerateContent,
        rig::http_client::DynHttpClient,
    >,
) {
    (
        client.completion("gemini-2.5-flash"),
        client.completion("gemini-2.5-flash"),
        client.completion("gemini-3-flash-preview"),
    )
}

/// The agent's conversation memory carries the reasoning into a second
/// prompt.
#[tokio::test]
async fn memory_unary() {
    const SCENARIO: &str = "session_matrix/memory_unary";
    with_gemini_cassette("session_matrix/memory_unary", |client| async move {
        let (first, _, _) = models(client);
        sessions::run_memory(first, CELL, false).await;
    })
    .await;
    sessions::assert_memory_recorded(CELL, SCENARIO);
}
