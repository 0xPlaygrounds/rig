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

/// A turn continued on another model replays from its canonical fields:
/// the first reply delivered a signed thinking block, and the continuation
/// carries its text but no thinking block and no signature, which only the
/// model that produced them reads.
fn assert_ported(scenario: &str) {
    let bodies = crate::cassettes::recorded_interaction_bodies("anthropic", scenario);
    assert_eq!(bodies.len(), 2, "one turn and one continuation");
    let reply: serde_json::Value =
        serde_json::from_str(&bodies[0].1).expect("the first reply is JSON");
    let thinking = reply["content"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|block| block["type"] == "thinking" && block["signature"] != "")
        .expect("premise: the first reply delivered a signed thinking block");
    let next: serde_json::Value =
        serde_json::from_str(&bodies[1].0).expect("the continuation request is JSON");
    let blocks: Vec<&serde_json::Value> = next["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|message| message["role"] == "assistant")
        .flat_map(|message| message["content"].as_array().into_iter().flatten())
        .collect();
    assert!(
        blocks
            .iter()
            .all(|block| block["type"] != "thinking" && block["type"] != "redacted_thinking"),
        "no thinking block reaches another model: {blocks:?}"
    );
    assert!(
        !bodies[1].0.contains(
            thinking["signature"]
                .as_str()
                .expect("the signature is a string")
        ),
        "the signature does not reach another model"
    );
    assert!(
        blocks
            .iter()
            .any(|block| block["type"] == "text" && block["text"] == thinking["thinking"]),
        "the reasoning reaches another model as text: {blocks:?}"
    );
}

#[tokio::test]
async fn same_model() {
    const SCENARIO: &str = "session_matrix/same_model";
    with_anthropic_cassette("session_matrix/same_model", |client| async move {
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
    with_anthropic_cassette("session_matrix/other_model", |client| async move {
        let (first, _, other) = models(client);
        sessions::run(first, other, CELL).await;
    })
    .await;
    assert_ported(SCENARIO);
}

/// The continuation is checkpointed, restored into a fresh world, and sent
/// by the restored world's handler for another model of the same provider.
#[tokio::test]
async fn checkpoint_other_model() {
    const SCENARIO: &str = "session_matrix/checkpoint_other_model";
    with_anthropic_cassette(
        "session_matrix/checkpoint_other_model",
        |client| async move {
            let (first, _, other) = models(client);
            sessions::run_checkpoint(first, other, CELL).await;
        },
    )
    .await;
    assert_ported(SCENARIO);
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
