//! Edge matrix for the per-model defaults rig derives from an Anthropic model
//! id: the `max_tokens` it sends when the caller sets none (rig#2505,
//! rig#2506) and whether a mid-conversation `role: "system"` message stays in
//! `messages` or is hoisted into the top-level `system` field.
//!
//! **Bug.** `default_max_tokens_for_model` capped `claude-sonnet-4-6` at
//! 64k although its published synchronous output limit is 128k, so an
//! uncapped long-output request was silently truncated at half the model's
//! budget. The Claude 5 family (`claude-fable-5-1`, `claude-fable-5`,
//! `claude-opus-5`, `claude-sonnet-5`) had no constants and no entry in the
//! table at all, so it fell through to the 2,048-token fallback.
//!
//! **How these cells fail on `origin/main`.** The cassette harness matches
//! the recorded *request body*, and `max_tokens` is part of it: the Sonnet
//! 4.6 cell recorded `128000` and is a mock miss against a build that sends
//! `64000`; each Claude 5 cell recorded its documented limit and is a mock
//! miss against the `2048` fallback. Each cell also reads its fixture back and
//! asserts the literal value so the claim is visible. A default the API
//! rejected (a `max_tokens` above the model's limit is a 400) could not have
//! been recorded, which is what makes the fixture evidence rather than a
//! transcription of the docs.
//!
//! | # | cell | model | default `max_tokens` | system placement |
//! |---|------|-------|----------------------|------------------|
//! | 1 | `sonnet_4_6_defaults_to_128k` | claude-sonnet-4-6 | 128000 | — |
//! | 2 | `opus_5_defaults_to_128k` | claude-opus-5 | 128000 | — |
//! | 3 | `sonnet_5_defaults_to_128k` | claude-sonnet-5 | 128000 | — |
//! | 4 | `fable_5_1_defaults_to_128k` | claude-fable-5-1 | 128000 | — |
//! | 5 | `haiku_4_5_defaults_to_64k` | claude-haiku-4-5 | 64000 (control) | — |
//! | 6 | `opus_5_preserves_mid_conversation_system_role` | claude-opus-5 | 128000 | kept in `messages` |
//! | 7 | `sonnet_5_hoists_mid_conversation_system_role` | claude-sonnet-5 | 128000 | hoisted to `system` |
//!
//! The table itself is definitory and unit-tested beside the implementation
//! (`current_model_default_max_tokens_match_anthropic_limits` in
//! `crates/rig-core/src/providers/anthropic/completion/tests.rs`); these
//! cells prove the provider accepts what the table says.

use rig::completion::{CompletionModel, Message};
use rig::driver::Bound;
use rig::providers::anthropic::completion::{
    CLAUDE_FABLE_5_1, CLAUDE_HAIKU_4_5, CLAUDE_OPUS_5, CLAUDE_SONNET_4_6, CLAUDE_SONNET_5,
};
use rig::providers::anthropic::wire::Anthropic;
use serde_json::Value;

use super::super::support::with_anthropic_cassette;
use crate::support::{assert_contains_any_case_insensitive, assistant_text_response};

const PROMPT: &str = "Reply with the single word OK.";
const SYSTEM_ROLE_INSTRUCTION: &str = "For the rest of this conversation, answer in Spanish only.";
const SKY_PROMPT: &str =
    "What color is a clear daytime sky? Reply with one lowercase Spanish word.";

fn recorded_request_bodies(scenario: &str) -> Vec<Value> {
    crate::cassettes::recorded_interaction_bodies("anthropic", scenario)
        .into_iter()
        .map(|(request, _)| serde_json::from_str(&request).expect("request body is JSON"))
        .collect()
}

fn assert_recorded_max_tokens(scenario: &str, expected: u64) {
    let bodies = recorded_request_bodies(scenario);
    assert!(
        !bodies.is_empty(),
        "{scenario}: fixture should record at least one request"
    );
    for body in &bodies {
        assert_eq!(
            body.get("max_tokens").and_then(Value::as_u64),
            Some(expected),
            "{scenario}: an uncapped request must carry the model's documented output limit"
        );
    }
}

fn message_has_role(message: &Value, role: &str) -> bool {
    message.get("role").and_then(Value::as_str) == Some(role)
}

fn blocks_contain_text(blocks: Option<&Value>, expected: &str) -> bool {
    blocks.and_then(Value::as_array).is_some_and(|blocks| {
        blocks
            .iter()
            .any(|block| block.get("text").and_then(Value::as_str) == Some(expected))
    })
}

fn assert_recorded_system_role_in_messages(scenario: &str) {
    let kept = recorded_request_bodies(scenario).iter().any(|body| {
        body.get("messages")
            .and_then(Value::as_array)
            .is_some_and(|messages| {
                messages.iter().any(|message| {
                    message_has_role(message, "system")
                        && blocks_contain_text(message.get("content"), SYSTEM_ROLE_INSTRUCTION)
                })
            })
    });
    assert!(
        kept,
        "{scenario}: a model that accepts mid-conversation system messages should keep \
         role=system inside messages[]"
    );
}

fn assert_recorded_system_role_hoisted(scenario: &str) {
    let bodies = recorded_request_bodies(scenario);
    let in_messages = bodies.iter().any(|body| {
        body.get("messages")
            .and_then(Value::as_array)
            .is_some_and(|messages| {
                messages
                    .iter()
                    .any(|message| message_has_role(message, "system"))
            })
    });
    assert!(
        !in_messages,
        "{scenario}: a model that rejects mid-conversation system messages must not see \
         role=system inside messages[]"
    );
    let hoisted = bodies
        .iter()
        .any(|body| blocks_contain_text(body.get("system"), SYSTEM_ROLE_INSTRUCTION));
    assert!(
        hoisted,
        "{scenario}: the instruction should be hoisted into the top-level system field"
    );
}

async fn assert_uncapped_turn(client: Bound<Anthropic>, model_id: &str) {
    let model = client.completion(model_id);
    let request = model.completion_request(PROMPT).build();
    let response = model
        .completion(request)
        .await
        .expect("an uncapped request must be accepted with the derived max_tokens");
    let text = assistant_text_response(&response.choice).expect("assistant text");
    assert_contains_any_case_insensitive(&text, &["ok"]);
}

async fn assert_mid_conversation_system_turn(client: Bound<Anthropic>, model_id: &str) {
    let model = client.completion(model_id);
    let request = model
        .completion_request(SKY_PROMPT)
        .messages([
            Message::user("Start a short language compliance check."),
            Message::system(SYSTEM_ROLE_INSTRUCTION),
            Message::assistant("Entendido."),
        ])
        .build();
    let response = model
        .completion(request)
        .await
        .expect("a mid-conversation system message must be accepted on both placements");
    let text = assistant_text_response(&response.choice).expect("assistant text");
    assert_contains_any_case_insensitive(&text, &["azul"]);
}

#[tokio::test]
async fn sonnet_4_6_defaults_to_128k() {
    with_anthropic_cassette(
        "model_defaults_matrix/sonnet_4_6_defaults_to_128k",
        |client| async move { assert_uncapped_turn(client, CLAUDE_SONNET_4_6).await },
    )
    .await;
    assert_recorded_max_tokens("model_defaults_matrix/sonnet_4_6_defaults_to_128k", 128_000);
}

#[tokio::test]
async fn opus_5_defaults_to_128k() {
    with_anthropic_cassette(
        "model_defaults_matrix/opus_5_defaults_to_128k",
        |client| async move { assert_uncapped_turn(client, CLAUDE_OPUS_5).await },
    )
    .await;
    assert_recorded_max_tokens("model_defaults_matrix/opus_5_defaults_to_128k", 128_000);
}

#[tokio::test]
async fn sonnet_5_defaults_to_128k() {
    with_anthropic_cassette(
        "model_defaults_matrix/sonnet_5_defaults_to_128k",
        |client| async move { assert_uncapped_turn(client, CLAUDE_SONNET_5).await },
    )
    .await;
    assert_recorded_max_tokens("model_defaults_matrix/sonnet_5_defaults_to_128k", 128_000);
}

#[tokio::test]
async fn fable_5_1_defaults_to_128k() {
    with_anthropic_cassette(
        "model_defaults_matrix/fable_5_1_defaults_to_128k",
        |client| async move { assert_uncapped_turn(client, CLAUDE_FABLE_5_1).await },
    )
    .await;
    assert_recorded_max_tokens("model_defaults_matrix/fable_5_1_defaults_to_128k", 128_000);
}

#[tokio::test]
async fn haiku_4_5_defaults_to_64k() {
    with_anthropic_cassette(
        "model_defaults_matrix/haiku_4_5_defaults_to_64k",
        |client| async move { assert_uncapped_turn(client, CLAUDE_HAIKU_4_5).await },
    )
    .await;
    assert_recorded_max_tokens("model_defaults_matrix/haiku_4_5_defaults_to_64k", 64_000);
}

#[tokio::test]
async fn opus_5_preserves_mid_conversation_system_role() {
    with_anthropic_cassette(
        "model_defaults_matrix/opus_5_preserves_mid_conversation_system_role",
        |client| async move { assert_mid_conversation_system_turn(client, CLAUDE_OPUS_5).await },
    )
    .await;
    assert_recorded_max_tokens(
        "model_defaults_matrix/opus_5_preserves_mid_conversation_system_role",
        128_000,
    );
    assert_recorded_system_role_in_messages(
        "model_defaults_matrix/opus_5_preserves_mid_conversation_system_role",
    );
}

#[tokio::test]
async fn sonnet_5_hoists_mid_conversation_system_role() {
    with_anthropic_cassette(
        "model_defaults_matrix/sonnet_5_hoists_mid_conversation_system_role",
        |client| async move { assert_mid_conversation_system_turn(client, CLAUDE_SONNET_5).await },
    )
    .await;
    assert_recorded_max_tokens(
        "model_defaults_matrix/sonnet_5_hoists_mid_conversation_system_role",
        128_000,
    );
    assert_recorded_system_role_hoisted(
        "model_defaults_matrix/sonnet_5_hoists_mid_conversation_system_role",
    );
}
