//! GLM's thinking output on the OpenAI dialect, round-tripped into a second
//! turn.
//!
//! GLM emits reasoning by default, and the OpenAI dialect carries it as
//! `reasoning_content` on the assistant message. Two things can go wrong and
//! neither is visible from a single turn: rig can drop or mislabel the field on
//! the way in, and rig can fail to echo it back on the way out — which is what
//! a provider requiring reasoning continuity rejects.
//!
//! Both halves are the shared round trip's job, so this delegates to it rather
//! than rebuilding history by hand: `run_reasoning_roundtrip_nonstreaming`
//! composes turn 2 from the *parsed* `response.choice`, so any
//! `AssistantContent::Reasoning` rig produced is what gets re-serialized. A
//! hand-rolled version that fed back `agent.prompt(...)`'s `String` would
//! discard the reasoning by construction and exercise neither half.

use rig::prelude::*;

use super::super::THINKING_MODEL;
use super::super::support::{recorded_response_body, with_zai_general_cassette};
use crate::reasoning::{self, ReasoningRoundtripAgent};

/// GLM's OpenAI dialect takes Z.AI's own `thinking` object rather than
/// OpenAI's `reasoning_effort`.
fn thinking_params() -> serde_json::Value {
    serde_json::json!({
        "thinking": { "type": "enabled" }
    })
}

#[tokio::test]
#[ignore = "unrecorded: requires ZAI_API_KEY to record the cassette"]
async fn general_thinking_roundtrip_blocking() {
    with_zai_general_cassette("general/thinking_roundtrip_blocking", |client| async move {
        reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
            client.completion_model(THINKING_MODEL),
            Some(thinking_params()),
        ))
        .await;
    })
    .await;

    // Premise: a cell about reasoning that stopped receiving reasoning would
    // keep passing while covering nothing. Read turn 1's recorded response, not
    // rig's parse — the parse is what the round trip above already asserts.
    let body = recorded_response_body("general/thinking_roundtrip_blocking");
    let reasoning = body["choices"][0]["message"]["reasoning_content"]
        .as_str()
        .expect("the recorded turn must actually carry reasoning_content");
    assert!(
        !reasoning.trim().is_empty(),
        "an empty reasoning_content covers nothing"
    );
}
