//! Edge matrix for items a stateless Responses client must send back
//! unchanged (rig#2269).
//!
//! **Bug.** Two things a client that manages its own conversation state
//! could not do: (a) send a `compaction` item back — `InputItem` rejected the
//! tag; (b) re-send an output message's `phase` — `OutputMessage` dropped it,
//! and OpenAI documents that as a quality regression on follow-ups.
//!
//! **Fix.** `InputContent::Compaction` / `Output::Compaction` round-trip
//! the item verbatim; `OutputMessage.phase` is captured, rides rig history
//! on the text block's own-wire extras, and is lifted back onto the assistant
//! input item at replay. Exposing the terminal `output[]` on the streamed
//! record was tried and reverted: `StreamFinal::raw` is replay identity for
//! every streamed effect log, and the field changed 66 goldens.
//!
//! **Fixtures.** Cell 1 is recorded live against `gpt-5.6-sol`, which
//! returns `phase: "final_answer"` on every message. Its second recorded
//! request is the proof: it carries the `phase` the first response
//! returned. Cell 2 is hand-derived from cell 1: a `compaction` item is
//! inserted at the head of turn 2's *request* `input[]` and at the head of
//! turn 1's *response* `output[]` (the shape `/responses/compact` returns),
//! with every other byte identical. Rig has no `/responses/compact` client
//! method — the reporter's fourth ask, deferred as maintainer-owned API
//! surface — so the item cannot be obtained live through rig; the derived
//! cell asserts the item is present in both places so a re-record cannot
//! silently drop it.
//!
//! **How these cells fail on `origin/main`.** Cell 1 misses the mock on
//! turn 2 (rig sent no `phase`, so the recorded body differs) and its
//! post-replay assertion fails; cell 2 decodes the item only as `Output::Unknown` and the input side
//! rejects it with `unknown variant \`compaction\``.
//!
//! | # | cell | transport | proves | fixture |
//! |---|------|-----------|--------|---------|
//! | 1 | `phase_round_trips_on_follow_up` | blocking, 2 turns | turn-2 request carries turn-1's `phase` | recorded |
//! | 2 | `compaction_item_decodes_on_the_response` | blocking | compaction on `output[]` decodes typed; the same item re-serialized is accepted on the input side verbatim | derived from 1 |
//!
//! Unit cells for the (de)serializers live in
//! `crates/rig-core/src/providers/openai/responses_api/stateless_replay_tests.rs`.

use rig::completion::CompletionModel;
use rig::message::Message;
use rig::prelude::*;
use rig::providers::openai;
use rig::providers::openai::OpenAI;
use rig::providers::openai::responses_api::{
    CompletionResponse as ProviderResponse, InputItem, Output,
};
use serde::Deserialize;
use serde_json::Value;

use super::super::support::with_openai_cassette;

const TURN_ONE: &str = "Remember the codeword ALPHA-17. Reply exactly: ACK-1";
const TURN_TWO: &str = "Reply with exactly the remembered codeword.";
const PHASE: &str = "final_answer";

/// Every recorded request body of `scenario`, decoded, in wire order.
fn recorded_requests(scenario: &str) -> Vec<Value> {
    crate::cassettes::recorded_interaction_bodies("openai", scenario)
        .into_iter()
        .map(|(request, _)| serde_json::from_str(&request).expect("request body is JSON"))
        .collect()
}

/// Every recorded non-streaming response body of `scenario`, decoded.
fn recorded_responses(scenario: &str) -> Vec<Value> {
    crate::cassettes::recorded_interaction_bodies("openai", scenario)
        .into_iter()
        .filter_map(|(_, response)| serde_json::from_str(&response).ok())
        .collect()
}

fn assistant_items(request: &Value) -> Vec<&Value> {
    request["input"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter(|item| {
                    item.get("type").and_then(Value::as_str) == Some("message")
                        && item.get("role").and_then(Value::as_str) == Some("assistant")
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The Responses API's own reply, read back out of
/// [`rig::completion::CompletionResponse::raw`] — the same value the
/// normalized response beside it was derived from.
fn provider_reply(response: &rig::completion::CompletionResponse) -> ProviderResponse {
    ProviderResponse::deserialize(&response.raw)
        .expect("`raw` is the serialized responses_api::CompletionResponse")
}

/// Two blocking turns, threading turn 1's normalized response back as history.
async fn two_turn_conversation(client: Bound<OpenAI>) -> (ProviderResponse, ProviderResponse) {
    let model = client.completion(openai::GPT_5_6_SOL);
    let first = model
        .completion(model.completion_request(TURN_ONE).build())
        .await
        .expect("turn 1 should succeed");
    let assistant = Message::Assistant {
        id: first.message_id.clone(),
        content: first.choice.clone(),
    };
    let second = model
        .completion(
            model
                .completion_request(TURN_TWO)
                .messages([Message::user(TURN_ONE), assistant])
                .build(),
        )
        .await
        .expect("turn 2 should succeed");
    (provider_reply(&first), provider_reply(&second))
}

#[tokio::test]
async fn phase_round_trips_on_follow_up() {
    with_openai_cassette(
        "stateless_replay_matrix/phase_round_trips_on_follow_up",
        |client| async move {
            let (first, _second) = two_turn_conversation(client.openai).await;
            let phases: Vec<&str> = first
                .output
                .iter()
                .filter_map(|item| match item {
                    Output::Message(message) => message.phase.as_deref(),
                    _ => None,
                })
                .collect();
            assert_eq!(phases, [PHASE], "turn 1 must decode the wire's phase");
        },
    )
    .await;

    let requests = recorded_requests("stateless_replay_matrix/phase_round_trips_on_follow_up");
    assert_eq!(requests.len(), 2, "two turns recorded");
    let replayed = assistant_items(&requests[1]);
    assert_eq!(
        replayed.len(),
        1,
        "turn 2 replays exactly one assistant item"
    );
    assert_eq!(
        replayed[0]["phase"], PHASE,
        "turn 2's request must re-send the phase turn 1 returned: {}",
        replayed[0]
    );
    // Never on the block: the flatten would put it beside `text`.
    for block in replayed[0]["content"].as_array().expect("content array") {
        assert!(
            block.get("phase").is_none(),
            "phase leaked onto a block: {block}"
        );
    }
}

#[tokio::test]
async fn compaction_item_decodes_on_the_response() {
    if crate::cassettes::skip_when_recording(
        "cell 2 is hand-derived from cell 1: the compaction item on the response output is not what the API returned",
    ) {
        return;
    }
    with_openai_cassette(
        "stateless_replay_matrix/compaction_item_decodes_on_the_response",
        |client| async move {
            let model = client.openai.completion(openai::GPT_5_6_SOL);
            let response = model
                .completion(model.completion_request(TURN_ONE).build())
                .await
                .expect("a response carrying a compaction item must decode");
            let first = provider_reply(&response);
            let compaction = first
                .output
                .iter()
                .find_map(|item| match item {
                    Output::Compaction(fields) => Some(fields.clone()),
                    _ => None,
                })
                .expect("turn 1's output must decode the compaction item as Output::Compaction");
            assert_eq!(compaction.get("id"), Some(&Value::from("cmp_REDACTED_1")));
            assert!(compaction.get("type").is_none());

            // The regular items beside it still decode and normalize.
            assert!(!response.choice.is_empty());

            // The same item, re-serialized, is accepted on the input side
            // byte-for-byte — this is what a stateless client sends back.
            let wire = serde_json::to_value(Output::Compaction(compaction)).expect("serializes");
            let input: InputItem =
                serde_json::from_value(wire.clone()).expect("the input side accepts the item");
            assert_eq!(
                serde_json::to_value(&input).expect("re-serializes"),
                wire,
                "the input item must re-emit the compaction item verbatim"
            );
        },
    )
    .await;

    let responses =
        recorded_responses("stateless_replay_matrix/compaction_item_decodes_on_the_response");
    assert!(
        responses[0]["output"][0]["type"] == "compaction",
        "derived fixture must carry the compaction item on the response output: {}",
        responses[0]["output"]
    );
}
