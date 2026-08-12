//! OpenAI Responses `phase` preservation across a stateless multi-turn
//! exchange (issue #2269).
//!
//! gpt-5.6-era models stamp a `phase` (e.g. `final_answer`) on message output
//! items, and OpenAI documents that follow-up requests should replay it —
//! dropping it can degrade performance. `OutputMessage` used to drop the
//! field at deserialization, so replayed history never carried it (every
//! committed multi-turn gpt-5.6 cassette shows `phase` in the responses and
//! not in the follow-up requests). The cassette harness matches outbound
//! request bodies, so the turn-2 request recorded here — which carries
//! `"phase":"final_answer"` on the replayed assistant message — pins the
//! roundtrip at the wire boundary.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.
use rig::completion::{CompletionModel, NormalizeCompletionResponse};
use rig::message::Message;
use rig::prelude::*;
use rig::providers::openai;

use super::super::support::with_openai_cassette;

#[tokio::test]
async fn phase_survives_ingest_and_replays_on_the_next_turn() {
    with_openai_cassette(
        "phase_roundtrip/phase_survives_ingest_and_replays_on_the_next_turn",
        |client| async move {
            let model = client.completion_model(openai::GPT_5_6);
            let first_user = Message::user("Reply with exactly: harbor-crimson");
            let request = model
                .completion_request(first_user.clone())
                .additional_params(serde_json::json!({
                    "reasoning": { "effort": "low" }
                }))
                .build();

            let raw = model
                .raw_completion(request)
                .await
                .expect("turn 1 should succeed");

            // Assertion derived from recorded traffic: the wire message item
            // carries `phase` (observed value `final_answer`), and the raw
            // response must preserve it verbatim.
            let raw_json = serde_json::to_value(&raw).expect("raw response should serialize");
            let message_item = raw_json["output"]
                .as_array()
                .expect("output should be an array")
                .iter()
                .find(|item| item["type"] == "message")
                .expect("output should contain a message item");
            assert_eq!(
                message_item["phase"], "final_answer",
                "the message item should preserve the recorded phase: {message_item}"
            );

            let response = raw.normalize("openai").expect("turn 1 should normalize");

            // Turn 2 replays turn 1's assistant message; the recorded request
            // body must carry `phase` on it (request-boundary assertion via
            // the cassette body match).
            let assistant = Message::Assistant {
                id: response.message_id.clone(),
                content: response.choice.clone(),
            };
            let request = model
                .completion_request(Message::user(
                    "Repeat the exact phrase from your previous reply.",
                ))
                .messages(vec![first_user, assistant])
                .additional_params(serde_json::json!({
                    "reasoning": { "effort": "low" }
                }))
                .build();

            let followup = model
                .completion(request)
                .await
                .expect("turn 2 should succeed");
            assert!(
                !followup.choice.is_empty(),
                "turn 2 should produce a response"
            );
        },
    )
    .await;
}
