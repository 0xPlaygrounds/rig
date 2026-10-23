//! Doubleword's Responses route, recorded live. It accepts `include` but
//! returns no ciphertext, so a reasoning request asks for none; its
//! reasoning items carry their own `reasoning_text` and go back as returned
//! under `store: false`.

use rig::completion::{Effort, GenerationOptions, ServiceTier};
use rig::providers::doubleword;
use serde_json::Value;

use super::super::support::with_doubleword_cassette;
use crate::responses_route;

const FLEX: &str = "responses_route/flex_tier_waits_for_its_answer";
const ROUND_TRIP: &str = "responses_route/stateless_reasoning_tool_round_trip";

/// A unary call at the flex tier waits for the answer, which reports the
/// tier; `top_p` and the effort reach the body.
#[tokio::test]
async fn flex_tier_waits_for_its_answer() {
    let options = GenerationOptions::default()
        .reasoning(Effort::Low)
        .service_tier(ServiceTier::Flex)
        .top_p(0.5);
    with_doubleword_cassette(
        "responses_route/flex_tier_waits_for_its_answer",
        |client| async move {
            let reply =
                responses_route::plain_turn(&client.responses(doubleword::GPT_OSS_20B), options)
                    .await;
            assert_eq!(reply.raw["service_tier"], "flex");
            assert_eq!(reply.raw["status"], "completed");
        },
    )
    .await;
    let recorded = crate::cassettes::recorded_interaction_bodies("doubleword", FLEX);
    let (request, _) = recorded.first().expect("one turn recorded");
    let request: Value = serde_json::from_str(request).expect("request is JSON");
    assert_eq!(request["service_tier"], "flex");
    assert_eq!(request["top_p"], 0.5);
    assert_eq!(request["reasoning"]["effort"], "low");
    assert!(request.get("background").is_none(), "{request}");
}

/// The realtime `priority` tier on a stateless tool round trip.
#[tokio::test]
async fn stateless_reasoning_tool_round_trip() {
    with_doubleword_cassette(
        "responses_route/stateless_reasoning_tool_round_trip",
        |client| async move {
            responses_route::stateless_tool_round_trip(
                &client.responses(doubleword::GPT_OSS_20B),
                GenerationOptions::default()
                    .reasoning(Effort::Low)
                    .service_tier(ServiceTier::Priority),
            )
            .await;
        },
    )
    .await;
    responses_route::assert_reasoning_replayed("doubleword", ROUND_TRIP);
    let recorded = crate::cassettes::recorded_interaction_bodies("doubleword", ROUND_TRIP);
    for (request, _) in &recorded {
        let request: Value = serde_json::from_str(request).expect("request is JSON");
        assert_eq!(request["service_tier"], "priority");
    }
}
