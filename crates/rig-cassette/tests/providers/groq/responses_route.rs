//! Groq's Responses route (beta), recorded live. Groq rejects `include`, so a
//! reasoning request asks for no ciphertext; its reasoning items carry their
//! own `reasoning_text` and go back as returned under `store: false`, which
//! Groq requires (it rejects `store: true`).

use rig::completion::{Effort, GenerationOptions, ServiceTier};
use rig::providers::groq;
use serde_json::Value;

use super::support::with_groq_cassette_result;
use crate::responses_route;

const PLAIN: &str = "responses_route/plain_turn_on_demand_tier";
const ROUND_TRIP: &str = "responses_route/stateless_reasoning_tool_round_trip";

/// The typed tier, `top_p` and effort reach Groq's Responses body, and the
/// reply echoes the tier.
#[tokio::test]
async fn plain_turn_on_demand_tier() {
    let options = GenerationOptions::default()
        .reasoning(Effort::Low)
        .service_tier(ServiceTier::Default)
        .top_p(0.9);
    with_groq_cassette_result(
        "responses_route/plain_turn_on_demand_tier",
        |client| async move {
            let reply =
                responses_route::plain_turn(&client.responses(groq::GPT_OSS_20B), options).await;
            assert_eq!(reply.raw["service_tier"], "on_demand");
            Ok::<(), ()>(())
        },
    )
    .await
    .expect("the cell runs");
    let recorded = crate::cassettes::recorded_interaction_bodies("groq", PLAIN);
    let (request, _) = recorded.first().expect("one turn recorded");
    let request: Value = serde_json::from_str(request).expect("request is JSON");
    assert_eq!(request["service_tier"], "on_demand");
    assert_eq!(request["top_p"], 0.9);
    assert_eq!(request["reasoning"]["effort"], "low");
    assert!(request.get("include").is_none(), "{request}");
}

#[tokio::test]
async fn stateless_reasoning_tool_round_trip() {
    with_groq_cassette_result(
        "responses_route/stateless_reasoning_tool_round_trip",
        |client| async move {
            responses_route::stateless_tool_round_trip(
                &client.responses(groq::GPT_OSS_20B),
                GenerationOptions::default().reasoning(Effort::Low),
            )
            .await;
            Ok::<(), ()>(())
        },
    )
    .await
    .expect("the cell runs");
    responses_route::assert_reasoning_replayed("groq", ROUND_TRIP);
}
