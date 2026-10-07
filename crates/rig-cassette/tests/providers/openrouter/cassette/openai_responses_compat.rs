//! Cassette-backed OpenRouter compatibility coverage through Rig's OpenAI
//! Responses wire: the `OPENROUTER` dialect routed to `/responses` once.

use crate::support::{assert_nonempty_response, collect_stream_final_response_and_provider_final};

use super::super::support::with_openrouter_openai_cassette;
use rig::completion::{CompletionRequest, Cost};
use rig::providers::openrouter::extension::OpenRouterExt;
use serde_json::Value;

/// The cost the recorded Responses reply of `scenario` reports in its
/// `usage.cost`, credits of one USD: the unary body's, or the stream
/// terminal's.
fn recorded_cost(scenario: &str) -> Option<Cost> {
    let interactions = crate::cassettes::recorded_interaction_bodies("openrouter", scenario);
    let [(_, body)] = interactions.as_slice() else {
        panic!("{scenario} records one exchange");
    };
    let response = match serde_json::from_str::<Value>(body) {
        Ok(body) => body,
        Err(_) => crate::cassettes::recorded_sse_json_frames("openrouter", scenario)
            .into_iter()
            .find(|frame| frame["type"] == "response.completed")
            .map(|frame| frame["response"].clone())
            .expect("the recorded stream ends"),
    };
    let cost = response["usage"]["cost"]
        .as_f64()
        .expect("the recorded usage reports its cost");
    Some(Cost::from_total(cost))
}

const DEFAULT_OPENAI_COMPAT_MODEL: &str = "google/gemini-3-flash-preview";

#[tokio::test]
async fn openai_responses_raw_response_accepts_service_tier_metadata() {
    with_openrouter_openai_cassette(
        "openai_responses_compat/openai_responses_raw_response_accepts_service_tier_metadata",
        |client| async move {
            let model = client.completion(DEFAULT_OPENAI_COMPAT_MODEL);
            let request = CompletionRequest::new("Reply with exactly: openrouter responses service tier ok")
                .preamble(
                    "Return the requested text exactly, with no extra commentary.".to_string(),
                );

            // `service_tier` is Responses-API metadata rig does not normalize,
            // so it is read from OpenRouter's typed reply extras.
            let response = model
                .call(request)
                .await
                .expect("OpenRouter Responses API completion should deserialize");

            let service_tier = response
                .extras::<OpenRouterExt>()
                .expect("an OpenRouter reply")
                .expect("OpenRouter extras decode")
                .service_tier
                .expect("OpenRouter response should include service_tier");

            assert!(
                !service_tier.is_empty(),
                "expected OpenRouter model {DEFAULT_OPENAI_COMPAT_MODEL} to return service_tier metadata"
            );
            assert_eq!(
                response.usage.cost,
                recorded_cost(
                    "openai_responses_compat/openai_responses_raw_response_accepts_service_tier_metadata"
                ),
                "the cost OpenRouter reports"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn openai_responses_stream_against_openrouter_completes() {
    with_openrouter_openai_cassette(
        "openai_responses_compat/openai_responses_stream_against_openrouter_completes",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(DEFAULT_OPENAI_COMPAT_MODEL))
                .preamble("You are concise. Answer directly.")
                .build();

            let mut stream = agent
                .prompt("In one sentence, confirm this streaming response works.")
                .stream();
            let (response, call) = collect_stream_final_response_and_provider_final(&mut stream)
                .await
                .expect("streaming prompt should not fail on OpenRouter service_tier metadata");

            assert_nonempty_response(&response);
            assert_eq!(
                call.usage.cost,
                recorded_cost(
                    "openai_responses_compat/openai_responses_stream_against_openrouter_completes"
                ),
                "the cost OpenRouter reports"
            );
        },
    )
    .await;
}
