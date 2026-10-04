//! OpenAI Responses API tool-argument shape regression tests.
//!
//! Locks down how tool-call arguments survive the Responses API wire format:
//! empty `{}` arguments, deeply nested objects, and non-ASCII/escaped strings,
//! in both streaming (fragmented `function_call_arguments.delta` reassembly)
//! and non-streaming form.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::completion::Message;
use rig::message::AssistantContent;
use rig::providers::openai;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;

use super::super::support::with_openai_cassette;

const NESTED_ARGS_PREAMBLE: &str = "\
You are a travel booking assistant. Use the plan_trip tool for every booking request \
and copy the requested values into the tool arguments exactly as given.";

const NESTED_ARGS_PROMPT: &str = "\
Book this trip by calling the plan_trip tool exactly once with these exact values: \
itinerary.city = \"Kyoto\", itinerary.days = 3, \
itinerary.activities = [\"temples\", \"tea ceremony\"], \
itinerary.lodging.name = \"Sakura Inn\", itinerary.lodging.rooms = 2. \
After the tool returns, repeat its confirmation code in one short sentence.";

#[derive(Debug, thiserror::Error)]
#[error("Trip planning failed")]
struct PlanTripError;

#[derive(Deserialize)]
struct Lodging {
    name: String,
    rooms: u32,
}

#[derive(Deserialize)]
struct Itinerary {
    city: String,
    days: u32,
    activities: Vec<String>,
    lodging: Lodging,
}

#[derive(Deserialize)]
struct PlanTripArgs {
    itinerary: Itinerary,
}

struct PlanTrip;

impl Tool for PlanTrip {
    const NAME: &'static str = "plan_trip";
    type Error = PlanTripError;
    type Args = PlanTripArgs;
    type Output = String;

    fn description(&self) -> String {
        "Book a trip from a nested itinerary object.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        plan_trip_parameters()
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(format!(
            "Booked {} for {} day(s), {} room(s) at {}, with {} planned activities. \
             Confirmation code SAKURA-77.",
            args.itinerary.city,
            args.itinerary.days,
            args.itinerary.lodging.rooms,
            args.itinerary.lodging.name,
            args.itinerary.activities.len(),
        ))
    }
}

fn plan_trip_parameters() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "itinerary": {
                "type": "object",
                "description": "The trip to book.",
                "properties": {
                    "city": { "type": "string" },
                    "days": { "type": "integer" },
                    "activities": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "lodging": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" },
                            "rooms": { "type": "integer" }
                        },
                        "required": ["name", "rooms"]
                    }
                },
                "required": ["city", "days", "activities", "lodging"]
            }
        },
        "required": ["itinerary"]
    })
}

fn assert_expected_plan_trip_arguments(arguments: &serde_json::Value) {
    let itinerary = arguments
        .get("itinerary")
        .expect("arguments should contain the nested itinerary object");
    assert_eq!(
        itinerary.get("city").and_then(|value| value.as_str()),
        Some("Kyoto"),
        "nested city should survive the wire format: {arguments:?}"
    );
    assert_eq!(
        itinerary.get("days").and_then(serde_json::Value::as_u64),
        Some(3),
        "nested integer should survive the wire format: {arguments:?}"
    );
    assert_eq!(
        itinerary.get("activities"),
        Some(&json!(["temples", "tea ceremony"])),
        "nested string array should survive the wire format: {arguments:?}"
    );
    let lodging = itinerary
        .get("lodging")
        .expect("arguments should contain the doubly nested lodging object");
    assert_eq!(
        lodging.get("name").and_then(|value| value.as_str()),
        Some("Sakura Inn"),
        "doubly nested string should survive the wire format: {arguments:?}"
    );
    assert_eq!(
        lodging.get("rooms").and_then(serde_json::Value::as_u64),
        Some(2),
        "doubly nested integer should survive the wire format: {arguments:?}"
    );
}

#[tokio::test]
async fn nested_arguments_roundtrip_nonstreaming() {
    with_openai_cassette(
        "responses_tool_args/nested_arguments_roundtrip_nonstreaming",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.openai.completion(openai::GPT_4O))
                .preamble(NESTED_ARGS_PREAMBLE)
                .tool(PlanTrip)
                .default_max_turns(4)
                .build();
            let mut history = Vec::<Message>::new();

            let result = agent
                .chat(NESTED_ARGS_PROMPT, &mut history)
                .await
                .expect("nested-args tool chat should succeed")
                .output();

            assert!(
                result.contains("SAKURA-77"),
                "final answer should repeat the tool's confirmation code, got {result:?}"
            );

            let arguments = history
                .iter()
                .find_map(|message| match message {
                    Message::Assistant(rig_core::message::AssistantMessage { content, .. }) => {
                        content.iter().find_map(|item| match item {
                            AssistantContent::ToolCall(tool_call)
                                if tool_call.function.name == PlanTrip::NAME =>
                            {
                                Some(tool_call.function.arguments_value())
                            }
                            _ => None,
                        })
                    }
                    _ => None,
                })
                .expect("chat history should record the plan_trip tool call");
            assert_expected_plan_trip_arguments(&arguments);
        },
    )
    .await;
}
