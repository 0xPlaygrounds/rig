//! Gemini function-call argument shape regression tests.
//!
//! Locks down how functionCall args survive the generateContent wire format:
//! deeply nested objects with arrays, non-ASCII/escaped strings, and optional
//! (nullable) fields, in both streaming and non-streaming form. Zero-argument
//! calls are already covered by `agent_tools` and `streaming_tools`.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::client::CompletionClient;
use rig::completion::{Chat, CompletionModel, Message, ToolDefinition};
use rig::message::AssistantContent;
use rig::providers::gemini;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;

use super::super::support::with_gemini_cassette;
use crate::support::collect_raw_stream_observation;

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

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Book a trip from a nested itinerary object.".to_string(),
            parameters: plan_trip_parameters(),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
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
        itinerary.get("days").and_then(|value| value.as_u64()),
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
        lodging.get("rooms").and_then(|value| value.as_u64()),
        Some(2),
        "doubly nested integer should survive the wire format: {arguments:?}"
    );
}

#[tokio::test]
async fn nested_arguments_roundtrip_nonstreaming() {
    with_gemini_cassette(
        "generate_tool_args/nested_arguments_roundtrip_nonstreaming",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(NESTED_ARGS_PREAMBLE)
                .temperature(0.0)
                .tool(PlanTrip)
                .default_max_turns(4)
                .build();
            let mut history = Vec::<Message>::new();

            let result = agent
                .chat(NESTED_ARGS_PROMPT, &mut history)
                .await
                .expect("nested-args tool chat should succeed");

            assert!(
                result.contains("SAKURA-77"),
                "final answer should repeat the tool's confirmation code, got {result:?}"
            );

            let arguments = history
                .iter()
                .find_map(|message| match message {
                    Message::Assistant { content, .. } => {
                        content.iter().find_map(|item| match item {
                            AssistantContent::ToolCall(tool_call)
                                if tool_call.function.name == PlanTrip::NAME =>
                            {
                                Some(tool_call.function.arguments.clone())
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

#[tokio::test]
async fn nested_arguments_streaming() {
    with_gemini_cassette(
        "generate_tool_args/nested_arguments_streaming",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request(NESTED_ARGS_PROMPT)
                .preamble(NESTED_ARGS_PREAMBLE.to_string())
                .temperature(0.0)
                .tool(PlanTrip.definition(String::new()).await)
                .build();

            let observation = collect_raw_stream_observation(
                model
                    .stream(request)
                    .await
                    .expect("nested-args streaming request should start"),
            )
            .await;

            assert!(
                observation.errors.is_empty(),
                "stream should not emit errors: {:?}",
                observation.errors
            );
            let tool_call = observation
                .tool_calls
                .iter()
                .find(|tool_call| tool_call.function.name == PlanTrip::NAME)
                .expect("stream should emit the plan_trip tool call");
            assert_expected_plan_trip_arguments(&tool_call.function.arguments);
        },
    )
    .await;
}

#[tokio::test]
async fn unicode_arguments_streaming() {
    with_gemini_cassette(
        "generate_tool_args/unicode_arguments_streaming",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request(
                    "Call the echo tool exactly once with the message argument set to \
                     exactly this text: Grüße aus 東京, from the \"naïve café\"!",
                )
                .preamble(
                    "You must call the echo tool with the exact text the user provides. \
                     Do not translate, reword, or drop any characters."
                        .to_string(),
                )
                .temperature(0.0)
                .tool(ToolDefinition {
                    name: "echo".to_string(),
                    description: "Echo a message back to the user.".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "message": { "type": "string" }
                        },
                        "required": ["message"]
                    }),
                })
                .build();

            let observation = collect_raw_stream_observation(
                model
                    .stream(request)
                    .await
                    .expect("unicode-args streaming request should start"),
            )
            .await;

            assert!(
                observation.errors.is_empty(),
                "stream should not emit errors: {:?}",
                observation.errors
            );
            let tool_call = observation
                .tool_calls
                .iter()
                .find(|tool_call| tool_call.function.name == "echo")
                .expect("stream should emit the echo tool call");
            let message = tool_call
                .function
                .arguments
                .get("message")
                .and_then(|value| value.as_str())
                .expect("echo arguments should contain a message string");
            for expected in ["Grüße", "東京", "naïve café"] {
                assert!(
                    message.contains(expected),
                    "streamed unicode arguments should preserve {expected:?}, got {message:?}"
                );
            }
        },
    )
    .await;
}

#[tokio::test]
async fn optional_nullable_argument_omitted_when_not_requested() {
    with_gemini_cassette(
        "generate_tool_args/optional_nullable_argument_omitted_when_not_requested",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request(
                    "Log an event named \"deploy\" using the log_event tool. \
                     Do not attach a note.",
                )
                .preamble(
                    "Use the log_event tool for every logging request. Only fill optional \
                     arguments when the user explicitly provides them."
                        .to_string(),
                )
                .temperature(0.0)
                .tool(ToolDefinition {
                    name: "log_event".to_string(),
                    description: "Record an event with an optional free-form note.".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" },
                            "note": {
                                "type": "string",
                                "nullable": true,
                                "description": "Optional note; omit when the user gives none."
                            }
                        },
                        "required": ["name"]
                    }),
                })
                .build();

            let response = model
                .completion(request)
                .await
                .expect("optional-arg completion should succeed");

            let tool_call = response
                .choice
                .iter()
                .find_map(|content| match content {
                    AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
                    _ => None,
                })
                .expect("response should contain the log_event tool call");
            assert_eq!(tool_call.function.name, "log_event");
            assert_eq!(
                tool_call
                    .function
                    .arguments
                    .get("name")
                    .and_then(|value| value.as_str()),
                Some("deploy"),
                "required argument should be present: {:?}",
                tool_call.function.arguments
            );
            let note = tool_call.function.arguments.get("note");
            assert!(
                note.is_none() || note.is_some_and(serde_json::Value::is_null),
                "optional nullable argument should be omitted or null when not requested, \
                 got {:?}",
                tool_call.function.arguments
            );
        },
    )
    .await;
}
