//! Anthropic Messages API tool-argument shape regression tests.
//!
//! Locks down how tool_use inputs survive the Messages API wire format: empty
//! `{}` arguments, deeply nested objects with arrays, and non-ASCII/escaped
//! strings, in both streaming (fragmented `input_json_delta` reassembly) and
//! non-streaming form.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::agent::tool::Tool;
use rig::client::{AgentClientExt, CompletionClient};
use rig::completion::{Chat, CompletionModel, Message, ToolDefinition};
use rig::message::AssistantContent;
use rig::providers::anthropic;
use serde::Deserialize;
use serde_json::json;

use super::super::support::with_anthropic_cassette;
use crate::support::{
    REQUIRED_ZERO_ARG_TOOL_PROMPT, assert_stream_contains_zero_arg_tool_call_named,
    collect_raw_stream_observation, zero_arg_tool_definition,
};

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
        _context: &mut rig::agent::tool::ToolContext,
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
async fn zero_argument_tool_use_streaming() {
    with_anthropic_cassette(
        "messages_tool_args/zero_argument_tool_use_streaming",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request(REQUIRED_ZERO_ARG_TOOL_PROMPT)
                .preamble("Follow the tool-calling instructions exactly.".to_string())
                .max_tokens(1024)
                .tool(zero_arg_tool_definition("ping"))
                .build();

            let stream = model
                .stream(request)
                .await
                .expect("zero-arg streaming request should start");

            assert_stream_contains_zero_arg_tool_call_named(stream, "ping", true).await;
        },
    )
    .await;
}

#[tokio::test]
async fn zero_argument_tool_use_nonstreaming() {
    with_anthropic_cassette(
        "messages_tool_args/zero_argument_tool_use_nonstreaming",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request(REQUIRED_ZERO_ARG_TOOL_PROMPT)
                .preamble("Follow the tool-calling instructions exactly.".to_string())
                .max_tokens(1024)
                .tool(zero_arg_tool_definition("ping"))
                .build();

            let response = model
                .completion(request)
                .await
                .expect("zero-arg completion should succeed");

            let tool_call = response
                .choice
                .iter()
                .find_map(|content| match content {
                    AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
                    _ => None,
                })
                .expect("response should contain the ping tool call");
            assert_eq!(tool_call.function.name, "ping");
            assert_eq!(
                tool_call.function.arguments,
                json!({}),
                "zero-argument tool_use should surface empty-object arguments"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn nested_arguments_roundtrip_nonstreaming() {
    with_anthropic_cassette(
        "messages_tool_args/nested_arguments_roundtrip_nonstreaming",
        |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_SONNET_4_6)
                .preamble(NESTED_ARGS_PREAMBLE)
                .max_tokens(2048)
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
    with_anthropic_cassette(
        "messages_tool_args/nested_arguments_streaming",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request(NESTED_ARGS_PROMPT)
                .preamble(NESTED_ARGS_PREAMBLE.to_string())
                .max_tokens(2048)
                .tool(rig::agent::tool::tool_definition(&PlanTrip))
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
    with_anthropic_cassette(
        "messages_tool_args/unicode_arguments_streaming",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
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
                .max_tokens(1024)
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
