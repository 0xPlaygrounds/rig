//! Request-side pinning of tool definitions: the cassette replay matcher
//! compares request bodies canonically, so these tests freeze the exact
//! `tools` JSON Gemini receives — rich schema conversion, `#[rig_tool]` macro
//! output, and duplicate-name registration semantics. If schema generation
//! changes (e.g. swapping the handrolled definitions for rmcp-derived ones),
//! replay fails with a body mismatch.

use rig::completion::Message;
use rig::providers::gemini;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::super::agent_run_support::tool_result_texts;
use super::super::support::with_gemini_cassette;
use super::super::tools_support::MathError;
use crate::support::assert_nonempty_response;

#[derive(Deserialize, Serialize)]
struct TripArgs {
    destination: String,
    travellers: u64,
    transport: String,
    waypoints: Vec<String>,
    notes: Option<String>,
}

/// Tool with a deliberately rich JSON schema (nested descriptions, enum,
/// array of strings, optional field) to pin Gemini's schema conversion.
#[derive(Clone, Default)]
struct PlanTrip;

impl Tool for PlanTrip {
    const NAME: &'static str = "plan_trip";
    type Error = MathError;
    type Args = TripArgs;
    type Output = String;

    fn description(&self) -> String {
        "Plan a trip itinerary.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": "Destination city"
                },
                "travellers": {
                    "type": "integer",
                    "description": "Number of travellers"
                },
                "transport": {
                    "type": "string",
                    "enum": ["train", "car", "plane"],
                    "description": "Mode of transport"
                },
                "waypoints": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Cities to pass through, in order"
                },
                "notes": {
                    "type": "string",
                    "description": "Optional free-form notes"
                }
            },
            "required": ["destination", "travellers", "transport", "waypoints"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(format!(
            "itinerary booked: {} travellers to {} by {} via {}",
            args.travellers,
            args.destination,
            args.transport,
            args.waypoints.join(" -> ")
        ))
    }
}

/// A schema with nested objects, arrays and enums reaches Gemini unchanged
/// as `parametersJsonSchema`, and the call it produces deserializes.
#[tokio::test]
async fn rich_json_schema_reaches_gemini_unchanged() {
    with_gemini_cassette(
        "tool_definitions/rich_json_schema_reaches_gemini_unchanged",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_2_5_FLASH))
                .preamble("You are a travel planner. You must use the plan_trip tool to plan trips, then confirm the booking to the user.")
                .temperature(0.0)
                .tool(PlanTrip)
                .default_max_turns(3)
                .build();

            let mut history = Vec::<Message>::new();
            let response = agent
                .chat(
                    "Plan a trip for 2 travellers to Lyon by train, passing through Dijon and then Macon, no notes.",
                    &mut history,
                )
                .await
                .expect("rich-schema tool prompt should succeed");

            let texts: Vec<String> = history.iter().flat_map(tool_result_texts).collect();
            assert_eq!(
                texts,
                vec!["itinerary booked: 2 travellers to Lyon by train via Dijon -> Macon".to_string()],
                "arguments matching the rich schema should deserialize and execute"
            );
            assert_nonempty_response(&response.output());
        },
    )
    .await;
}

mod derive_macro {
    use rig::completion::Message;
    use rig::rig_tool;

    use super::super::super::agent_run_support::tool_result_texts;
    use super::super::super::support::with_gemini_cassette;

    use rig::providers::gemini;

    #[rig_tool(
        description = "Perform basic arithmetic operations",
        params(
            x = "First number in the calculation",
            y = "Second number in the calculation",
            operation = "The operation to perform (add, subtract, multiply, divide)"
        )
    )]
    async fn macro_calculator(
        x: i64,
        y: i64,
        operation: String,
    ) -> Result<i64, rig::tool::ToolExecutionError> {
        match operation.as_str() {
            "add" => Ok(x + y),
            "subtract" => Ok(x - y),
            "multiply" => Ok(x * y),
            "divide" => Ok(x / y),
            _ => Err(rig::tool::ToolExecutionError::other(format!(
                "Unknown operation: {operation}"
            ))),
        }
    }

    #[tokio::test]
    async fn rig_tool_macro_schema_round_trips() {
        with_gemini_cassette(
            "tool_definitions/rig_tool_macro_schema_round_trips",
            |client| async move {
                let agent = rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_2_5_FLASH))
                    .preamble("You must use the macro_calculator tool for arithmetic, then report the result.")
                    .temperature(0.0)
                    .tool(MacroCalculator)
                    .default_max_turns(3)
                    .build();

                let mut history = Vec::<Message>::new();
                let response = agent
                    .chat("Multiply 6 by 7.", &mut history)
                    .await
                    .expect("macro tool prompt should succeed");

                let texts: Vec<String> = history.iter().flat_map(tool_result_texts).collect();
                assert_eq!(
                    texts,
                    vec!["42".to_string()],
                    "the macro-generated tool should execute with the model's arguments"
                );
                assert!(
                    response.output().contains("42"),
                    "final answer should report 42: {response:?}"
                );
            },
        )
        .await;
    }
}
