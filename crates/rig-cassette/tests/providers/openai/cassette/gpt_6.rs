//! GPT-6 cassette coverage.
//!
//! GPT-6 Sol reasons by default and calls tools with reasoning only through
//! the Responses API. The cell runs an agent tool loop there, stateless
//! (`store: false`), so each follow-up replays the reasoning items with their
//! encrypted content.

use rig::providers::openai;
use serde_json::{Value, json};

use super::super::support::with_openai_cassette;
use crate::support::{Adder, STREAMING_TOOLS_PREAMBLE, Subtract, assert_mentions_expected_number};

/// Every recorded request body of `scenario`, decoded, in wire order.
fn recorded_requests(scenario: &str) -> Vec<Value> {
    crate::cassettes::recorded_interaction_bodies("openai", scenario)
        .into_iter()
        .map(|(request, _)| serde_json::from_str(&request).expect("request body is JSON"))
        .collect()
}

#[test]
fn model_constants() {
    assert_eq!(openai::GPT_6_ASTRA, "gpt-6-astra");
    assert_eq!(openai::GPT_6_SOL, "gpt-6-sol");
    assert_eq!(openai::GPT_6_LUNA, "gpt-6-luna");
}

/// A task with several tool calls, so the model reasons between them.
const MULTI_STEP_PROMPT: &str = "Compute (17 - 5) + (8 - 23). Use one tool call per step.";

#[tokio::test]
async fn sol_responses_tool_loop() {
    with_openai_cassette("gpt_6/sol_responses_tool_loop", |client| async move {
        let agent = rig::AgentBuilder::new(client.openai.responses(openai::GPT_6_SOL))
            .preamble(STREAMING_TOOLS_PREAMBLE)
            .tool(Adder)
            .tool(Subtract)
            .additional_params(json!({"store": false, "reasoning": {"effort": "high"}}))
            .default_max_turns(6)
            .build();

        let response = agent
            .prompt(MULTI_STEP_PROMPT)
            .await
            .expect("the tool loop completes")
            .output;

        assert_mentions_expected_number(&response, -3);
    })
    .await;

    let requests = recorded_requests("gpt_6/sol_responses_tool_loop");
    assert!(requests.len() >= 2, "tool calls and their follow-ups");
    for request in &requests {
        assert_eq!(request["model"], openai::GPT_6_SOL);
        assert_eq!(request["store"], false);
        assert_eq!(request.get("temperature"), None);
    }
    let replayed: Vec<&Value> = requests
        .iter()
        .flat_map(|request| request["input"].as_array().into_iter().flatten())
        .collect();
    assert!(
        replayed
            .iter()
            .any(|item| item["type"] == "function_call_output"),
        "a follow-up carries a tool result"
    );
    assert!(
        replayed
            .iter()
            .filter(|item| item["type"] == "reasoning")
            .all(|item| item["encrypted_content"].is_string()),
        "stateless replay sends each reasoning item with its encrypted content"
    );
}
