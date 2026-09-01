//! Mistral request finalization matrix.
//!
//! Mistral rejects structured output beside a forced tool choice. Rig's
//! provider finalizer must preserve `auto`/`none`, translate `required` to
//! Mistral's `any` dialect for plain output, and relax that forced choice back
//! to `auto` when `json_object` is present.
//!
//! The complete input space exercised here is a 2 × 2 × 2 × 3 cross-product:
//!
//! | dimension | values |
//! |---|---|
//! | transport | blocking, streaming |
//! | model | `mistral-small-latest`, `ministral-3b-latest` |
//! | response format | plain text, `json_object` |
//! | tool policy | `auto`, forced (`any`), `none` |
//!
//! That is 24 recorded cells. Every cell proves the finalized wire request and
//! compares normalized text/tool output to the exact blocking or SSE response.
//!
//! Coverage ledger: the pre-pruning Cartesian product is 24 and all 24 cells
//! are recorded; none is unit-only. Each explicit test maps to
//! `tests/cassettes/mistral/request_shape_matrix/<test-name>.yaml`.
//! The small and 3B aliases are inexpensive current models spanning Mistral's
//! served chat families. Assertions cover finalized blocking/streaming request
//! fields, `tool_choice` translation, structured-output compatibility, exact
//! recorded text or tool calls, and terminal finish reasons. Invalid raw value
//! types that the live API cannot be asked to synthesize remain unit tests.
//!
//! | recorded cells | exact fixture set |
//! |---|---|
//! | all 24 | `tests/cassettes/mistral/request_shape_matrix/{blocking,streaming}_{mistral_small,ministral_3b}_{plain,json}_{auto,any,none}.yaml` |

use std::sync::{Arc, Mutex};

use anyhow::Result;
use futures::StreamExt as _;
use rig::completion::{CompletionModel, FinishReason};
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::mistral;
use rig::streaming::StreamedAssistantContent;
use serde_json::{Value, json};

use super::support::with_mistral_request_shape_cassette_result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Transport {
    Blocking,
    Streaming,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelVariant {
    MistralSmall,
    Ministral3b,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Plain,
    Json,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ToolPolicy {
    Auto,
    Any,
    None,
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    transport: Transport,
    model: ModelVariant,
    format: Format,
    tool_policy: ToolPolicy,
}

#[derive(Debug, Default)]
struct Observation {
    text: String,
    finish_reason: Option<FinishReason>,
    calls: Vec<(String, Value)>,
}

type SharedObservation = Arc<Mutex<Option<Observation>>>;

fn prompt(cell: Cell) -> &'static str {
    match (cell.format, cell.tool_policy) {
        (Format::Plain, ToolPolicy::Any) => {
            "Call add exactly once with x=17 and y=25. Do not answer in prose."
        }
        (Format::Plain, _) => "Reply with exactly: cobalt",
        (Format::Json, _) => "Return only this JSON object and do not call tools: {\"total\":42}",
    }
}

fn tool_definition() -> rig::completion::ToolDefinition {
    rig::completion::ToolDefinition {
        name: "add".to_owned(),
        description: "Add two integers".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": {
                "x": { "type": "integer" },
                "y": { "type": "integer" }
            },
            "required": ["x", "y"]
        }),
    }
}

fn request(model: &mistral::CompletionModel, cell: Cell) -> rig::completion::CompletionRequest {
    let mut params = json!({
        "tool_choice": match cell.tool_policy {
            ToolPolicy::Auto => "auto",
            ToolPolicy::Any => "any",
            ToolPolicy::None => "none",
        }
    });
    if cell.format == Format::Json {
        params["response_format"] = json!({ "type": "json_object" });
    }

    model
        .completion_request(prompt(cell))
        .tool(tool_definition())
        .additional_params(params)
        .max_tokens(64)
        .build()
}

fn normalized_text(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

fn model_name(model: ModelVariant) -> &'static str {
    match model {
        ModelVariant::MistralSmall => "mistral-small-latest",
        ModelVariant::Ministral3b => "ministral-3b-latest",
    }
}

async fn run_cell(client: mistral::Client, cell: Cell, observed: SharedObservation) -> Result<()> {
    let model = client.completion_model(model_name(cell.model));
    let observation = match cell.transport {
        Transport::Blocking => {
            let response = model.completion(request(&model, cell)).await?;
            let calls = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::ToolCall(call) => {
                        Some((call.function.name.clone(), call.function.arguments.clone()))
                    }
                    _ => None,
                })
                .collect();
            Observation {
                text: normalized_text(&response.choice),
                finish_reason: response.finish_reason(),
                calls,
            }
        }
        Transport::Streaming => {
            let mut stream = model.stream(request(&model, cell)).await?;
            let mut observation = Observation::default();
            while let Some(item) = stream.next().await {
                match item? {
                    StreamedAssistantContent::Text(text) => observation.text.push_str(&text.text),
                    StreamedAssistantContent::ToolCall { tool_call, .. } => observation
                        .calls
                        .push((tool_call.function.name, tool_call.function.arguments)),
                    StreamedAssistantContent::Final(final_record) => {
                        observation.finish_reason = final_record.finish_reason;
                    }
                    _ => {}
                }
            }
            observation
        }
    };

    *observed.lock().expect("observation mutex poisoned") = Some(observation);
    Ok(())
}

fn content_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect(),
        _ => String::new(),
    }
}

#[derive(Debug)]
struct RecordedOutcome {
    finish: String,
    text: String,
    calls: Vec<(String, Value)>,
}

fn recorded_outcome(scenario: &str, transport: Transport) -> RecordedOutcome {
    match transport {
        Transport::Blocking => {
            let response = recorded_response(scenario);
            let choice = &response["choices"][0];
            let calls = choice["message"]["tool_calls"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|call| {
                    (
                        call["function"]["name"]
                            .as_str()
                            .unwrap_or_default()
                            .to_owned(),
                        serde_json::from_str(
                            call["function"]["arguments"].as_str().unwrap_or_default(),
                        )
                        .expect("recorded tool arguments should be JSON"),
                    )
                })
                .collect();
            RecordedOutcome {
                finish: choice["finish_reason"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
                text: content_text(&choice["message"]["content"]),
                calls,
            }
        }
        Transport::Streaming => {
            let mut finish = String::new();
            let mut text = String::new();
            let mut calls = Vec::<(String, String)>::new();
            for chunk in recorded_stream_chunks(scenario) {
                for choice in chunk["choices"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|choice| choice["index"].as_u64() == Some(0))
                {
                    text.push_str(&content_text(&choice["delta"]["content"]));
                    if let Some(reason) = choice["finish_reason"].as_str() {
                        finish = reason.to_owned();
                    }
                    for call in choice["delta"]["tool_calls"]
                        .as_array()
                        .into_iter()
                        .flatten()
                    {
                        let index = call["index"].as_u64().unwrap_or(0) as usize;
                        while calls.len() <= index {
                            calls.push((String::new(), String::new()));
                        }
                        if let Some(name) = call["function"]["name"].as_str() {
                            calls[index].0.push_str(name);
                        }
                        if let Some(arguments) = call["function"]["arguments"].as_str() {
                            calls[index].1.push_str(arguments);
                        }
                    }
                }
            }
            RecordedOutcome {
                finish,
                text,
                calls: calls
                    .into_iter()
                    .map(|(name, arguments)| {
                        (
                            name,
                            serde_json::from_str(&arguments)
                                .expect("recorded streamed tool arguments should be JSON"),
                        )
                    })
                    .collect(),
            }
        }
    }
}

fn recorded_request(scenario: &str) -> Value {
    crate::cassettes::recorded_json_request("mistral", scenario)
}

fn recorded_response(scenario: &str) -> Value {
    crate::cassettes::recorded_json_response("mistral", scenario)
}

fn recorded_stream_chunks(scenario: &str) -> Vec<Value> {
    crate::cassettes::recorded_sse_json_frames("mistral", scenario)
}

fn assert_cell(scenario: &str, cell: Cell, observed: SharedObservation) {
    let request = recorded_request(scenario);
    assert_eq!(
        request["model"],
        model_name(cell.model),
        "{scenario}: model"
    );
    assert_eq!(request["max_tokens"], 64, "{scenario}: output cap");
    assert_eq!(
        request["stream"].as_bool().unwrap_or(false),
        cell.transport == Transport::Streaming,
        "{scenario}: transport"
    );
    assert_eq!(
        request["tools"].as_array().map(Vec::len),
        Some(1),
        "{scenario}: tool definition"
    );
    assert_eq!(request["tools"][0]["function"]["name"], "add", "{scenario}");
    let expected_tool_choice = match (cell.tool_policy, cell.format) {
        (ToolPolicy::Any, Format::Json) => "auto",
        (ToolPolicy::Any, Format::Plain) => "any",
        (ToolPolicy::Auto, _) => "auto",
        (ToolPolicy::None, _) => "none",
    };
    assert_eq!(
        request["tool_choice"], expected_tool_choice,
        "{scenario}: finalized Mistral tool choice"
    );
    match cell.format {
        Format::Plain => assert!(
            request.get("response_format").is_none(),
            "{scenario}: plain request has no response format"
        ),
        Format::Json => assert_eq!(
            request["response_format"],
            json!({ "type": "json_object" }),
            "{scenario}: JSON-object request"
        ),
    }

    let wire = recorded_outcome(scenario, cell.transport);
    let forced_plain = cell.tool_policy == ToolPolicy::Any && cell.format == Format::Plain;
    if forced_plain {
        assert_eq!(wire.finish, "tool_calls", "{scenario}: tool terminal");
        assert_eq!(
            wire.calls,
            vec![("add".to_owned(), json!({ "x": 17, "y": 25 }))],
            "{scenario}: exact recorded call"
        );
    } else if cell.tool_policy == ToolPolicy::None {
        assert_eq!(wire.finish, "stop", "{scenario}: none is text-only");
        assert!(wire.calls.is_empty(), "{scenario}: none forbids tool calls");
    } else {
        assert!(
            matches!(wire.finish.as_str(), "stop" | "tool_calls"),
            "{scenario}: auto terminal: {}",
            wire.finish
        );
        if wire.finish == "tool_calls" {
            assert!(
                wire.calls
                    .iter()
                    .all(|(name, arguments)| name == "add" && arguments.is_object()),
                "{scenario}: auto may only call the offered add tool: {:?}",
                wire.calls
            );
        } else {
            assert!(wire.calls.is_empty(), "{scenario}: stop has no tool calls");
        }
    }

    match cell.format {
        Format::Plain if wire.finish == "stop" => assert!(
            wire.text.to_ascii_lowercase().contains("cobalt"),
            "{scenario}: plain response premise: {:?}",
            wire.text
        ),
        Format::Json if !wire.text.trim().is_empty() => {
            let value: Value = serde_json::from_str(wire.text.trim())
                .expect("Mistral json_object response should be valid JSON");
            assert!(
                value.as_object().is_some_and(|object| !object.is_empty()),
                "{scenario}: JSON response should be a non-empty object: {value}"
            );
        }
        _ => {}
    }

    let observation = observed
        .lock()
        .expect("observation mutex poisoned")
        .take()
        .expect("test body should save an observation");
    assert_eq!(observation.text, wire.text, "{scenario}: normalized text");
    assert_eq!(
        observation.calls, wire.calls,
        "{scenario}: normalized calls"
    );
    assert_eq!(
        observation.finish_reason,
        Some(if wire.finish == "tool_calls" {
            FinishReason::ToolCalls
        } else {
            FinishReason::Stop
        }),
        "{scenario}: normalized finish reason"
    );
}

fn cell(
    transport: Transport,
    model: ModelVariant,
    format: Format,
    tool_policy: ToolPolicy,
) -> Cell {
    Cell {
        transport,
        model,
        format,
        tool_policy,
    }
}

// Explicit cells keep the cassette source scanner able to prove a one-to-one
// mapping between tests and fixtures.

#[tokio::test]
async fn blocking_mistral_small_plain_auto() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/blocking_mistral_small_plain_auto";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::MistralSmall,
        Format::Plain,
        ToolPolicy::Auto,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/blocking_mistral_small_plain_auto",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_mistral_small_plain_any() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/blocking_mistral_small_plain_any";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::MistralSmall,
        Format::Plain,
        ToolPolicy::Any,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/blocking_mistral_small_plain_any",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_mistral_small_plain_none() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/blocking_mistral_small_plain_none";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::MistralSmall,
        Format::Plain,
        ToolPolicy::None,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/blocking_mistral_small_plain_none",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_mistral_small_json_auto() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/blocking_mistral_small_json_auto";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::MistralSmall,
        Format::Json,
        ToolPolicy::Auto,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/blocking_mistral_small_json_auto",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_mistral_small_json_any() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/blocking_mistral_small_json_any";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::MistralSmall,
        Format::Json,
        ToolPolicy::Any,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/blocking_mistral_small_json_any",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_mistral_small_json_none() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/blocking_mistral_small_json_none";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::MistralSmall,
        Format::Json,
        ToolPolicy::None,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/blocking_mistral_small_json_none",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_plain_auto() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/blocking_ministral_3b_plain_auto";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Ministral3b,
        Format::Plain,
        ToolPolicy::Auto,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/blocking_ministral_3b_plain_auto",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_plain_any() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/blocking_ministral_3b_plain_any";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Ministral3b,
        Format::Plain,
        ToolPolicy::Any,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/blocking_ministral_3b_plain_any",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_plain_none() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/blocking_ministral_3b_plain_none";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Ministral3b,
        Format::Plain,
        ToolPolicy::None,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/blocking_ministral_3b_plain_none",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_json_auto() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/blocking_ministral_3b_json_auto";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Ministral3b,
        Format::Json,
        ToolPolicy::Auto,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/blocking_ministral_3b_json_auto",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_json_any() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/blocking_ministral_3b_json_any";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Ministral3b,
        Format::Json,
        ToolPolicy::Any,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/blocking_ministral_3b_json_any",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_json_none() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/blocking_ministral_3b_json_none";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Ministral3b,
        Format::Json,
        ToolPolicy::None,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/blocking_ministral_3b_json_none",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_plain_auto() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/streaming_mistral_small_plain_auto";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::MistralSmall,
        Format::Plain,
        ToolPolicy::Auto,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/streaming_mistral_small_plain_auto",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_plain_any() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/streaming_mistral_small_plain_any";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::MistralSmall,
        Format::Plain,
        ToolPolicy::Any,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/streaming_mistral_small_plain_any",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_plain_none() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/streaming_mistral_small_plain_none";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::MistralSmall,
        Format::Plain,
        ToolPolicy::None,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/streaming_mistral_small_plain_none",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_json_auto() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/streaming_mistral_small_json_auto";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::MistralSmall,
        Format::Json,
        ToolPolicy::Auto,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/streaming_mistral_small_json_auto",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_json_any() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/streaming_mistral_small_json_any";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::MistralSmall,
        Format::Json,
        ToolPolicy::Any,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/streaming_mistral_small_json_any",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_json_none() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/streaming_mistral_small_json_none";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::MistralSmall,
        Format::Json,
        ToolPolicy::None,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/streaming_mistral_small_json_none",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_plain_auto() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/streaming_ministral_3b_plain_auto";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Ministral3b,
        Format::Plain,
        ToolPolicy::Auto,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/streaming_ministral_3b_plain_auto",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_plain_any() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/streaming_ministral_3b_plain_any";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Ministral3b,
        Format::Plain,
        ToolPolicy::Any,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/streaming_ministral_3b_plain_any",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_plain_none() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/streaming_ministral_3b_plain_none";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Ministral3b,
        Format::Plain,
        ToolPolicy::None,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/streaming_ministral_3b_plain_none",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_json_auto() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/streaming_ministral_3b_json_auto";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Ministral3b,
        Format::Json,
        ToolPolicy::Auto,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/streaming_ministral_3b_json_auto",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_json_any() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/streaming_ministral_3b_json_any";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Ministral3b,
        Format::Json,
        ToolPolicy::Any,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/streaming_ministral_3b_json_any",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_json_none() -> Result<()> {
    const SCENARIO: &str = "request_shape_matrix/streaming_ministral_3b_json_none";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Ministral3b,
        Format::Json,
        ToolPolicy::None,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_request_shape_cassette_result(
        "request_shape_matrix/streaming_ministral_3b_json_none",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}
