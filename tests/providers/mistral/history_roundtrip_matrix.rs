//! Mistral history serialization and response matrix.
//!
//! This matrix exercises the request serializer and response normalizer around
//! caller-owned history, including tool-call ids, arguments, ordered tool
//! results, Unicode-bearing text, and both public completion surfaces.
//!
//! The complete input space exercised here is a 2 × 2 × 2 × 3 cross-product:
//!
//! | dimension | values |
//! |---|---|
//! | transport | blocking, streaming |
//! | model | `mistral-small-latest`, `ministral-3b-latest` |
//! | surface | provider-native raw, normalized Rig response |
//! | history shape | text, one tool result, two ordered tool results |
//!
//! That is 24 recorded cells. Every cell proves the exact serialized history
//! from its fixture and compares the observed response text to those exact
//! blocking or SSE bytes.
//!
//! Coverage ledger: the pre-pruning Cartesian product is 24 and no cell was
//! pruned or assigned to unit-only coverage. Each explicit test maps to
//! `tests/cassettes/mistral/history_roundtrip_matrix/<test-name>.yaml`.
//! The inexpensive small and 3B aliases are stable served families with tool
//! history support. Assertions cover native and normalized blocking/streaming
//! surfaces, Unicode, exact tool ids and arguments, ordered tool results, the
//! current prompt, provider-observed response content, and terminal arrival.
//!
//! | recorded cells | exact fixture set |
//! |---|---|
//! | all 24 | `tests/cassettes/mistral/history_roundtrip_matrix/{blocking,streaming}_{mistral_small,ministral_3b}_{raw,normalized}_{text,single_tool,parallel_tool}.yaml` |

use std::sync::{Arc, Mutex};

use anyhow::Result;
use futures::StreamExt as _;
use rig::completion::{CompletionModel, Message, NormalizeCompletionResponse};
use rig::message::{AssistantContent, ToolResultContent, UserContent};
use rig::prelude::*;
use rig::providers::mistral;
use rig::streaming::{Delta, StreamEvent};
use serde_json::{Value, json};

use super::support::with_mistral_history_roundtrip_cassette_result;

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
enum Surface {
    Raw,
    Normalized,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Text,
    SingleTool,
    ParallelTool,
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    transport: Transport,
    model: ModelVariant,
    surface: Surface,
    shape: Shape,
}

#[derive(Debug)]
struct Observation {
    text: String,
    saw_terminal: bool,
}

type SharedObservation = Arc<Mutex<Option<Observation>>>;

fn prompt(shape: Shape) -> &'static str {
    match shape {
        Shape::Text => "Reply with exactly the marker from the prior conversation.",
        Shape::SingleTool => {
            "The lookup_marker argument is not the answer. Copy exactly the content of its following tool-role result; it is the nonce ZXQ9281."
        }
        Shape::ParallelTool => {
            "Reply with exactly the alpha and beta tool results joined by one hyphen, in call order."
        }
    }
}

fn expected_text(shape: Shape) -> &'static str {
    match shape {
        Shape::Text => "lantern-42",
        Shape::SingleTool => "ZXQ9281",
        Shape::ParallelTool => "red-blue",
    }
}

fn history(shape: Shape) -> Vec<Message> {
    match shape {
        Shape::Text => vec![
            Message::User {
                content: vec![UserContent::text(
                    "Unicode context: café 東京. The marker is exactly: lantern-42.",
                )],
            },
            Message::Assistant {
                id: None,
                content: vec![AssistantContent::text("lantern-42")],
            },
        ],
        Shape::SingleTool => vec![
            Message::Assistant {
                id: None,
                content: vec![AssistantContent::tool_call(
                    "call_history_single",
                    "lookup_marker",
                    json!({ "key": "argument-not-answer" }),
                )],
            },
            Message::User {
                content: vec![UserContent::tool_result(
                    "call_history_single",
                    "lookup_marker",
                    vec![ToolResultContent::text("ZXQ9281")],
                )],
            },
        ],
        Shape::ParallelTool => vec![
            Message::Assistant {
                id: None,
                content: vec![
                    AssistantContent::tool_call(
                        "call_history_alpha",
                        "alpha",
                        json!({ "slot": 1 }),
                    ),
                    AssistantContent::tool_call("call_history_beta", "beta", json!({ "slot": 2 })),
                ],
            },
            Message::User {
                content: vec![
                    UserContent::tool_result(
                        "call_history_alpha",
                        "alpha",
                        vec![ToolResultContent::text("red")],
                    ),
                    UserContent::tool_result(
                        "call_history_beta",
                        "beta",
                        vec![ToolResultContent::text("blue")],
                    ),
                ],
            },
        ],
    }
}

fn request(model: &mistral::CompletionModel, cell: Cell) -> rig::completion::CompletionRequest {
    let mut builder = model.completion_request(prompt(cell.shape)).max_tokens(24);
    for message in history(cell.shape) {
        builder = builder.message(message);
    }
    builder.build()
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
    let observation = match (cell.transport, cell.surface) {
        (Transport::Blocking, Surface::Raw) => {
            let response = model.raw_completion(request(&model, cell)).await?;
            let normalized = response.normalize("mistral")?;
            Observation {
                text: normalized_text(&normalized.choice),
                saw_terminal: true,
            }
        }
        (Transport::Blocking, Surface::Normalized) => {
            let response = model.completion(request(&model, cell)).await?;
            Observation {
                text: normalized_text(&response.choice),
                saw_terminal: true,
            }
        }
        (Transport::Streaming, Surface::Raw) => {
            let mut stream = model.stream(request(&model, cell)).await?;
            let mut observation = Observation {
                text: String::new(),
                saw_terminal: false,
            };
            while let Some(item) = stream.next().await {
                match item? {
                    StreamEvent::BlockDelta {
                        delta: Delta::Text { text },
                        ..
                    } => observation.text.push_str(&text),
                    StreamEvent::Final(_) => observation.saw_terminal = true,
                    _ => {}
                }
            }
            observation
        }
        (Transport::Streaming, Surface::Normalized) => {
            let mut stream = model.stream(request(&model, cell)).await?;
            let mut observation = Observation {
                text: String::new(),
                saw_terminal: false,
            };
            while let Some(item) = stream.next().await {
                match item? {
                    StreamEvent::BlockDelta {
                        delta: Delta::Text { text },
                        ..
                    } => observation.text.push_str(&text),
                    StreamEvent::Final(_) => observation.saw_terminal = true,
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

fn recorded_text(scenario: &str, transport: Transport) -> String {
    match transport {
        Transport::Blocking => {
            content_text(&recorded_response(scenario)["choices"][0]["message"]["content"])
        }
        Transport::Streaming => recorded_stream_chunks(scenario)
            .iter()
            .flat_map(|chunk| chunk["choices"].as_array().into_iter().flatten())
            .filter(|choice| choice["index"].as_u64() == Some(0))
            .map(|choice| content_text(&choice["delta"]["content"]))
            .collect(),
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
    assert_eq!(request["max_tokens"], 24, "{scenario}: output cap");
    assert_eq!(
        request["stream"].as_bool().unwrap_or(false),
        cell.transport == Transport::Streaming,
        "{scenario}: transport"
    );
    assert!(request.get("tools").is_none(), "{scenario}: history only");

    let messages = request["messages"]
        .as_array()
        .expect("recorded request messages");
    match cell.shape {
        Shape::Text => {
            assert_eq!(messages.len(), 3, "{scenario}: text history length");
            assert_eq!(messages[0]["role"], "user", "{scenario}");
            assert!(
                content_text(&messages[0]["content"]).contains("café 東京"),
                "{scenario}: Unicode history survives"
            );
            assert_eq!(messages[1]["role"], "assistant", "{scenario}");
            assert_eq!(content_text(&messages[1]["content"]), "lantern-42");
        }
        Shape::SingleTool => {
            assert_eq!(messages.len(), 3, "{scenario}: single-tool history length");
            assert_eq!(messages[0]["role"], "assistant", "{scenario}");
            let calls = messages[0]["tool_calls"]
                .as_array()
                .expect("assistant tool calls");
            assert_eq!(calls.len(), 1, "{scenario}");
            assert_eq!(calls[0]["function"]["name"], "lookup_marker", "{scenario}");
            assert_eq!(
                serde_json::from_str::<Value>(
                    calls[0]["function"]["arguments"]
                        .as_str()
                        .expect("tool arguments string")
                )
                .expect("tool arguments JSON"),
                json!({ "key": "argument-not-answer" }),
                "{scenario}"
            );
            assert_eq!(messages[1]["role"], "tool", "{scenario}");
            assert_eq!(messages[1]["tool_call_id"], calls[0]["id"], "{scenario}");
            assert_eq!(content_text(&messages[1]["content"]), "ZXQ9281");
        }
        Shape::ParallelTool => {
            assert_eq!(messages.len(), 4, "{scenario}: parallel history length");
            assert_eq!(messages[0]["role"], "assistant", "{scenario}");
            let calls = messages[0]["tool_calls"]
                .as_array()
                .expect("assistant tool calls");
            assert_eq!(calls.len(), 2, "{scenario}");
            assert_eq!(calls[0]["function"]["name"], "alpha", "{scenario}");
            assert_eq!(calls[1]["function"]["name"], "beta", "{scenario}");
            for (index, (name, slot, result)) in [("alpha", 1, "red"), ("beta", 2, "blue")]
                .into_iter()
                .enumerate()
            {
                assert_eq!(
                    serde_json::from_str::<Value>(
                        calls[index]["function"]["arguments"]
                            .as_str()
                            .expect("tool arguments string")
                    )
                    .expect("tool arguments JSON"),
                    json!({ "slot": slot }),
                    "{scenario}: {name} arguments"
                );
                assert_eq!(messages[index + 1]["role"], "tool", "{scenario}");
                assert_eq!(
                    messages[index + 1]["tool_call_id"],
                    calls[index]["id"],
                    "{scenario}: {name} correlation id"
                );
                assert_eq!(
                    content_text(&messages[index + 1]["content"]),
                    result,
                    "{scenario}: {name} result"
                );
            }
        }
    }
    assert_eq!(
        messages.last().expect("current prompt")["role"],
        "user",
        "{scenario}: current prompt role"
    );
    assert_eq!(
        content_text(&messages.last().expect("current prompt")["content"]),
        prompt(cell.shape),
        "{scenario}: current prompt text"
    );

    let wire_text = recorded_text(scenario, cell.transport);
    assert!(
        wire_text.contains(expected_text(cell.shape)),
        "{scenario}: recorded provider used the expected historical value: {wire_text:?}"
    );

    let observation = observed
        .lock()
        .expect("observation mutex poisoned")
        .take()
        .expect("test body should save an observation");
    assert!(observation.saw_terminal, "{scenario}: terminal observed");
    assert_eq!(observation.text, wire_text, "{scenario}: surface text");
}

fn cell(transport: Transport, model: ModelVariant, surface: Surface, shape: Shape) -> Cell {
    Cell {
        transport,
        model,
        surface,
        shape,
    }
}

// Explicit cells keep the cassette source scanner able to prove a one-to-one
// mapping between tests and fixtures.

#[tokio::test]
async fn blocking_mistral_small_raw_text() -> Result<()> {
    const SCENARIO: &str = "history_roundtrip_matrix/blocking_mistral_small_raw_text";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::MistralSmall,
        Surface::Raw,
        Shape::Text,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/blocking_mistral_small_raw_text",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_mistral_small_raw_single_tool() -> Result<()> {
    const SCENARIO: &str = "history_roundtrip_matrix/blocking_mistral_small_raw_single_tool";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::MistralSmall,
        Surface::Raw,
        Shape::SingleTool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/blocking_mistral_small_raw_single_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_mistral_small_raw_parallel_tool() -> Result<()> {
    const SCENARIO: &str = "history_roundtrip_matrix/blocking_mistral_small_raw_parallel_tool";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::MistralSmall,
        Surface::Raw,
        Shape::ParallelTool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/blocking_mistral_small_raw_parallel_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_mistral_small_normalized_text() -> Result<()> {
    const SCENARIO: &str = "history_roundtrip_matrix/blocking_mistral_small_normalized_text";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::MistralSmall,
        Surface::Normalized,
        Shape::Text,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/blocking_mistral_small_normalized_text",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_mistral_small_normalized_single_tool() -> Result<()> {
    const SCENARIO: &str = "history_roundtrip_matrix/blocking_mistral_small_normalized_single_tool";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::MistralSmall,
        Surface::Normalized,
        Shape::SingleTool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/blocking_mistral_small_normalized_single_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_mistral_small_normalized_parallel_tool() -> Result<()> {
    const SCENARIO: &str =
        "history_roundtrip_matrix/blocking_mistral_small_normalized_parallel_tool";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::MistralSmall,
        Surface::Normalized,
        Shape::ParallelTool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/blocking_mistral_small_normalized_parallel_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_raw_text() -> Result<()> {
    const SCENARIO: &str = "history_roundtrip_matrix/blocking_ministral_3b_raw_text";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Ministral3b,
        Surface::Raw,
        Shape::Text,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/blocking_ministral_3b_raw_text",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_raw_single_tool() -> Result<()> {
    const SCENARIO: &str = "history_roundtrip_matrix/blocking_ministral_3b_raw_single_tool";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Ministral3b,
        Surface::Raw,
        Shape::SingleTool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/blocking_ministral_3b_raw_single_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_raw_parallel_tool() -> Result<()> {
    const SCENARIO: &str = "history_roundtrip_matrix/blocking_ministral_3b_raw_parallel_tool";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Ministral3b,
        Surface::Raw,
        Shape::ParallelTool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/blocking_ministral_3b_raw_parallel_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_normalized_text() -> Result<()> {
    const SCENARIO: &str = "history_roundtrip_matrix/blocking_ministral_3b_normalized_text";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Ministral3b,
        Surface::Normalized,
        Shape::Text,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/blocking_ministral_3b_normalized_text",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_normalized_single_tool() -> Result<()> {
    const SCENARIO: &str = "history_roundtrip_matrix/blocking_ministral_3b_normalized_single_tool";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Ministral3b,
        Surface::Normalized,
        Shape::SingleTool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/blocking_ministral_3b_normalized_single_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_normalized_parallel_tool() -> Result<()> {
    const SCENARIO: &str =
        "history_roundtrip_matrix/blocking_ministral_3b_normalized_parallel_tool";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Ministral3b,
        Surface::Normalized,
        Shape::ParallelTool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/blocking_ministral_3b_normalized_parallel_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_raw_text() -> Result<()> {
    const SCENARIO: &str = "history_roundtrip_matrix/streaming_mistral_small_raw_text";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::MistralSmall,
        Surface::Raw,
        Shape::Text,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/streaming_mistral_small_raw_text",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_raw_single_tool() -> Result<()> {
    const SCENARIO: &str = "history_roundtrip_matrix/streaming_mistral_small_raw_single_tool";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::MistralSmall,
        Surface::Raw,
        Shape::SingleTool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/streaming_mistral_small_raw_single_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_raw_parallel_tool() -> Result<()> {
    const SCENARIO: &str = "history_roundtrip_matrix/streaming_mistral_small_raw_parallel_tool";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::MistralSmall,
        Surface::Raw,
        Shape::ParallelTool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/streaming_mistral_small_raw_parallel_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_normalized_text() -> Result<()> {
    const SCENARIO: &str = "history_roundtrip_matrix/streaming_mistral_small_normalized_text";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::MistralSmall,
        Surface::Normalized,
        Shape::Text,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/streaming_mistral_small_normalized_text",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_normalized_single_tool() -> Result<()> {
    const SCENARIO: &str =
        "history_roundtrip_matrix/streaming_mistral_small_normalized_single_tool";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::MistralSmall,
        Surface::Normalized,
        Shape::SingleTool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/streaming_mistral_small_normalized_single_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_normalized_parallel_tool() -> Result<()> {
    const SCENARIO: &str =
        "history_roundtrip_matrix/streaming_mistral_small_normalized_parallel_tool";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::MistralSmall,
        Surface::Normalized,
        Shape::ParallelTool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/streaming_mistral_small_normalized_parallel_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_raw_text() -> Result<()> {
    const SCENARIO: &str = "history_roundtrip_matrix/streaming_ministral_3b_raw_text";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Ministral3b,
        Surface::Raw,
        Shape::Text,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/streaming_ministral_3b_raw_text",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_raw_single_tool() -> Result<()> {
    const SCENARIO: &str = "history_roundtrip_matrix/streaming_ministral_3b_raw_single_tool";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Ministral3b,
        Surface::Raw,
        Shape::SingleTool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/streaming_ministral_3b_raw_single_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_raw_parallel_tool() -> Result<()> {
    const SCENARIO: &str = "history_roundtrip_matrix/streaming_ministral_3b_raw_parallel_tool";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Ministral3b,
        Surface::Raw,
        Shape::ParallelTool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/streaming_ministral_3b_raw_parallel_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_normalized_text() -> Result<()> {
    const SCENARIO: &str = "history_roundtrip_matrix/streaming_ministral_3b_normalized_text";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Ministral3b,
        Surface::Normalized,
        Shape::Text,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/streaming_ministral_3b_normalized_text",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_normalized_single_tool() -> Result<()> {
    const SCENARIO: &str = "history_roundtrip_matrix/streaming_ministral_3b_normalized_single_tool";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Ministral3b,
        Surface::Normalized,
        Shape::SingleTool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/streaming_ministral_3b_normalized_single_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_normalized_parallel_tool() -> Result<()> {
    const SCENARIO: &str =
        "history_roundtrip_matrix/streaming_ministral_3b_normalized_parallel_tool";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Ministral3b,
        Surface::Normalized,
        Shape::ParallelTool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_history_roundtrip_cassette_result(
        "history_roundtrip_matrix/streaming_ministral_3b_normalized_parallel_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}
