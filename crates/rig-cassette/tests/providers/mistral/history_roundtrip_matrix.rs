//! Mistral history serialization and response matrix.
//!
//! This matrix exercises the request serializer and response normalizer around
//! caller-owned text history, Unicode-bearing text, and both public completion
//! surfaces.
//!
//! The input space is a 2 × 2 × 2 cross-product:
//!
//! | dimension | values |
//! |---|---|
//! | transport | blocking, streaming |
//! | model | `mistral-small-latest`, `ministral-3b-latest` |
//! | surface | provider-native raw, normalized Rig response |
//!
//! That is 8 recorded cells. Every cell proves the exact serialized history
//! from its fixture and compares the observed response text to those exact
//! blocking or SSE bytes.
//!
//! Coverage ledger: tool-history shapes are covered on OpenAI Chat and
//! OpenRouter by the same matrix. Each explicit test maps to
//! `crates/rig-cassette/fixtures/cassettes/mistral/history_roundtrip_matrix/<test-name>.yaml`.
//!
//! | recorded cells | exact fixture set |
//! |---|---|
//! | all 8 | `crates/rig-cassette/fixtures/cassettes/mistral/history_roundtrip_matrix/{blocking,streaming}_{mistral_small,ministral_3b}_{raw,normalized}_text.yaml` |

use std::sync::{Arc, Mutex};

use anyhow::Result;
use futures::StreamExt as _;
use rig::completion::{CompletionModel, Message};
use rig::message::{AssistantContent, UserContent};
use rig::streaming::{Delta, StreamEvent};
use serde_json::Value;

use super::support::{BoundMistral, with_mistral_history_roundtrip_cassette_result};

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
    }
}

fn expected_text(shape: Shape) -> &'static str {
    match shape {
        Shape::Text => "lantern-42",
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
    }
}

fn request(
    model: &(impl CompletionModel + Clone),
    cell: Cell,
) -> rig::completion::CompletionRequest {
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

async fn run_cell(client: BoundMistral, cell: Cell, observed: SharedObservation) -> Result<()> {
    let model = client.completion(model_name(cell.model));
    let observation = match (cell.transport, cell.surface) {
        (Transport::Blocking, Surface::Raw) => {
            // The provider-native surface: the reply's verbatim JSON, which
            // the driver keeps on `CompletionResponse::raw`, read the way a
            // caller reaching past the normalized view reads it.
            let response = model.completion(request(&model, cell)).await?;
            Observation {
                text: content_text(&response.raw["choices"][0]["message"]["content"]),
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

crate::matrix::case_matrix! {
    wrapper: with_mistral_history_roundtrip_cassette_result, family: history_roundtrip_matrix_case;
    # [tokio :: test]
    blocking_mistral_small_raw_text: ("history_roundtrip_matrix/blocking_mistral_small_raw_text", configured, cell (Transport :: Blocking , ModelVariant :: MistralSmall , Surface :: Raw , Shape :: Text ,));
    # [tokio :: test]
    blocking_mistral_small_normalized_text: ("history_roundtrip_matrix/blocking_mistral_small_normalized_text", configured, cell (Transport :: Blocking , ModelVariant :: MistralSmall , Surface :: Normalized , Shape :: Text ,));
    # [tokio :: test]
    blocking_ministral_3b_raw_text: ("history_roundtrip_matrix/blocking_ministral_3b_raw_text", configured, cell (Transport :: Blocking , ModelVariant :: Ministral3b , Surface :: Raw , Shape :: Text ,));
    # [tokio :: test]
    blocking_ministral_3b_normalized_text: ("history_roundtrip_matrix/blocking_ministral_3b_normalized_text", configured, cell (Transport :: Blocking , ModelVariant :: Ministral3b , Surface :: Normalized , Shape :: Text ,));
    # [tokio :: test]
    streaming_mistral_small_raw_text: ("history_roundtrip_matrix/streaming_mistral_small_raw_text", configured, cell (Transport :: Streaming , ModelVariant :: MistralSmall , Surface :: Raw , Shape :: Text ,));
    # [tokio :: test]
    streaming_mistral_small_normalized_text: ("history_roundtrip_matrix/streaming_mistral_small_normalized_text", configured, cell (Transport :: Streaming , ModelVariant :: MistralSmall , Surface :: Normalized , Shape :: Text ,));
    # [tokio :: test]
    streaming_ministral_3b_raw_text: ("history_roundtrip_matrix/streaming_ministral_3b_raw_text", configured, cell (Transport :: Streaming , ModelVariant :: Ministral3b , Surface :: Raw , Shape :: Text ,));
    # [tokio :: test]
    streaming_ministral_3b_normalized_text: ("history_roundtrip_matrix/streaming_ministral_3b_normalized_text", configured, cell (Transport :: Streaming , ModelVariant :: Ministral3b , Surface :: Normalized , Shape :: Text ,));
}
