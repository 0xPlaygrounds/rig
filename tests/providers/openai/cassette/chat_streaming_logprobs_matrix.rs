//! Exhaustive matrix for streamed Chat Completions log probabilities.
//!
//! Before this fix, `StreamingChoice` did not model `logprobs`, so serde
//! discarded every per-token object before the shared compatible adapter saw
//! it. Blocking `raw_completion` retained the same field on `Choice`; raw
//! streaming therefore carried strictly less provider-native information.
//!
//! The complete input space exercised here is a 2 × 2 × 2 × 3 cross-product:
//!
//! | dimension | values |
//! |---|---|
//! | transport | blocking control, raw streaming regression |
//! | model | `gpt-4o-mini`, `gpt-4.1-mini` |
//! | termination | natural `stop`, one-token `length` |
//! | top candidates | `top_logprobs` absent, `0`, `2` |
//!
//! That is 24 recorded cells. Each one re-derives the exact expected
//! log-probability object from its fixture and compares it with the serialized
//! native response. Streaming cells recursively concatenate token arrays in
//! wire order. The two model families are deliberate: this is shared Chat
//! Completions metadata rather than a one-model quirk.
//!
//! Coverage ledger: the pre-pruning Cartesian product is 24 and no cell was
//! pruned or replaced by a unit test. Each explicit test below maps to
//! `tests/cassettes/openai/chat_streaming_logprobs_matrix/<test-name>.yaml`.
//! Synthetic `logprobs` values (`null`, `{}`, and a non-object) stay in the
//! shared deserializer's unit tests because a live model cannot be instructed
//! to emit them. The two inexpensive mini models are stable Chat Completions
//! controls; every recorded cell asserts the request, provider-native blocking
//! response or raw-stream terminal, finish reason, and complete probability
//! payload.
//!
//! | recorded cells | exact fixture set |
//! |---|---|
//! | all 24 | `tests/cassettes/openai/chat_streaming_logprobs_matrix/{blocking,streaming}_{gpt_4o_mini,gpt_4_1_mini}_{stop,length}_top_{absent,zero,two}.yaml` |

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use futures::StreamExt as _;
use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::openai;
use rig::streaming::RawStreamingChoice;
use serde_json::{Value, json};

use super::super::support::with_openai_chat_stream_logprobs_cassette_result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Transport {
    Blocking,
    Streaming,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelVariant {
    Gpt4oMini,
    Gpt41Mini,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Termination {
    Stop,
    Length,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Top {
    Absent,
    Zero,
    Two,
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    transport: Transport,
    model: ModelVariant,
    termination: Termination,
    top: Top,
}

#[derive(Debug)]
struct Observation {
    logprobs: Value,
    finish_reason: Value,
}

type SharedObservation = Arc<Mutex<Option<Observation>>>;

fn params(cell: Cell) -> Value {
    let mut params = json!({ "logprobs": true });
    if let Some(top) = match cell.top {
        Top::Absent => None,
        Top::Zero => Some(0),
        Top::Two => Some(2),
    } {
        params["top_logprobs"] = json!(top);
    }
    params
}

fn prompt(cell: Cell) -> &'static str {
    match (cell.model, cell.termination) {
        (ModelVariant::Gpt4oMini, Termination::Stop) => "Reply with exactly: cobalt",
        (ModelVariant::Gpt4oMini, Termination::Length) => {
            "Write the English alphabet in order without spaces."
        }
        (ModelVariant::Gpt41Mini, Termination::Stop) => {
            "What is 17 + 25? Answer with only the number."
        }
        (ModelVariant::Gpt41Mini, Termination::Length) => {
            "Work out 17 + 25 carefully, then answer with only the number."
        }
    }
}

fn max_tokens(cell: Cell) -> u64 {
    match (cell.model, cell.termination) {
        (_, Termination::Length) => 1,
        (_, Termination::Stop) => 8,
    }
}

fn model_name(model: ModelVariant) -> &'static str {
    match model {
        ModelVariant::Gpt4oMini => "gpt-4o-mini",
        ModelVariant::Gpt41Mini => "gpt-4.1-mini",
    }
}

async fn run_cell(client: openai::Client, cell: Cell, observed: SharedObservation) -> Result<()> {
    let model = client
        .completions_api()
        .completion_model(model_name(cell.model));
    let request = model
        .completion_request(prompt(cell))
        .additional_params(params(cell))
        .max_tokens(max_tokens(cell))
        .build();

    let observation = match cell.transport {
        Transport::Blocking => {
            let raw = model.raw_completion(request).await?;
            let choice = raw
                .choices
                .first()
                .context("blocking response should carry a choice")?;
            Observation {
                logprobs: choice.logprobs.clone().unwrap_or(Value::Null),
                finish_reason: json!(choice.finish_reason),
            }
        }
        Transport::Streaming => {
            let mut stream = model.raw_stream(request).await?;
            let mut terminal = None;
            while let Some(item) = stream.next().await {
                if let RawStreamingChoice::FinalResponse(response) = item? {
                    terminal = Some(response);
                }
            }
            let terminal = terminal.context("raw stream should carry a terminal response")?;
            let serialized = serde_json::to_value(terminal)?;
            Observation {
                logprobs: serialized["logprobs"].clone(),
                finish_reason: serialized["finish_reason"].clone(),
            }
        }
    };

    *observed.lock().expect("observation mutex poisoned") = Some(observation);
    Ok(())
}

fn merge_json(existing: &mut Value, incoming: Value) {
    match (existing, incoming) {
        (Value::Object(existing), Value::Object(incoming)) => {
            for (key, incoming) in incoming {
                match existing.get_mut(&key) {
                    Some(existing) => merge_json(existing, incoming),
                    None => {
                        existing.insert(key, incoming);
                    }
                }
            }
        }
        (Value::Array(existing), Value::Array(mut incoming)) => existing.append(&mut incoming),
        (existing, incoming) => *existing = incoming,
    }
}

fn recorded_stream_logprobs(scenario: &str) -> Value {
    let mut accumulated = Value::Null;
    for chunk in recorded_stream_chunks(scenario) {
        let Some(choice) = chunk["choices"].as_array().and_then(|choices| {
            choices
                .iter()
                .find(|choice| choice["index"].as_u64() == Some(0))
        }) else {
            continue;
        };
        let logprobs = &choice["logprobs"];
        if logprobs.is_null() {
            continue;
        }
        if accumulated.is_null() {
            accumulated = logprobs.clone();
        } else {
            merge_json(&mut accumulated, logprobs.clone());
        }
    }
    accumulated
}

fn recorded_finish_reason(scenario: &str, transport: Transport) -> Value {
    match transport {
        Transport::Blocking => recorded_response(scenario)["choices"][0]["finish_reason"].clone(),
        Transport::Streaming => recorded_stream_chunks(scenario)
            .into_iter()
            .filter_map(|chunk| {
                chunk["choices"]
                    .as_array()?
                    .iter()
                    .find(|choice| choice["index"].as_u64() == Some(0))?
                    .get("finish_reason")
                    .filter(|reason| !reason.is_null())
                    .cloned()
            })
            .next_back()
            .unwrap_or(Value::Null),
    }
}

fn recorded_request(scenario: &str) -> Value {
    crate::cassettes::recorded_json_request("openai", scenario)
}

fn recorded_response(scenario: &str) -> Value {
    crate::cassettes::recorded_json_response("openai", scenario)
}

fn recorded_stream_chunks(scenario: &str) -> Vec<Value> {
    crate::cassettes::recorded_sse_json_frames("openai", scenario)
}

fn assert_cell(scenario: &str, cell: Cell, observed: SharedObservation) {
    let request = recorded_request(scenario);
    assert_eq!(request["logprobs"], true, "{scenario}: request premise");
    match cell.top {
        Top::Absent => assert!(
            request.get("top_logprobs").is_none(),
            "{scenario}: top_logprobs should be absent"
        ),
        Top::Zero => assert_eq!(request["top_logprobs"], 0, "{scenario}"),
        Top::Two => assert_eq!(request["top_logprobs"], 2, "{scenario}"),
    }
    assert_eq!(request["model"], model_name(cell.model), "{scenario}");
    match cell.transport {
        Transport::Blocking => assert!(request.get("stream").is_none(), "{scenario}"),
        Transport::Streaming => assert_eq!(request["stream"], true, "{scenario}"),
    }

    let expected_logprobs = match cell.transport {
        Transport::Blocking => recorded_response(scenario)["choices"][0]["logprobs"].clone(),
        Transport::Streaming => recorded_stream_logprobs(scenario),
    };
    assert!(
        expected_logprobs.is_object(),
        "{scenario}: recorded response must exercise logprobs"
    );
    let token_count = expected_logprobs
        .as_object()
        .expect("checked object")
        .values()
        .filter_map(Value::as_array)
        .map(Vec::len)
        .sum::<usize>();
    assert!(token_count > 0, "{scenario}: premise must contain tokens");

    let expected_finish = match cell.termination {
        Termination::Stop => json!("stop"),
        Termination::Length => json!("length"),
    };
    assert_eq!(
        recorded_finish_reason(scenario, cell.transport),
        expected_finish,
        "{scenario}: recorded termination premise"
    );

    let observation = observed
        .lock()
        .expect("observation mutex poisoned")
        .take()
        .expect("test body should save an observation");
    assert_eq!(
        observation.logprobs, expected_logprobs,
        "{scenario}: native response must retain every recorded probability"
    );
    assert_eq!(
        observation.finish_reason, expected_finish,
        "{scenario}: terminal reason"
    );
}

fn cell(transport: Transport, model: ModelVariant, termination: Termination, top: Top) -> Cell {
    Cell {
        transport,
        model,
        termination,
        top,
    }
}

// Blocking controls: 2 models × 2 termination classes × 3 top-N
// configurations. The matching streaming half below traverses the fixed path.

#[tokio::test]
async fn blocking_gpt_4o_mini_stop_top_absent() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/blocking_gpt_4o_mini_stop_top_absent";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt4oMini,
        Termination::Stop,
        Top::Absent,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/blocking_gpt_4o_mini_stop_top_absent",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4o_mini_stop_top_zero() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/blocking_gpt_4o_mini_stop_top_zero";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt4oMini,
        Termination::Stop,
        Top::Zero,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/blocking_gpt_4o_mini_stop_top_zero",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4o_mini_stop_top_two() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/blocking_gpt_4o_mini_stop_top_two";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt4oMini,
        Termination::Stop,
        Top::Two,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/blocking_gpt_4o_mini_stop_top_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4o_mini_length_top_absent() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/blocking_gpt_4o_mini_length_top_absent";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt4oMini,
        Termination::Length,
        Top::Absent,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/blocking_gpt_4o_mini_length_top_absent",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4o_mini_length_top_zero() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/blocking_gpt_4o_mini_length_top_zero";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt4oMini,
        Termination::Length,
        Top::Zero,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/blocking_gpt_4o_mini_length_top_zero",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4o_mini_length_top_two() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/blocking_gpt_4o_mini_length_top_two";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt4oMini,
        Termination::Length,
        Top::Two,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/blocking_gpt_4o_mini_length_top_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4_1_mini_stop_top_absent() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/blocking_gpt_4_1_mini_stop_top_absent";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt41Mini,
        Termination::Stop,
        Top::Absent,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/blocking_gpt_4_1_mini_stop_top_absent",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4_1_mini_stop_top_zero() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/blocking_gpt_4_1_mini_stop_top_zero";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt41Mini,
        Termination::Stop,
        Top::Zero,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/blocking_gpt_4_1_mini_stop_top_zero",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4_1_mini_stop_top_two() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/blocking_gpt_4_1_mini_stop_top_two";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt41Mini,
        Termination::Stop,
        Top::Two,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/blocking_gpt_4_1_mini_stop_top_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4_1_mini_length_top_absent() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/blocking_gpt_4_1_mini_length_top_absent";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt41Mini,
        Termination::Length,
        Top::Absent,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/blocking_gpt_4_1_mini_length_top_absent",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4_1_mini_length_top_zero() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/blocking_gpt_4_1_mini_length_top_zero";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt41Mini,
        Termination::Length,
        Top::Zero,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/blocking_gpt_4_1_mini_length_top_zero",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4_1_mini_length_top_two() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/blocking_gpt_4_1_mini_length_top_two";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt41Mini,
        Termination::Length,
        Top::Two,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/blocking_gpt_4_1_mini_length_top_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4o_mini_stop_top_absent() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/streaming_gpt_4o_mini_stop_top_absent";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt4oMini,
        Termination::Stop,
        Top::Absent,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/streaming_gpt_4o_mini_stop_top_absent",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4o_mini_stop_top_zero() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/streaming_gpt_4o_mini_stop_top_zero";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt4oMini,
        Termination::Stop,
        Top::Zero,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/streaming_gpt_4o_mini_stop_top_zero",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4o_mini_stop_top_two() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/streaming_gpt_4o_mini_stop_top_two";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt4oMini,
        Termination::Stop,
        Top::Two,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/streaming_gpt_4o_mini_stop_top_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4o_mini_length_top_absent() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/streaming_gpt_4o_mini_length_top_absent";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt4oMini,
        Termination::Length,
        Top::Absent,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/streaming_gpt_4o_mini_length_top_absent",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4o_mini_length_top_zero() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/streaming_gpt_4o_mini_length_top_zero";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt4oMini,
        Termination::Length,
        Top::Zero,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/streaming_gpt_4o_mini_length_top_zero",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4o_mini_length_top_two() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/streaming_gpt_4o_mini_length_top_two";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt4oMini,
        Termination::Length,
        Top::Two,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/streaming_gpt_4o_mini_length_top_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4_1_mini_stop_top_absent() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/streaming_gpt_4_1_mini_stop_top_absent";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt41Mini,
        Termination::Stop,
        Top::Absent,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/streaming_gpt_4_1_mini_stop_top_absent",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4_1_mini_stop_top_zero() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/streaming_gpt_4_1_mini_stop_top_zero";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt41Mini,
        Termination::Stop,
        Top::Zero,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/streaming_gpt_4_1_mini_stop_top_zero",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4_1_mini_stop_top_two() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/streaming_gpt_4_1_mini_stop_top_two";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt41Mini,
        Termination::Stop,
        Top::Two,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/streaming_gpt_4_1_mini_stop_top_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4_1_mini_length_top_absent() -> Result<()> {
    const SCENARIO: &str =
        "chat_streaming_logprobs_matrix/streaming_gpt_4_1_mini_length_top_absent";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt41Mini,
        Termination::Length,
        Top::Absent,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/streaming_gpt_4_1_mini_length_top_absent",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4_1_mini_length_top_zero() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/streaming_gpt_4_1_mini_length_top_zero";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt41Mini,
        Termination::Length,
        Top::Zero,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/streaming_gpt_4_1_mini_length_top_zero",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4_1_mini_length_top_two() -> Result<()> {
    const SCENARIO: &str = "chat_streaming_logprobs_matrix/streaming_gpt_4_1_mini_length_top_two";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt41Mini,
        Termination::Length,
        Top::Two,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_chat_stream_logprobs_cassette_result(
        "chat_streaming_logprobs_matrix/streaming_gpt_4_1_mini_length_top_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}
