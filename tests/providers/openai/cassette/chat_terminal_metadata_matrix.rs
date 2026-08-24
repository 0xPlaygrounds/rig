//! OpenAI Chat Completions terminal metadata and primary-choice matrix.
//!
//! Live recordings showed that the provider-native blocking response erased
//! `service_tier`, while the shared raw-stream terminal erased every unknown
//! top-level chunk field, including `service_tier` and `system_fingerprint`.
//! This matrix locks down that metadata together with primary-choice
//! selection, length/tool-call terminal mapping, response identity, and usage.
//!
//! The complete input space exercised here is a 2 × 2 × 2 × 3 cross-product:
//!
//! | dimension | values |
//! |---|---|
//! | transport | blocking control, raw streaming regression |
//! | model | `gpt-4o-mini`, `gpt-4.1-mini` |
//! | output budget | roomy, constrained (`1` text token / `16` tool tokens) |
//! | response shape | one text candidate, two text candidates, forced tool |
//!
//! That is 24 recorded cells. Each cell proves its request and terminal premise
//! from the fixture, then compares the native raw response/terminal metadata
//! with those exact bytes. The `n = 2` cells additionally prove that the shared
//! streaming adapter selects candidate zero without concatenating candidate
//! one into it.
//!
//! Coverage ledger: the pre-pruning Cartesian product is 24 and all 24 cells
//! are recorded; none is unit-only. Each explicit test maps to
//! `tests/cassettes/openai/chat_terminal_metadata_matrix/<test-name>.yaml`.
//! The inexpensive mini models provide stable text, multi-choice, and tool
//! controls across two model families. Every cell asserts the request and
//! provider-native terminal; together they cover ids, model, finish reason,
//! usage, primary-choice routing, `service_tier`, `system_fingerprint`, and
//! otherwise-unmodeled top-level streaming metadata.
//!
//! | recorded cells | exact fixture set |
//! |---|---|
//! | all 24 | `tests/cassettes/openai/chat_terminal_metadata_matrix/{blocking,streaming}_{gpt_4o_mini,gpt_4_1_mini}_{roomy,tiny}_{plain_one,plain_two,tool}.yaml` |

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use futures::StreamExt as _;
use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::openai;
use rig::streaming::RawStreamingChoice;
use serde_json::{Value, json};

use super::super::support::with_openai_terminal_metadata_cassette_result;

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
enum Limit {
    Roomy,
    Tiny,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    PlainOne,
    PlainTwo,
    Tool,
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    transport: Transport,
    model: ModelVariant,
    limit: Limit,
    shape: Shape,
}

#[derive(Debug)]
struct Observation {
    raw: Value,
}

type SharedObservation = Arc<Mutex<Option<Observation>>>;

fn params(cell: Cell) -> Value {
    match cell.shape {
        Shape::PlainOne => json!({}),
        Shape::PlainTwo => json!({ "n": 2 }),
        Shape::Tool => json!({ "tool_choice": "required" }),
    }
}

fn prompt(cell: Cell) -> &'static str {
    match (cell.shape, cell.limit) {
        (Shape::Tool, _) => {
            "Call the add tool exactly once with x=17 and y=25. Do not answer in prose."
        }
        (_, Limit::Roomy) => "Reply with exactly: cobalt",
        (_, Limit::Tiny) => "Write the word cobalt exactly 100 times, separated by spaces.",
    }
}

fn max_tokens(cell: Cell) -> u64 {
    match (cell.shape, cell.limit) {
        (Shape::Tool, Limit::Tiny) => 16,
        (_, Limit::Tiny) => 1,
        (Shape::Tool, Limit::Roomy) => 64,
        (_, Limit::Roomy) => 8,
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
    let mut builder = model
        .completion_request(prompt(cell))
        .additional_params(params(cell))
        .max_tokens(max_tokens(cell));
    if cell.shape == Shape::Tool {
        builder = builder.tool(rig::tool::tool_definition(&crate::support::Adder));
    }
    let request = builder.build();

    let raw = match cell.transport {
        Transport::Blocking => {
            let response = model.raw_completion(request).await?;
            serde_json::to_value(response)?
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
            serde_json::to_value(terminal)?
        }
    };

    *observed.lock().expect("observation mutex poisoned") = Some(Observation { raw });
    Ok(())
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

fn assert_usage(scenario: &str, actual: &Value, wire: &Value) {
    for field in ["completion_tokens", "prompt_tokens", "total_tokens"] {
        assert_eq!(actual[field], wire[field], "{scenario}: usage.{field}");
    }
    assert_eq!(
        actual["prompt_tokens_details"]["cached_tokens"],
        wire["prompt_tokens_details"]["cached_tokens"],
        "{scenario}: cached prompt tokens"
    );
}

fn last_chunk_field(chunks: &[Value], field: &str) -> Value {
    chunks
        .iter()
        .filter_map(|chunk| chunk.get(field))
        .next_back()
        .cloned()
        .unwrap_or(Value::Null)
}

fn recorded_additional_params(chunks: &[Value]) -> Value {
    let mut accumulated: Option<rig::message::AdditionalParams> = None;
    for chunk in chunks {
        let mut extras = chunk
            .as_object()
            .cloned()
            .expect("recorded SSE frame should be an object");
        for modeled in ["id", "model", "choices", "usage"] {
            extras.remove(modeled);
        }
        if let Some(incoming) = rig::message::AdditionalParams::new(extras) {
            match accumulated.as_mut() {
                Some(current) => current.merge(incoming),
                None => accumulated = Some(incoming),
            }
        }
    }
    serde_json::to_value(accumulated).expect("recorded metadata should serialize")
}

fn assert_scrubbed_optional_string(scenario: &str, field: &str, actual: &Value, wire: &Value) {
    assert_eq!(
        actual.is_null(),
        wire.is_null(),
        "{scenario}: {field} nullability"
    );
    if !actual.is_null() {
        assert!(
            actual.as_str().is_some_and(|value| !value.is_empty())
                && wire.as_str().is_some_and(|value| !value.is_empty()),
            "{scenario}: {field} should be non-empty on both live and scrubbed sides"
        );
    }
}

fn assert_cell(scenario: &str, cell: Cell, observed: SharedObservation) {
    let request = recorded_request(scenario);
    match cell.shape {
        Shape::PlainOne => {
            assert!(request.get("n").is_none(), "{scenario}: n should be absent");
            assert!(request.get("tools").is_none(), "{scenario}: no tools");
        }
        Shape::PlainTwo => {
            assert_eq!(request["n"], 2, "{scenario}");
            assert!(request.get("tools").is_none(), "{scenario}: no tools");
        }
        Shape::Tool => {
            assert_eq!(request["tool_choice"], "required", "{scenario}");
            assert_eq!(
                request["tools"].as_array().map(Vec::len),
                Some(1),
                "{scenario}"
            );
        }
    }
    assert_eq!(request["model"], model_name(cell.model), "{scenario}");
    assert_eq!(request["max_tokens"], max_tokens(cell), "{scenario}");
    match cell.transport {
        Transport::Blocking => assert!(request.get("stream").is_none(), "{scenario}"),
        Transport::Streaming => assert_eq!(request["stream"], true, "{scenario}"),
    }

    let expected_finish = match (cell.shape, cell.limit) {
        (Shape::Tool, Limit::Roomy) => json!("tool_calls"),
        (_, Limit::Roomy) => json!("stop"),
        (_, Limit::Tiny) => json!("length"),
    };
    assert_eq!(
        recorded_finish_reason(scenario, cell.transport),
        expected_finish,
        "{scenario}: recorded limit premise"
    );

    let observation = observed
        .lock()
        .expect("observation mutex poisoned")
        .take()
        .expect("test body should save an observation");
    match cell.transport {
        Transport::Blocking => {
            let response = recorded_response(scenario);
            let expected_choices = if cell.shape == Shape::PlainTwo { 2 } else { 1 };
            assert_eq!(
                response["choices"].as_array().map(Vec::len),
                Some(expected_choices)
            );
            assert!(
                observation.raw["id"]
                    .as_str()
                    .is_some_and(|id| !id.is_empty())
                    && response["id"].as_str().is_some_and(|id| !id.is_empty()),
                "{scenario}: both the native response and fixture retain an id"
            );
            assert_eq!(
                observation.raw["model"], response["model"],
                "{scenario}: model"
            );
            assert_usage(scenario, &observation.raw["usage"], &response["usage"]);
            assert_eq!(
                observation.raw["service_tier"], response["service_tier"],
                "{scenario}: blocking service tier"
            );
            assert_scrubbed_optional_string(
                scenario,
                "blocking system fingerprint",
                &observation.raw["system_fingerprint"],
                &response["system_fingerprint"],
            );
            assert_eq!(
                observation.raw["choices"][0]["finish_reason"], expected_finish,
                "{scenario}: terminal reason"
            );
        }
        Transport::Streaming => {
            let chunks = recorded_stream_chunks(scenario);
            let indexes = chunks
                .iter()
                .flat_map(|chunk| chunk["choices"].as_array().into_iter().flatten())
                .filter_map(|choice| choice["index"].as_u64())
                .collect::<std::collections::BTreeSet<_>>();
            let expected_indexes = if cell.shape == Shape::PlainTwo {
                std::collections::BTreeSet::from([0, 1])
            } else {
                std::collections::BTreeSet::from([0])
            };
            assert_eq!(indexes, expected_indexes, "{scenario}: candidate premise");

            let response_id = chunks
                .iter()
                .filter_map(|chunk| chunk["id"].as_str())
                .next_back()
                .map(str::to_owned);
            let response_model = chunks
                .iter()
                .filter_map(|chunk| chunk["model"].as_str())
                .next_back()
                .map(str::to_owned);
            let usage = chunks
                .iter()
                .filter_map(|chunk| chunk.get("usage"))
                .rfind(|usage| !usage.is_null())
                .cloned()
                .unwrap_or_else(|| json!({}));
            assert!(
                observation.raw["response_id"]
                    .as_str()
                    .is_some_and(|id| !id.is_empty())
                    && response_id.as_deref().is_some_and(|id| !id.is_empty()),
                "{scenario}: both the native terminal and fixture retain an id"
            );
            assert_eq!(
                observation.raw["model"],
                json!(response_model),
                "{scenario}: model"
            );
            assert_usage(scenario, &observation.raw["usage"], &usage);
            assert_eq!(
                observation.raw["additional_params"],
                recorded_additional_params(&chunks),
                "{scenario}: every unmodeled top-level SSE field"
            );
            assert_eq!(
                observation.raw["additional_params"]["service_tier"],
                last_chunk_field(&chunks, "service_tier"),
                "{scenario}: streaming service tier"
            );
            assert_scrubbed_optional_string(
                scenario,
                "streaming system fingerprint",
                &observation.raw["additional_params"]["system_fingerprint"],
                &last_chunk_field(&chunks, "system_fingerprint"),
            );
            assert_eq!(
                observation.raw["finish_reason"], expected_finish,
                "{scenario}"
            );
        }
    }
}

fn cell(transport: Transport, model: ModelVariant, limit: Limit, shape: Shape) -> Cell {
    Cell {
        transport,
        model,
        limit,
        shape,
    }
}

// Blocking controls: 2 models × 2 output budgets × 3 response shapes. The
// matching streaming half traverses the shared compatible stream adapter.

#[tokio::test]
async fn blocking_gpt_4o_mini_roomy_plain_one() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/blocking_gpt_4o_mini_roomy_plain_one";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt4oMini,
        Limit::Roomy,
        Shape::PlainOne,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/blocking_gpt_4o_mini_roomy_plain_one",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4o_mini_roomy_plain_two() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/blocking_gpt_4o_mini_roomy_plain_two";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt4oMini,
        Limit::Roomy,
        Shape::PlainTwo,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/blocking_gpt_4o_mini_roomy_plain_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4o_mini_roomy_tool() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/blocking_gpt_4o_mini_roomy_tool";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt4oMini,
        Limit::Roomy,
        Shape::Tool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/blocking_gpt_4o_mini_roomy_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4o_mini_tiny_plain_one() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/blocking_gpt_4o_mini_tiny_plain_one";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt4oMini,
        Limit::Tiny,
        Shape::PlainOne,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/blocking_gpt_4o_mini_tiny_plain_one",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4o_mini_tiny_plain_two() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/blocking_gpt_4o_mini_tiny_plain_two";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt4oMini,
        Limit::Tiny,
        Shape::PlainTwo,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/blocking_gpt_4o_mini_tiny_plain_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4o_mini_tiny_tool() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/blocking_gpt_4o_mini_tiny_tool";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt4oMini,
        Limit::Tiny,
        Shape::Tool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/blocking_gpt_4o_mini_tiny_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4_1_mini_roomy_plain_one() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/blocking_gpt_4_1_mini_roomy_plain_one";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt41Mini,
        Limit::Roomy,
        Shape::PlainOne,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/blocking_gpt_4_1_mini_roomy_plain_one",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4_1_mini_roomy_plain_two() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/blocking_gpt_4_1_mini_roomy_plain_two";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt41Mini,
        Limit::Roomy,
        Shape::PlainTwo,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/blocking_gpt_4_1_mini_roomy_plain_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4_1_mini_roomy_tool() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/blocking_gpt_4_1_mini_roomy_tool";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt41Mini,
        Limit::Roomy,
        Shape::Tool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/blocking_gpt_4_1_mini_roomy_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4_1_mini_tiny_plain_one() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/blocking_gpt_4_1_mini_tiny_plain_one";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt41Mini,
        Limit::Tiny,
        Shape::PlainOne,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/blocking_gpt_4_1_mini_tiny_plain_one",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4_1_mini_tiny_plain_two() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/blocking_gpt_4_1_mini_tiny_plain_two";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt41Mini,
        Limit::Tiny,
        Shape::PlainTwo,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/blocking_gpt_4_1_mini_tiny_plain_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_gpt_4_1_mini_tiny_tool() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/blocking_gpt_4_1_mini_tiny_tool";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Gpt41Mini,
        Limit::Tiny,
        Shape::Tool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/blocking_gpt_4_1_mini_tiny_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4o_mini_roomy_plain_one() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/streaming_gpt_4o_mini_roomy_plain_one";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt4oMini,
        Limit::Roomy,
        Shape::PlainOne,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/streaming_gpt_4o_mini_roomy_plain_one",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4o_mini_roomy_plain_two() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/streaming_gpt_4o_mini_roomy_plain_two";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt4oMini,
        Limit::Roomy,
        Shape::PlainTwo,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/streaming_gpt_4o_mini_roomy_plain_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4o_mini_roomy_tool() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/streaming_gpt_4o_mini_roomy_tool";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt4oMini,
        Limit::Roomy,
        Shape::Tool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/streaming_gpt_4o_mini_roomy_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4o_mini_tiny_plain_one() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/streaming_gpt_4o_mini_tiny_plain_one";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt4oMini,
        Limit::Tiny,
        Shape::PlainOne,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/streaming_gpt_4o_mini_tiny_plain_one",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4o_mini_tiny_plain_two() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/streaming_gpt_4o_mini_tiny_plain_two";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt4oMini,
        Limit::Tiny,
        Shape::PlainTwo,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/streaming_gpt_4o_mini_tiny_plain_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4o_mini_tiny_tool() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/streaming_gpt_4o_mini_tiny_tool";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt4oMini,
        Limit::Tiny,
        Shape::Tool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/streaming_gpt_4o_mini_tiny_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4_1_mini_roomy_plain_one() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/streaming_gpt_4_1_mini_roomy_plain_one";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt41Mini,
        Limit::Roomy,
        Shape::PlainOne,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/streaming_gpt_4_1_mini_roomy_plain_one",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4_1_mini_roomy_plain_two() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/streaming_gpt_4_1_mini_roomy_plain_two";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt41Mini,
        Limit::Roomy,
        Shape::PlainTwo,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/streaming_gpt_4_1_mini_roomy_plain_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4_1_mini_roomy_tool() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/streaming_gpt_4_1_mini_roomy_tool";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt41Mini,
        Limit::Roomy,
        Shape::Tool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/streaming_gpt_4_1_mini_roomy_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4_1_mini_tiny_plain_one() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/streaming_gpt_4_1_mini_tiny_plain_one";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt41Mini,
        Limit::Tiny,
        Shape::PlainOne,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/streaming_gpt_4_1_mini_tiny_plain_one",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4_1_mini_tiny_plain_two() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/streaming_gpt_4_1_mini_tiny_plain_two";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt41Mini,
        Limit::Tiny,
        Shape::PlainTwo,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/streaming_gpt_4_1_mini_tiny_plain_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_gpt_4_1_mini_tiny_tool() -> Result<()> {
    const SCENARIO: &str = "chat_terminal_metadata_matrix/streaming_gpt_4_1_mini_tiny_tool";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Gpt41Mini,
        Limit::Tiny,
        Shape::Tool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_openai_terminal_metadata_cassette_result(
        "chat_terminal_metadata_matrix/streaming_gpt_4_1_mini_tiny_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}
