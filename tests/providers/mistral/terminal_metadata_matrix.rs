//! Mistral terminal metadata and primary-choice matrix.
//!
//! Mistral's live API rejects `logprobs` for both accessible model families
//! (`3051: Logprobs are not enabled for this model`), so that proposed matrix
//! cannot provide successful provider evidence. This replacement exercises
//! the adjacent shared paths changed by #2359: primary-choice selection,
//! length/tool-call terminal mapping, response identity, and usage retention.
//!
//! The complete input space exercised here is a 2 × 2 × 2 × 3 cross-product:
//!
//! | dimension | values |
//! |---|---|
//! | transport | blocking control, raw streaming regression |
//! | model | `mistral-small-latest`, `ministral-3b-latest` |
//! | output budget | roomy, one-token tiny |
//! | response shape | one text candidate, two text candidates, forced tool |
//!
//! That is 24 recorded cells. Each cell proves its request and terminal premise
//! from the fixture, then compares the native raw response/terminal metadata
//! with those exact bytes. The `n = 2` cells additionally prove that the shared
//! streaming adapter selects candidate zero without concatenating candidate
//! one into it.
//!
//! Coverage ledger: the proposed 24-cell logprobs product was pruned in full
//! after both models rejected live blocking and streaming probes with code
//! 3051; it was replaced, not counted, by this unpruned 24-cell terminal
//! product. Each explicit test maps to
//! `tests/cassettes/mistral/terminal_metadata_matrix/<test-name>.yaml`; none of
//! these cells is unit-only. `mistral-small-latest` and
//! `ministral-3b-latest` are the two inexpensive, currently served chat/tool
//! families. Assertions cover requests, native terminals, ids, models, finish
//! reasons, usage including `service_tier`, primary-choice behavior, and every
//! otherwise-unmodeled top-level SSE field (including the observed `p` key).
//!
//! | recorded cells | exact fixture set |
//! |---|---|
//! | all 24 replacement cells | `tests/cassettes/mistral/terminal_metadata_matrix/{blocking,streaming}_{mistral_small,ministral_3b}_{roomy,tiny}_{plain_one,plain_two,tool}.yaml` |

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use futures::StreamExt as _;
use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::mistral;
use rig::streaming::RawStreamingChoice;
use serde_json::{Value, json};

use super::support::with_mistral_terminal_metadata_cassette_result;

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
        Shape::Tool => json!({ "tool_choice": "any" }),
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
        (_, Limit::Tiny) => 1,
        (Shape::Tool, Limit::Roomy) => 64,
        (_, Limit::Roomy) => 8,
    }
}

fn model_name(model: ModelVariant) -> &'static str {
    match model {
        ModelVariant::MistralSmall => "mistral-small-latest",
        ModelVariant::Ministral3b => "ministral-3b-latest",
    }
}

async fn run_cell(client: mistral::Client, cell: Cell, observed: SharedObservation) -> Result<()> {
    let model = client.completion_model(model_name(cell.model));
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
    crate::cassettes::recorded_json_request("mistral", scenario)
}

fn recorded_response(scenario: &str) -> Value {
    crate::cassettes::recorded_json_response("mistral", scenario)
}

fn recorded_stream_chunks(scenario: &str) -> Vec<Value> {
    crate::cassettes::recorded_sse_json_frames("mistral", scenario)
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

fn assert_usage(scenario: &str, actual: &Value, wire: &Value) {
    for field in ["completion_tokens", "prompt_tokens", "total_tokens"] {
        assert_eq!(actual[field], wire[field], "{scenario}: usage.{field}");
    }
    assert_eq!(
        actual["prompt_tokens_details"]["cached_tokens"],
        wire["prompt_tokens_details"]["cached_tokens"],
        "{scenario}: cached prompt tokens"
    );
    assert_eq!(
        actual["service_tier"], wire["service_tier"],
        "{scenario}: service tier"
    );
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
            assert_eq!(request["tool_choice"], "any", "{scenario}");
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
async fn blocking_mistral_small_roomy_plain_one() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/blocking_mistral_small_roomy_plain_one";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::MistralSmall,
        Limit::Roomy,
        Shape::PlainOne,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/blocking_mistral_small_roomy_plain_one",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_mistral_small_roomy_plain_two() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/blocking_mistral_small_roomy_plain_two";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::MistralSmall,
        Limit::Roomy,
        Shape::PlainTwo,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/blocking_mistral_small_roomy_plain_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_mistral_small_roomy_tool() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/blocking_mistral_small_roomy_tool";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::MistralSmall,
        Limit::Roomy,
        Shape::Tool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/blocking_mistral_small_roomy_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_mistral_small_tiny_plain_one() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/blocking_mistral_small_tiny_plain_one";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::MistralSmall,
        Limit::Tiny,
        Shape::PlainOne,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/blocking_mistral_small_tiny_plain_one",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_mistral_small_tiny_plain_two() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/blocking_mistral_small_tiny_plain_two";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::MistralSmall,
        Limit::Tiny,
        Shape::PlainTwo,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/blocking_mistral_small_tiny_plain_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_mistral_small_tiny_tool() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/blocking_mistral_small_tiny_tool";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::MistralSmall,
        Limit::Tiny,
        Shape::Tool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/blocking_mistral_small_tiny_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_roomy_plain_one() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/blocking_ministral_3b_roomy_plain_one";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Ministral3b,
        Limit::Roomy,
        Shape::PlainOne,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/blocking_ministral_3b_roomy_plain_one",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_roomy_plain_two() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/blocking_ministral_3b_roomy_plain_two";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Ministral3b,
        Limit::Roomy,
        Shape::PlainTwo,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/blocking_ministral_3b_roomy_plain_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_roomy_tool() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/blocking_ministral_3b_roomy_tool";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Ministral3b,
        Limit::Roomy,
        Shape::Tool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/blocking_ministral_3b_roomy_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_tiny_plain_one() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/blocking_ministral_3b_tiny_plain_one";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Ministral3b,
        Limit::Tiny,
        Shape::PlainOne,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/blocking_ministral_3b_tiny_plain_one",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_tiny_plain_two() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/blocking_ministral_3b_tiny_plain_two";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Ministral3b,
        Limit::Tiny,
        Shape::PlainTwo,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/blocking_ministral_3b_tiny_plain_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_tiny_tool() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/blocking_ministral_3b_tiny_tool";
    let cell = cell(
        Transport::Blocking,
        ModelVariant::Ministral3b,
        Limit::Tiny,
        Shape::Tool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/blocking_ministral_3b_tiny_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_roomy_plain_one() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/streaming_mistral_small_roomy_plain_one";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::MistralSmall,
        Limit::Roomy,
        Shape::PlainOne,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/streaming_mistral_small_roomy_plain_one",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_roomy_plain_two() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/streaming_mistral_small_roomy_plain_two";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::MistralSmall,
        Limit::Roomy,
        Shape::PlainTwo,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/streaming_mistral_small_roomy_plain_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_roomy_tool() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/streaming_mistral_small_roomy_tool";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::MistralSmall,
        Limit::Roomy,
        Shape::Tool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/streaming_mistral_small_roomy_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_tiny_plain_one() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/streaming_mistral_small_tiny_plain_one";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::MistralSmall,
        Limit::Tiny,
        Shape::PlainOne,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/streaming_mistral_small_tiny_plain_one",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_tiny_plain_two() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/streaming_mistral_small_tiny_plain_two";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::MistralSmall,
        Limit::Tiny,
        Shape::PlainTwo,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/streaming_mistral_small_tiny_plain_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_tiny_tool() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/streaming_mistral_small_tiny_tool";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::MistralSmall,
        Limit::Tiny,
        Shape::Tool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/streaming_mistral_small_tiny_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_roomy_plain_one() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/streaming_ministral_3b_roomy_plain_one";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Ministral3b,
        Limit::Roomy,
        Shape::PlainOne,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/streaming_ministral_3b_roomy_plain_one",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_roomy_plain_two() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/streaming_ministral_3b_roomy_plain_two";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Ministral3b,
        Limit::Roomy,
        Shape::PlainTwo,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/streaming_ministral_3b_roomy_plain_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_roomy_tool() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/streaming_ministral_3b_roomy_tool";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Ministral3b,
        Limit::Roomy,
        Shape::Tool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/streaming_ministral_3b_roomy_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_tiny_plain_one() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/streaming_ministral_3b_tiny_plain_one";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Ministral3b,
        Limit::Tiny,
        Shape::PlainOne,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/streaming_ministral_3b_tiny_plain_one",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_tiny_plain_two() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/streaming_ministral_3b_tiny_plain_two";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Ministral3b,
        Limit::Tiny,
        Shape::PlainTwo,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/streaming_ministral_3b_tiny_plain_two",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_tiny_tool() -> Result<()> {
    const SCENARIO: &str = "terminal_metadata_matrix/streaming_ministral_3b_tiny_tool";
    let cell = cell(
        Transport::Streaming,
        ModelVariant::Ministral3b,
        Limit::Tiny,
        Shape::Tool,
    );
    let observed = SharedObservation::default();
    let capture = Arc::clone(&observed);
    with_mistral_terminal_metadata_cassette_result(
        "terminal_metadata_matrix/streaming_ministral_3b_tiny_tool",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    assert_cell(SCENARIO, cell, observed);
    Ok(())
}
