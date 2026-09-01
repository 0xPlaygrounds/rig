//! Live Chat Completions tool-call lifecycle and argument-integrity matrix.
//!
//! The complete recorded space is 2 transports × 2 models × 3 call shapes ×
//! 2 public surfaces = 24 cells. The shapes cover a deliberate zero-argument
//! call, a nested object containing an array and Unicode, and two parallel
//! calls. Model cells deserialize the provider-native response and normalize
//! it; agent cells prove exact-once invocation. Streaming cells additionally
//! reassemble every id, name, and argument fragment from their fixtures.
//!
//! Auto/none/specific tool-choice controls are assigned to the separate
//! request-shape matrix; this matrix fixes choice to `required` so call shape
//! rather than model discretion is the independent variable. Sequential
//! call/result/follow-up history is assigned to the content/history matrix.
//!
//! | dimension | values |
//! |---|---|
//! | transport | blocking, streaming |
//! | model | `openai/gpt-4o-mini`, `openai/gpt-4.1-mini` pinned to OpenAI |
//! | call shape | zero arguments, nested Unicode/array object, parallel pair |
//! | surface | raw/normalized model, one-turn agent |
//!
//! Coverage ledger: the pre-pruning Cartesian product is 24 and every cell is
//! recorded; none is unit-only. Each explicit test maps to
//! `tests/cassettes/openrouter/tool_lifecycle_matrix/<test-name>.yaml`.
//! Both cheap mini routes are pinned to OpenAI with fallbacks disabled, making
//! the gateway wire stable across model families. Assertions span request
//! schemas, native blocking/streaming assembly, normalized ids, names,
//! arguments and order, and agent exact-once dispatch. Synthetic malformed
//! call objects remain shared unit tests because they cannot be requested live.
//!
//! | recorded cells | exact fixture set |
//! |---|---|
//! | all 24 | `tests/cassettes/openrouter/tool_lifecycle_matrix/{blocking,streaming}_{gpt4o,gpt41}_{zero,nested,parallel}_{model,agent}.yaml` |

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use futures::StreamExt as _;
use rig::completion::{
    AssistantContent, CompletionModel, FinishReason, NormalizeCompletionResponse,
};
use rig::prelude::*;
use rig::providers::openrouter;
use rig::streaming::{StreamedAssistantContent, StreamingCompletionResponse};
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::super::support::with_openrouter_tool_lifecycle_cassette_result;

const PREAMBLE: &str = "Follow the user's tool-call instruction exactly. Do not answer in prose.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Transport {
    Blocking,
    Streaming,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Model {
    Gpt4oMini,
    Gpt41Mini,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Zero,
    Nested,
    Parallel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Surface {
    Model,
    Agent,
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    transport: Transport,
    model: Model,
    shape: Shape,
    surface: Surface,
}

#[derive(Debug, Default)]
struct Observation {
    finish_reason: Option<FinishReason>,
    names: Vec<String>,
    ids: Vec<String>,
    arguments: Vec<Value>,
    errors: Vec<String>,
    invocations: Vec<String>,
}

type SharedObservation = Arc<Mutex<Option<Observation>>>;

fn cell(transport: Transport, model: Model, shape: Shape, surface: Surface) -> Cell {
    Cell {
        transport,
        model,
        shape,
        surface,
    }
}

fn model_name(model: Model) -> &'static str {
    match model {
        Model::Gpt4oMini => "openai/gpt-4o-mini",
        Model::Gpt41Mini => "openai/gpt-4.1-mini",
    }
}

fn prompt(shape: Shape) -> &'static str {
    match shape {
        Shape::Zero => "Call ping exactly once with no arguments.",
        Shape::Nested => {
            "Call record_payload exactly once with label café 東京, values [3, 5, 8], and meta.active true."
        }
        Shape::Parallel => {
            "Call alpha with value red and beta with value blue in the same turn, in that order."
        }
    }
}

fn expected_names(shape: Shape) -> &'static [&'static str] {
    match shape {
        Shape::Zero => &["ping"],
        Shape::Nested => &["record_payload"],
        Shape::Parallel => &["alpha", "beta"],
    }
}

fn tool_definition(name: &str) -> rig::completion::ToolDefinition {
    let parameters = match name {
        "ping" => json!({ "type": "object", "properties": {}, "additionalProperties": false }),
        "record_payload" => json!({
            "type": "object",
            "properties": {
                "label": { "type": "string" },
                "values": { "type": "array", "items": { "type": "integer" } },
                "meta": { "type": "object", "properties": { "active": { "type": "boolean" } }, "required": ["active"] }
            },
            "required": ["label", "values", "meta"]
        }),
        "alpha" | "beta" => {
            json!({ "type": "object", "properties": { "value": { "type": "string" } }, "required": ["value"] })
        }
        other => panic!("unknown matrix tool {other}"),
    };
    rig::completion::ToolDefinition {
        name: name.to_owned(),
        description: format!("Matrix tool {name}"),
        parameters,
    }
}

fn request(model: &openrouter::CompletionModel, cell: Cell) -> rig::completion::CompletionRequest {
    let mut builder = model
        .completion_request(prompt(cell.shape))
        .preamble(PREAMBLE.to_owned())
        .additional_params(json!({
            "tool_choice": "required",
            "parallel_tool_calls": cell.shape == Shape::Parallel,
            "provider": { "order": ["OpenAI"], "allow_fallbacks": false }
        }))
        .max_tokens(128);
    for name in expected_names(cell.shape) {
        builder = builder.tool(tool_definition(name));
    }
    builder.build()
}

fn normalized_calls(choice: &[AssistantContent]) -> (Vec<String>, Vec<String>, Vec<Value>) {
    let calls = choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect::<Vec<_>>();
    (
        calls
            .iter()
            .map(|call| call.function.name.clone())
            .collect(),
        calls.iter().map(|call| call.id.to_string()).collect(),
        calls
            .iter()
            .map(|call| call.function.arguments.clone())
            .collect(),
    )
}

type InvocationLog = Arc<Mutex<Vec<String>>>;

#[derive(Debug, thiserror::Error)]
#[error("matrix tool failed")]
struct MatrixToolError;

#[derive(Debug, Deserialize, Serialize)]
struct EmptyArgs {}

#[derive(Debug, Deserialize, Serialize)]
struct PayloadArgs {
    label: String,
    values: Vec<i64>,
    meta: PayloadMeta,
}

#[derive(Debug, Deserialize, Serialize)]
struct PayloadMeta {
    active: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct ValueArgs {
    value: String,
}

fn note(log: &InvocationLog, name: &str) {
    log.lock()
        .expect("invocation log poisoned")
        .push(name.to_owned());
}

macro_rules! impl_matrix_tool {
    ($ty:ident, $name:literal, $args:ty) => {
        #[derive(Clone)]
        struct $ty {
            log: InvocationLog,
        }

        impl Tool for $ty {
            const NAME: &'static str = $name;
            type Error = MatrixToolError;
            type Args = $args;
            type Output = String;

            fn description(&self) -> String {
                format!("Matrix tool {}", Self::NAME)
            }
            fn parameters(&self) -> Value {
                tool_definition(Self::NAME).parameters
            }

            async fn call(
                &self,
                _context: &mut rig::tool::ToolContext,
                _args: Self::Args,
            ) -> std::result::Result<Self::Output, Self::Error> {
                note(&self.log, Self::NAME);
                Ok(Self::NAME.to_owned())
            }
        }
    };
}

impl_matrix_tool!(Ping, "ping", EmptyArgs);
impl_matrix_tool!(RecordPayload, "record_payload", PayloadArgs);
impl_matrix_tool!(Alpha, "alpha", ValueArgs);
impl_matrix_tool!(Beta, "beta", ValueArgs);

async fn run_model(client: openrouter::Client, cell: Cell) -> Observation {
    let model = client.completion_model(model_name(cell.model));
    match cell.transport {
        Transport::Blocking => match model.raw_completion(request(&model, cell)).await {
            Ok(raw) => match raw.normalize("openrouter") {
                Ok(response) => {
                    let (names, ids, arguments) = normalized_calls(&response.choice);
                    Observation {
                        finish_reason: response.finish_reason(),
                        names,
                        ids,
                        arguments,
                        ..Default::default()
                    }
                }
                Err(error) => Observation {
                    errors: vec![error.to_string()],
                    ..Default::default()
                },
            },
            Err(error) => Observation {
                errors: vec![error.to_string()],
                ..Default::default()
            },
        },
        Transport::Streaming => {
            let raw = match model.raw_stream(request(&model, cell)).await {
                Ok(raw) => raw,
                Err(error) => {
                    return Observation {
                        errors: vec![error.to_string()],
                        ..Default::default()
                    };
                }
            };
            let normalized = rig::streaming::normalize_stream(raw, |terminal| {
                Ok::<_, rig::completion::CompletionError>(("openrouter", terminal).into())
            });
            let mut stream = StreamingCompletionResponse::stream("openrouter", normalized);
            let mut observation = Observation::default();
            while let Some(item) = stream.next().await {
                match item {
                    Ok(StreamedAssistantContent::ToolCall { tool_call, .. }) => {
                        observation.names.push(tool_call.function.name);
                        observation.ids.push(tool_call.id.to_string());
                        observation.arguments.push(tool_call.function.arguments);
                    }
                    Ok(StreamedAssistantContent::Final(terminal)) => {
                        observation.finish_reason = terminal.finish_reason;
                    }
                    Ok(_) => {}
                    Err(error) => observation.errors.push(error.to_string()),
                }
            }
            observation
        }
    }
}

async fn run_agent(client: openrouter::Client, cell: Cell) -> Observation {
    let invocations = InvocationLog::default();
    let builder = client
        .agent(model_name(cell.model))
        .preamble(PREAMBLE)
        .additional_params(json!({
            "tool_choice": "required",
            "parallel_tool_calls": cell.shape == Shape::Parallel,
            "provider": { "order": ["OpenAI"], "allow_fallbacks": false }
        }))
        .max_tokens(128)
        .default_max_turns(1);
    let agent = match cell.shape {
        Shape::Zero => builder
            .tool(Ping {
                log: Arc::clone(&invocations),
            })
            .build(),
        Shape::Nested => builder
            .tool(RecordPayload {
                log: Arc::clone(&invocations),
            })
            .build(),
        Shape::Parallel => builder
            .tool(Alpha {
                log: Arc::clone(&invocations),
            })
            .tool(Beta {
                log: Arc::clone(&invocations),
            })
            .build(),
    };
    let mut errors = Vec::new();
    match cell.transport {
        Transport::Blocking => {
            if let Err(error) = rig::completion::Prompt::prompt(&agent, prompt(cell.shape)).await {
                errors.push(error.to_string());
            }
        }
        Transport::Streaming => {
            let mut stream = rig::streaming::StreamingChat::stream_chat(
                &agent,
                prompt(cell.shape),
                Vec::<rig::completion::Message>::new(),
            )
            .max_turns(1)
            .await;
            errors = crate::support::collect_stream_observation(&mut stream)
                .await
                .errors;
        }
    }
    Observation {
        errors,
        invocations: invocations.lock().expect("invocation log poisoned").clone(),
        ..Default::default()
    }
}

async fn run_cell(
    client: openrouter::Client,
    cell: Cell,
    observed: SharedObservation,
) -> Result<()> {
    let observation = match cell.surface {
        Surface::Model => run_model(client, cell).await,
        Surface::Agent => run_agent(client, cell).await,
    };
    *observed.lock().expect("observation mutex poisoned") = Some(observation);
    Ok(())
}

fn recorded_request(scenario: &str) -> Value {
    crate::cassettes::recorded_json_request("openrouter", scenario)
}

fn recorded_response(scenario: &str) -> Value {
    crate::cassettes::recorded_json_response("openrouter", scenario)
}

fn recorded_chunks(scenario: &str) -> Vec<Value> {
    crate::cassettes::recorded_sse_json_frames("openrouter", scenario)
}

fn recorded_provider(scenario: &str, transport: Transport) -> Value {
    match transport {
        Transport::Blocking => recorded_response(scenario)["provider"].clone(),
        Transport::Streaming => recorded_chunks(scenario)
            .into_iter()
            .filter_map(|chunk| chunk.get("provider").cloned())
            .next_back()
            .unwrap_or(Value::Null),
    }
}

#[derive(Debug)]
struct RecordedCall {
    id: String,
    name: String,
    arguments: String,
}

fn recorded_finish_and_calls(scenario: &str, transport: Transport) -> (String, Vec<RecordedCall>) {
    match transport {
        Transport::Blocking => {
            let response = recorded_response(scenario);
            let finish = response["choices"][0]["finish_reason"]
                .as_str()
                .unwrap_or_default()
                .to_owned();
            let calls = response["choices"][0]["message"]["tool_calls"]
                .as_array()
                .map(|calls| {
                    calls
                        .iter()
                        .map(|call| RecordedCall {
                            id: call["id"].as_str().unwrap_or_default().to_owned(),
                            name: call["function"]["name"]
                                .as_str()
                                .unwrap_or_default()
                                .to_owned(),
                            arguments: call["function"]["arguments"]
                                .as_str()
                                .unwrap_or_default()
                                .to_owned(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            (finish, calls)
        }
        Transport::Streaming => {
            let mut finish = String::new();
            let mut calls = Vec::<RecordedCall>::new();
            for chunk in recorded_chunks(scenario) {
                for choice in chunk["choices"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|choice| choice["index"].as_u64() == Some(0))
                {
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
                            calls.push(RecordedCall {
                                id: String::new(),
                                name: String::new(),
                                arguments: String::new(),
                            });
                        }
                        if let Some(id) = call["id"].as_str().filter(|id| !id.is_empty()) {
                            calls[index].id = id.to_owned();
                        }
                        if let Some(name) = call["function"]["name"]
                            .as_str()
                            .filter(|name| !name.is_empty())
                        {
                            calls[index].name = name.to_owned();
                        }
                        if let Some(fragment) = call["function"]["arguments"].as_str() {
                            calls[index].arguments.push_str(fragment);
                        }
                    }
                }
            }
            (finish, calls)
        }
    }
}

fn expected_arguments(shape: Shape) -> Vec<Value> {
    match shape {
        Shape::Zero => vec![json!({})],
        Shape::Nested => {
            vec![json!({ "label": "café 東京", "values": [3, 5, 8], "meta": { "active": true } })]
        }
        Shape::Parallel => vec![json!({ "value": "red" }), json!({ "value": "blue" })],
    }
}

fn assert_nonempty_distinct_ids(scenario: &str, ids: &[String]) {
    assert!(
        ids.iter().all(|id| !id.is_empty()),
        "{scenario}: every call has an id: {ids:?}"
    );
    assert_eq!(
        ids.iter().collect::<std::collections::BTreeSet<_>>().len(),
        ids.len(),
        "{scenario}: call ids are distinct"
    );
}

fn assert_cell(scenario: &str, cell: Cell, observed: SharedObservation) {
    let request = recorded_request(scenario);
    assert_eq!(
        request["model"],
        model_name(cell.model),
        "{scenario}: model"
    );
    assert_eq!(request["max_tokens"], 128, "{scenario}: cap");
    assert!(
        request.get("max_completion_tokens").is_none(),
        "{scenario}: these legacy Chat Completions models use max_tokens"
    );
    assert_eq!(
        request["tool_choice"], "required",
        "{scenario}: forced tool"
    );
    assert_eq!(
        request["provider"]["order"],
        json!(["OpenAI"]),
        "{scenario}: upstream route"
    );
    assert_eq!(
        request["provider"]["allow_fallbacks"], false,
        "{scenario}: fallback policy"
    );
    assert_eq!(
        recorded_provider(scenario, cell.transport),
        "OpenAI",
        "{scenario}: recorded upstream"
    );
    assert_eq!(
        request["tools"].as_array().map(Vec::len),
        Some(expected_names(cell.shape).len()),
        "{scenario}: tool schema"
    );
    assert_eq!(
        request["parallel_tool_calls"],
        cell.shape == Shape::Parallel,
        "{scenario}: parallel policy"
    );
    assert_eq!(
        request["stream"].as_bool().unwrap_or(false),
        cell.transport == Transport::Streaming,
        "{scenario}: transport"
    );

    let (finish, wire_calls) = recorded_finish_and_calls(scenario, cell.transport);
    assert_eq!(finish, "tool_calls", "{scenario}: tool terminal");
    assert_eq!(
        wire_calls
            .iter()
            .map(|call| call.name.as_str())
            .collect::<Vec<_>>(),
        expected_names(cell.shape),
        "{scenario}: wire call order"
    );
    assert_nonempty_distinct_ids(
        scenario,
        &wire_calls
            .iter()
            .map(|call| call.id.clone())
            .collect::<Vec<_>>(),
    );
    let wire_arguments = wire_calls
        .iter()
        .map(|call| {
            serde_json::from_str::<Value>(&call.arguments)
                .with_context(|| format!("{scenario}: parse {} arguments", call.name))
                .expect("recorded tool JSON")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        wire_arguments,
        expected_arguments(cell.shape),
        "{scenario}: exact wire arguments"
    );

    let observation = observed
        .lock()
        .expect("observation mutex poisoned")
        .take()
        .expect("cell observation");
    match cell.surface {
        Surface::Model => {
            assert!(
                observation.errors.is_empty(),
                "{scenario}: {:?}",
                observation.errors
            );
            assert_eq!(
                observation.finish_reason,
                Some(FinishReason::ToolCalls),
                "{scenario}"
            );
            assert_eq!(
                observation
                    .names
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                expected_names(cell.shape),
                "{scenario}: normalized call order"
            );
            assert_nonempty_distinct_ids(scenario, &observation.ids);
            assert_eq!(
                observation.arguments,
                expected_arguments(cell.shape),
                "{scenario}: normalized arguments"
            );
        }
        Surface::Agent => {
            assert_eq!(
                observation
                    .invocations
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                expected_names(cell.shape),
                "{scenario}: exact-once invocation order"
            );
            assert!(
                observation
                    .errors
                    .iter()
                    .all(|error| !error.contains("ProviderResponseError")),
                "{scenario}: lifecycle reaches agent loop: {:?}",
                observation.errors
            );
        }
    }
}

async fn execute(scenario: &'static str, cell: Cell, observed: SharedObservation) {
    assert_cell(scenario, cell, observed);
}

// The literal wrapper calls below are intentionally explicit: cassette safety
// parses source rather than macro expansion.

#[tokio::test]
async fn blocking_gpt4o_zero_model() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/blocking_gpt4o_zero_model";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Shape::Zero,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/blocking_gpt4o_zero_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt4o_zero_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/blocking_gpt4o_zero_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Shape::Zero,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/blocking_gpt4o_zero_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt4o_nested_model() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/blocking_gpt4o_nested_model";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Shape::Nested,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/blocking_gpt4o_nested_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt4o_nested_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/blocking_gpt4o_nested_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Shape::Nested,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/blocking_gpt4o_nested_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt4o_parallel_model() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/blocking_gpt4o_parallel_model";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Shape::Parallel,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/blocking_gpt4o_parallel_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt4o_parallel_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/blocking_gpt4o_parallel_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Shape::Parallel,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/blocking_gpt4o_parallel_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt41_zero_model() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/blocking_gpt41_zero_model";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Shape::Zero,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/blocking_gpt41_zero_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt41_zero_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/blocking_gpt41_zero_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Shape::Zero,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/blocking_gpt41_zero_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt41_nested_model() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/blocking_gpt41_nested_model";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Shape::Nested,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/blocking_gpt41_nested_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt41_nested_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/blocking_gpt41_nested_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Shape::Nested,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/blocking_gpt41_nested_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt41_parallel_model() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/blocking_gpt41_parallel_model";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Shape::Parallel,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/blocking_gpt41_parallel_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt41_parallel_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/blocking_gpt41_parallel_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Shape::Parallel,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/blocking_gpt41_parallel_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt4o_zero_model() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/streaming_gpt4o_zero_model";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Shape::Zero,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/streaming_gpt4o_zero_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt4o_zero_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/streaming_gpt4o_zero_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Shape::Zero,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/streaming_gpt4o_zero_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt4o_nested_model() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/streaming_gpt4o_nested_model";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Shape::Nested,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/streaming_gpt4o_nested_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt4o_nested_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/streaming_gpt4o_nested_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Shape::Nested,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/streaming_gpt4o_nested_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt4o_parallel_model() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/streaming_gpt4o_parallel_model";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Shape::Parallel,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/streaming_gpt4o_parallel_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt4o_parallel_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/streaming_gpt4o_parallel_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Shape::Parallel,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/streaming_gpt4o_parallel_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt41_zero_model() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/streaming_gpt41_zero_model";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Shape::Zero,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/streaming_gpt41_zero_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt41_zero_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/streaming_gpt41_zero_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Shape::Zero,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/streaming_gpt41_zero_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt41_nested_model() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/streaming_gpt41_nested_model";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Shape::Nested,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/streaming_gpt41_nested_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt41_nested_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/streaming_gpt41_nested_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Shape::Nested,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/streaming_gpt41_nested_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt41_parallel_model() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/streaming_gpt41_parallel_model";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Shape::Parallel,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/streaming_gpt41_parallel_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt41_parallel_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/streaming_gpt41_parallel_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Shape::Parallel,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/streaming_gpt41_parallel_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
