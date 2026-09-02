//! Live Chat Completions matrix for the outer-reason truncation contract.
//!
//! A live budget sweep on 2026-08-16 pinned OpenRouter to the OpenAI upstream
//! with fallbacks disabled. Unlike OpenAI directly, caps 16 and 32 returned a
//! non-empty, unparseable argument string while still claiming
//! `finish_reason: tool_calls`; cap 48 returned the complete object. That wire
//! is not an outer truncation signal, so Rig must reject it loudly instead of
//! applying #2359's narrow `length` tolerance. The two selected model routes
//! are cheap, stable, and expose the same provider-specific mismatch.
//!
//! The complete recorded space is 2 transports × 2 models × 3 budgets × 2
//! public surfaces = 24 cells. Model cells deserialize the provider-native
//! blocking response or drive the normalized stream; agent cells prove that a
//! partial call is never invoked and a complete control is invoked exactly
//! once. Every cell asserts its own request, finish reason, and accumulated
//! argument bytes from the fixture.
//!
//! | dimension | values |
//! |---|---|
//! | transport | blocking, streaming |
//! | model | `openai/gpt-4o-mini`, `openai/gpt-4.1-mini` |
//! | output cap | 16 malformed completed, 32 malformed completed, 48 valid completed |
//! | surface | raw/normalized model, one-turn agent |
//!
//! Coverage ledger: the pre-pruning Cartesian product is 24 and all 24 cells
//! are recorded; none is unit-only. Each explicit test maps to
//! `tests/cassettes/openrouter/tool_truncation_contract_matrix/<test-name>.yaml`.
//! The inexpensive mini routes are pinned to the same OpenAI upstream so the
//! gateway contract, not routing variance, is under test. Assertions cover
//! exact wire arguments and reasons, native/normalized model surfaces, raw
//! streaming assembly, loud rejection of malformed completed calls, and
//! exact-once agent invocation of the valid control.
//!
//! | recorded cells | exact fixture set |
//! |---|---|
//! | all 24 | `tests/cassettes/openrouter/tool_truncation_contract_matrix/{blocking,streaming}_{gpt4o,gpt41}_{low,mid,complete}_{model,agent}.yaml` |

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use anyhow::Result;
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

use super::super::support::with_openrouter_tool_truncation_cassette_result;

const PREAMBLE: &str = "Call file_report exactly once. Copy the entire user incident verbatim into the required summary argument. Do not answer in prose.";
const PROMPT: &str = "The cache warmer raced the artifact uploader, the retry storm saturated the queue, three regions were drained by hand, dashboards lagged nine minutes, and rollback took forty minutes.";

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
enum Budget {
    Low,
    Mid,
    Complete,
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
    budget: Budget,
    surface: Surface,
}

#[derive(Debug, Default)]
struct Observation {
    finish_reason: Option<FinishReason>,
    arguments: Vec<Value>,
    errors: Vec<String>,
    invocations: usize,
}

type SharedObservation = Arc<Mutex<Option<Observation>>>;

fn cell(transport: Transport, model: Model, budget: Budget, surface: Surface) -> Cell {
    Cell {
        transport,
        model,
        budget,
        surface,
    }
}

fn model_name(model: Model) -> &'static str {
    match model {
        Model::Gpt4oMini => "openai/gpt-4o-mini",
        Model::Gpt41Mini => "openai/gpt-4.1-mini",
    }
}

fn max_tokens(budget: Budget) -> u64 {
    match budget {
        Budget::Low => 16,
        Budget::Mid => 32,
        Budget::Complete => 48,
    }
}

fn tool_definition() -> rig::completion::ToolDefinition {
    rig::completion::ToolDefinition {
        name: "file_report".to_owned(),
        description: "File an incident report".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "summary": { "type": "string" } },
            "required": ["summary"]
        }),
    }
}

fn request(model: &openrouter::CompletionModel, cell: Cell) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .preamble(PREAMBLE.to_owned())
        .tool(tool_definition())
        .additional_params(json!({
            "tool_choice": "required",
            "provider": { "order": ["OpenAI"], "allow_fallbacks": false }
        }))
        .max_tokens(max_tokens(cell.budget))
        .build()
}

fn calls(choice: &[AssistantContent]) -> Vec<Value> {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call.function.arguments.clone()),
            _ => None,
        })
        .collect()
}

#[derive(Clone)]
struct FileReport {
    invocations: Arc<AtomicUsize>,
}

#[derive(Debug, Deserialize, Serialize)]
struct FileReportArgs {
    summary: String,
}

#[derive(Debug, thiserror::Error)]
#[error("file report failed")]
struct FileReportError;

impl Tool for FileReport {
    const NAME: &'static str = "file_report";
    type Error = FileReportError;
    type Args = FileReportArgs;
    type Output = String;

    fn description(&self) -> String {
        "File an incident report".to_owned()
    }
    fn parameters(&self) -> Value {
        tool_definition().parameters
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> std::result::Result<Self::Output, Self::Error> {
        self.invocations.fetch_add(1, Ordering::SeqCst);
        Ok(args.summary)
    }
}

async fn run_model(client: openrouter::Client, cell: Cell) -> Observation {
    let model = client.completion_model(model_name(cell.model));
    match cell.transport {
        Transport::Blocking => match model.raw_completion(request(&model, cell)).await {
            Ok(raw) => match raw.normalize("openrouter") {
                Ok(response) => Observation {
                    finish_reason: response.finish_reason(),
                    arguments: calls(&response.choice),
                    ..Default::default()
                },
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
    let invocations = Arc::new(AtomicUsize::new(0));
    let agent = client
        .agent(model_name(cell.model))
        .preamble(PREAMBLE)
        .tool(FileReport {
            invocations: Arc::clone(&invocations),
        })
        .additional_params(json!({
            "tool_choice": "required",
            "provider": { "order": ["OpenAI"], "allow_fallbacks": false }
        }))
        .max_tokens(max_tokens(cell.budget))
        .default_max_turns(1)
        .build();
    let mut errors = Vec::new();
    match cell.transport {
        Transport::Blocking => {
            if let Err(error) = agent.prompt(PROMPT).await {
                errors.push(error.to_string());
            }
        }
        Transport::Streaming => {
            let mut stream = agent
                .stream_chat(PROMPT, Vec::<rig::completion::Message>::new())
                .max_turns(1)
                .stream()
                .await;
            errors = crate::support::collect_stream_observation(&mut stream)
                .await
                .errors;
        }
    }
    Observation {
        errors,
        invocations: invocations.load(Ordering::SeqCst),
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

fn recorded_finish_and_arguments(scenario: &str, transport: Transport) -> (String, Vec<String>) {
    match transport {
        Transport::Blocking => {
            let response = recorded_response(scenario);
            let finish = response["choices"][0]["finish_reason"]
                .as_str()
                .unwrap_or_default()
                .to_owned();
            let arguments = response["choices"][0]["message"]["tool_calls"]
                .as_array()
                .map(|calls| {
                    calls
                        .iter()
                        .filter_map(|call| {
                            call["function"]["arguments"].as_str().map(str::to_owned)
                        })
                        .collect()
                })
                .unwrap_or_default();
            (finish, arguments)
        }
        Transport::Streaming => {
            let mut finish = String::new();
            let mut arguments = Vec::<String>::new();
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
                        if arguments.len() <= index {
                            arguments.resize(index + 1, String::new());
                        }
                        if let Some(fragment) = call["function"]["arguments"].as_str() {
                            arguments[index].push_str(fragment);
                        }
                    }
                }
            }
            (finish, arguments)
        }
    }
}

fn assert_cell(scenario: &str, cell: Cell, observed: SharedObservation) {
    let request = recorded_request(scenario);
    assert_eq!(
        request["model"],
        model_name(cell.model),
        "{scenario}: model"
    );
    assert_eq!(
        request["max_tokens"],
        max_tokens(cell.budget),
        "{scenario}: cap"
    );
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
        Some(1),
        "{scenario}: tool schema"
    );
    assert_eq!(
        request["stream"].as_bool().unwrap_or(false),
        cell.transport == Transport::Streaming,
        "{scenario}: transport"
    );

    let (finish, wire_arguments) = recorded_finish_and_arguments(scenario, cell.transport);
    assert_eq!(wire_arguments.len(), 1, "{scenario}: exactly one wire call");
    let complete = cell.budget == Budget::Complete;
    if complete {
        assert_eq!(finish, "tool_calls", "{scenario}: completion control");
        let value: Value =
            serde_json::from_str(&wire_arguments[0]).expect("complete wire arguments");
        assert_eq!(
            value["summary"], PROMPT,
            "{scenario}: exact complete arguments"
        );
    } else {
        assert_eq!(
            finish, "tool_calls",
            "{scenario}: malformed completed-call premise"
        );
        assert!(
            !wire_arguments[0].is_empty(),
            "{scenario}: OpenRouter boundary is non-empty"
        );
        assert!(
            serde_json::from_str::<Value>(&wire_arguments[0]).is_err(),
            "{scenario}: partial JSON premise"
        );
    }

    let observation = observed
        .lock()
        .expect("observation mutex poisoned")
        .take()
        .expect("cell observation");
    match cell.surface {
        Surface::Model if complete => {
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
            assert_eq!(observation.arguments.len(), 1, "{scenario}");
            assert_eq!(observation.arguments[0]["summary"], PROMPT, "{scenario}");
        }
        Surface::Model => {
            assert_eq!(
                observation.errors.len(),
                1,
                "{scenario}: malformed non-length call stays loud: {:?}",
                observation.errors
            );
            assert!(
                observation.arguments.is_empty(),
                "{scenario}: malformed call is never emitted"
            );
            if cell.transport == Transport::Streaming {
                assert_eq!(
                    observation.finish_reason,
                    Some(FinishReason::ToolCalls),
                    "{scenario}: terminal metadata survives the in-band error"
                );
            }
        }
        Surface::Agent if complete => assert_eq!(
            observation.invocations, 1,
            "{scenario}: complete call invoked once"
        ),
        Surface::Agent => {
            assert_eq!(
                observation.invocations, 0,
                "{scenario}: partial call is never invoked"
            );
            assert!(
                !observation.errors.is_empty(),
                "{scenario}: malformed non-length call must fail the agent turn"
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
async fn blocking_gpt4o_low_model() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/blocking_gpt4o_low_model";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Budget::Low,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/blocking_gpt4o_low_model",
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
async fn blocking_gpt4o_low_agent() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/blocking_gpt4o_low_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Budget::Low,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/blocking_gpt4o_low_agent",
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
async fn blocking_gpt4o_mid_model() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/blocking_gpt4o_mid_model";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Budget::Mid,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/blocking_gpt4o_mid_model",
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
async fn blocking_gpt4o_mid_agent() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/blocking_gpt4o_mid_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Budget::Mid,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/blocking_gpt4o_mid_agent",
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
async fn blocking_gpt4o_complete_model() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/blocking_gpt4o_complete_model";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Budget::Complete,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/blocking_gpt4o_complete_model",
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
async fn blocking_gpt4o_complete_agent() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/blocking_gpt4o_complete_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Budget::Complete,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/blocking_gpt4o_complete_agent",
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
async fn blocking_gpt41_low_model() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/blocking_gpt41_low_model";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Budget::Low,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/blocking_gpt41_low_model",
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
async fn blocking_gpt41_low_agent() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/blocking_gpt41_low_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Budget::Low,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/blocking_gpt41_low_agent",
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
async fn blocking_gpt41_mid_model() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/blocking_gpt41_mid_model";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Budget::Mid,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/blocking_gpt41_mid_model",
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
async fn blocking_gpt41_mid_agent() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/blocking_gpt41_mid_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Budget::Mid,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/blocking_gpt41_mid_agent",
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
async fn blocking_gpt41_complete_model() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/blocking_gpt41_complete_model";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Budget::Complete,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/blocking_gpt41_complete_model",
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
async fn blocking_gpt41_complete_agent() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/blocking_gpt41_complete_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Budget::Complete,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/blocking_gpt41_complete_agent",
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
async fn streaming_gpt4o_low_model() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/streaming_gpt4o_low_model";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Budget::Low,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/streaming_gpt4o_low_model",
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
async fn streaming_gpt4o_low_agent() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/streaming_gpt4o_low_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Budget::Low,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/streaming_gpt4o_low_agent",
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
async fn streaming_gpt4o_mid_model() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/streaming_gpt4o_mid_model";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Budget::Mid,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/streaming_gpt4o_mid_model",
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
async fn streaming_gpt4o_mid_agent() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/streaming_gpt4o_mid_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Budget::Mid,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/streaming_gpt4o_mid_agent",
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
async fn streaming_gpt4o_complete_model() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/streaming_gpt4o_complete_model";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Budget::Complete,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/streaming_gpt4o_complete_model",
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
async fn streaming_gpt4o_complete_agent() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/streaming_gpt4o_complete_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Budget::Complete,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/streaming_gpt4o_complete_agent",
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
async fn streaming_gpt41_low_model() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/streaming_gpt41_low_model";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Budget::Low,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/streaming_gpt41_low_model",
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
async fn streaming_gpt41_low_agent() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/streaming_gpt41_low_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Budget::Low,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/streaming_gpt41_low_agent",
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
async fn streaming_gpt41_mid_model() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/streaming_gpt41_mid_model";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Budget::Mid,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/streaming_gpt41_mid_model",
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
async fn streaming_gpt41_mid_agent() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/streaming_gpt41_mid_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Budget::Mid,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/streaming_gpt41_mid_agent",
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
async fn streaming_gpt41_complete_model() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/streaming_gpt41_complete_model";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Budget::Complete,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/streaming_gpt41_complete_model",
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
async fn streaming_gpt41_complete_agent() -> Result<()> {
    const S: &str = "tool_truncation_contract_matrix/streaming_gpt41_complete_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Budget::Complete,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openrouter_tool_truncation_cassette_result(
        "tool_truncation_contract_matrix/streaming_gpt41_complete_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
