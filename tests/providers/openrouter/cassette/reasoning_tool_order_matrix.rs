//! Live OpenRouter reasoning/tool block-order regression matrix.
//!
//! The five 24-cell matrices in this PR found a transport-parity gap that
//! requires a reasoning-capable route: blocking responses normalized tool
//! calls before reasoning, while the shared streaming adapter emitted the
//! same turn as reasoning then tool calls. These supplemental cells pin
//! Anthropic through OpenRouter because that live route exposes plaintext
//! reasoning beside tool calls on both transports.
//!
//! The finite raw-order space is 2 transports × 2 call shapes = 4 cells. Two
//! additional blocking/streaming agent-loop controls replay the signed
//! single-tool turn into a live follow-up, for 6 recorded cells total. The
//! upstream and reasoning budget are intentionally fixed:
//! OpenRouter is pinned to Anthropic with fallbacks disabled, extended
//! reasoning is fixed at its supported 1024-token minimum. Raw responses are
//! normalized at the exact boundary that regressed; agent controls prove the
//! streamed signature reaches replay history and is accepted by the next live
//! call. Parallel agent loops are the only pruned candidate cells: signature
//! attachment/serialization is independent of tool count, while raw cells
//! already pin parallel order and exact cardinality on both transports. No
//! other cells are pruned or unit-only. Each test maps one-to-one to
//! `tests/cassettes/openrouter/reasoning_tool_order_matrix/<test-name>.yaml`.
//!
//! | dimension | values |
//! |---|---|
//! | transport | blocking, streaming |
//! | call shape | one call, two parallel calls |
//! | model/upstream | `anthropic/claude-haiku-4.5` / Anthropic pinned |
//! | reasoning | `max_tokens: 1024` |
//! | surface | normalized raw model for both shapes; two-turn agent for single |
//!
//! | recorded cells | fixtures |
//! |---|---|
//! | 4 raw order | `{blocking,streaming}_{single,parallel}.yaml` |
//! | 2 signed agent loops | `{blocking,streaming}_signed_agent_roundtrip.yaml` |

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use anyhow::{Result, ensure};
use futures::StreamExt as _;
use rig::completion::{AssistantContent, CompletionModel, NormalizeCompletionResponse};
use rig::prelude::*;
use rig::providers::openrouter;
use rig::streaming::StreamingCompletionResponse;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::super::support::with_openrouter_reasoning_tool_order_cassette_result;

const MODEL: &str = "anthropic/claude-haiku-4.5";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Transport {
    Blocking,
    Streaming,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Single,
    Parallel,
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    transport: Transport,
    shape: Shape,
}

type SharedChoice = Arc<Mutex<Option<Vec<AssistantContent>>>>;

#[derive(Clone)]
struct Lookup {
    invocations: Arc<AtomicUsize>,
}

#[derive(Debug, Deserialize, Serialize)]
struct LookupArgs {
    value: String,
}

#[derive(Debug, thiserror::Error)]
#[error("lookup failed")]
struct LookupError;

impl Tool for Lookup {
    const NAME: &'static str = "lookup";
    type Error = LookupError;
    type Args = LookupArgs;
    type Output = String;

    fn description(&self) -> String {
        "Record one lookup value".to_owned()
    }

    fn parameters(&self) -> Value {
        tool(Self::NAME).parameters
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> std::result::Result<Self::Output, Self::Error> {
        assert_eq!(args.value, "cobalt");
        self.invocations.fetch_add(1, Ordering::SeqCst);
        Ok("lookup recorded; answer exactly DONE without another tool".to_owned())
    }
}

fn tool(name: &str) -> rig::completion::ToolDefinition {
    rig::completion::ToolDefinition {
        name: name.to_owned(),
        description: format!("Record the required {name} value"),
        parameters: json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
            "additionalProperties": false
        }),
    }
}

fn expected_names(shape: Shape) -> &'static [&'static str] {
    match shape {
        Shape::Single => &["lookup"],
        Shape::Parallel => &["alpha", "beta"],
    }
}

fn prompt(shape: Shape) -> &'static str {
    match shape {
        Shape::Single => {
            "Think briefly about the instruction, then call lookup exactly once with value cobalt. Do not answer in prose."
        }
        Shape::Parallel => {
            "Think briefly about the instruction, then call alpha with value red and beta with value blue in the same turn and in that order. Do not answer in prose."
        }
    }
}

fn request(model: &openrouter::CompletionModel, cell: Cell) -> rig::completion::CompletionRequest {
    let mut builder = model
        .completion_request(prompt(cell.shape))
        .preamble("Reason first, then obey the requested tool calls exactly.".to_owned())
        .additional_params(json!({
            "reasoning": { "max_tokens": 1024 },
            "parallel_tool_calls": cell.shape == Shape::Parallel,
            "provider": { "order": ["Anthropic"], "allow_fallbacks": false }
        }))
        .max_tokens(1200);
    for name in expected_names(cell.shape) {
        builder = builder.tool(tool(name));
    }
    builder.build()
}

async fn run_cell(client: openrouter::Client, cell: Cell, observed: SharedChoice) -> Result<()> {
    let model = client.completion_model(MODEL);
    let choice = match cell.transport {
        Transport::Blocking => {
            model
                .raw_completion(request(&model, cell))
                .await?
                .normalize("openrouter")?
                .choice
        }
        Transport::Streaming => {
            let raw = model.raw_stream(request(&model, cell)).await?;
            let normalized = rig::streaming::normalize_stream(raw, false, |terminal| {
                Ok::<_, rig::completion::CompletionError>(("openrouter", terminal).into())
            });
            let mut stream = StreamingCompletionResponse::stream("openrouter", normalized);
            while let Some(item) = stream.next().await {
                item?;
            }
            stream.choice.into_iter().collect()
        }
    };

    *observed.lock().expect("choice observation poisoned") = Some(choice);
    Ok(())
}

async fn run_signed_agent(
    client: openrouter::Client,
    transport: Transport,
    invocations: Arc<AtomicUsize>,
) -> Result<()> {
    let agent = client
        .agent(MODEL)
        .preamble(
            "Reason before the requested first tool call. After its result, answer exactly DONE without calling another tool.",
        )
        .tool(Lookup { invocations })
        .additional_params(json!({
            "reasoning": { "max_tokens": 1024 },
            "provider": { "order": ["Anthropic"], "allow_fallbacks": false }
        }))
        .max_tokens(1200)
        .default_max_turns(2)
        .build();

    match transport {
        Transport::Blocking => {
            rig::completion::Prompt::prompt(&agent, prompt(Shape::Single)).await?;
        }
        Transport::Streaming => {
            let mut stream = rig::streaming::StreamingChat::stream_chat(
                &agent,
                prompt(Shape::Single),
                Vec::<rig::completion::Message>::new(),
            )
            .max_turns(2)
            .await;
            while let Some(item) = stream.next().await {
                item?;
            }
        }
    }
    Ok(())
}

fn recorded_request(scenario: &str) -> Value {
    crate::cassettes::recorded_json_request("openrouter", scenario)
}

fn recorded_response(scenario: &str) -> Value {
    crate::cassettes::recorded_json_response("openrouter", scenario)
}

fn recorded_frames(scenario: &str) -> Vec<Value> {
    crate::cassettes::recorded_sse_json_frames("openrouter", scenario)
}

fn assert_fixture(scenario: &str, cell: Cell) {
    let request = recorded_request(scenario);
    assert_eq!(request["model"], MODEL, "{scenario}: model");
    assert_eq!(request["reasoning"]["max_tokens"], 1024, "{scenario}");
    assert_eq!(request["provider"]["order"], json!(["Anthropic"]));
    assert_eq!(request["provider"]["allow_fallbacks"], false);
    assert_eq!(
        request["stream"].as_bool().unwrap_or(false),
        cell.transport == Transport::Streaming,
        "{scenario}: transport"
    );
    let request_names = request["tools"]
        .as_array()
        .expect("recorded tools")
        .iter()
        .map(|tool| tool["function"]["name"].as_str().expect("tool name"))
        .collect::<Vec<_>>();
    assert_eq!(request_names, expected_names(cell.shape), "{scenario}");

    match cell.transport {
        Transport::Blocking => {
            let response = recorded_response(scenario);
            assert_eq!(response["provider"], "Anthropic", "{scenario}: route");
            let message = &response["choices"][0]["message"];
            assert!(
                message["reasoning"]
                    .as_str()
                    .is_some_and(|reasoning| !reasoning.is_empty())
                    || message["reasoning_details"]
                        .as_array()
                        .is_some_and(|details| !details.is_empty()),
                "{scenario}: blocking reasoning premise"
            );
            assert!(
                message["reasoning_details"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .any(|detail| detail["signature"]
                        .as_str()
                        .is_some_and(|signature| !signature.is_empty())),
                "{scenario}: blocking signature premise"
            );
            let names = message["tool_calls"]
                .as_array()
                .expect("blocking tool calls")
                .iter()
                .map(|call| call["function"]["name"].as_str().expect("call name"))
                .collect::<Vec<_>>();
            assert_eq!(names, expected_names(cell.shape), "{scenario}");
        }
        Transport::Streaming => {
            let frames = recorded_frames(scenario);
            assert!(
                frames.iter().any(|frame| frame["provider"] == "Anthropic"),
                "{scenario}: route"
            );
            assert!(
                frames.iter().any(|frame| {
                    let delta = &frame["choices"][0]["delta"];
                    delta["reasoning"]
                        .as_str()
                        .is_some_and(|reasoning| !reasoning.is_empty())
                        || delta["reasoning_details"]
                            .as_array()
                            .is_some_and(|details| !details.is_empty())
                }),
                "{scenario}: streaming reasoning premise"
            );
            assert!(
                frames.iter().any(|frame| {
                    frame["choices"][0]["delta"]["reasoning_details"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .any(|detail| {
                            detail["signature"]
                                .as_str()
                                .is_some_and(|signature| !signature.is_empty())
                        })
                }),
                "{scenario}: streaming signature premise"
            );
            let names = frames
                .iter()
                .flat_map(|frame| {
                    frame["choices"][0]["delta"]["tool_calls"]
                        .as_array()
                        .into_iter()
                        .flatten()
                })
                .filter_map(|call| call["function"]["name"].as_str())
                .collect::<Vec<_>>();
            assert_eq!(names, expected_names(cell.shape), "{scenario}");
        }
    }
}

fn assert_normalized_order(scenario: &str, cell: Cell, observed: SharedChoice) {
    let choice = observed
        .lock()
        .expect("choice observation poisoned")
        .take()
        .expect("test body should record a normalized choice");
    assert!(
        matches!(choice.first(), Some(AssistantContent::Reasoning(_))),
        "{scenario}: reasoning must be the first normalized block: {choice:#?}"
    );
    assert!(
        choice.iter().any(|content| matches!(
            content,
            AssistantContent::Reasoning(reasoning)
                if reasoning.content.iter().any(|part| matches!(
                    part,
                    rig::message::ReasoningContent::Text {
                        signature: Some(signature),
                        ..
                    } if !signature.is_empty()
                ))
        )),
        "{scenario}: live reasoning signature must survive normalization: {choice:#?}"
    );
    let names = choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call.function.name.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(names, expected_names(cell.shape), "{scenario}");
    let first_tool = choice
        .iter()
        .position(|content| matches!(content, AssistantContent::ToolCall(_)))
        .expect("normalized tool call");
    assert!(
        choice[..first_tool]
            .iter()
            .any(|content| matches!(content, AssistantContent::Reasoning(_))),
        "{scenario}: every tool-call sequence must follow reasoning"
    );
}

fn assert_signed_agent_fixture(scenario: &str, transport: Transport) {
    let interactions = crate::cassettes::recorded_interaction_bodies("openrouter", scenario);
    assert_eq!(
        interactions.len(),
        2,
        "{scenario}: one tool turn and one signed follow-up"
    );
    for (index, (request_body, response_body)) in interactions.iter().enumerate() {
        let request: Value = serde_json::from_str(request_body).expect("agent request JSON");
        assert_eq!(request["model"], MODEL, "{scenario}: request {index}");
        assert_eq!(
            request["reasoning"]["max_tokens"], 1024,
            "{scenario}: request {index} reasoning budget"
        );
        assert_eq!(
            request["provider"]["order"],
            json!(["Anthropic"]),
            "{scenario}: request {index} pinned route"
        );
        assert_eq!(
            request["provider"]["allow_fallbacks"], false,
            "{scenario}: request {index} fallback policy"
        );
        assert_eq!(
            request["stream"].as_bool().unwrap_or(false),
            transport == Transport::Streaming,
            "{scenario}: request {index} transport"
        );

        let routed_to_anthropic = match transport {
            Transport::Blocking => {
                serde_json::from_str::<Value>(response_body).expect("blocking agent response JSON")
                    ["provider"]
                    == "Anthropic"
            }
            Transport::Streaming => response_body
                .lines()
                .filter_map(|line| line.trim().strip_prefix("data:"))
                .map(str::trim)
                .filter(|data| *data != "[DONE]")
                .filter_map(|data| serde_json::from_str::<Value>(data).ok())
                .any(|frame| frame["provider"] == "Anthropic"),
        };
        assert!(
            routed_to_anthropic,
            "{scenario}: response {index} must prove Anthropic routing"
        );
    }

    let second: Value = serde_json::from_str(&interactions[1].0).expect("follow-up request JSON");
    let assistant = second["messages"]
        .as_array()
        .expect("follow-up messages")
        .iter()
        .find(|message| message["role"] == "assistant")
        .expect("replayed assistant tool turn");
    assert!(
        assistant["reasoning_details"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|detail| detail["signature"]
                .as_str()
                .is_some_and(|signature| !signature.is_empty())),
        "{scenario}: follow-up must replay the live reasoning signature"
    );
    assert_eq!(
        assistant["tool_calls"]
            .as_array()
            .expect("replayed tool calls")
            .len(),
        1
    );
}

async fn finish(scenario: &str, cell: Cell, observed: SharedChoice) {
    assert_fixture(scenario, cell);
    assert_normalized_order(scenario, cell, observed);
}

#[tokio::test]
async fn blocking_single() -> Result<()> {
    const S: &str = "reasoning_tool_order_matrix/blocking_single";
    let cell = Cell {
        transport: Transport::Blocking,
        shape: Shape::Single,
    };
    let observed = SharedChoice::default();
    let capture = Arc::clone(&observed);
    with_openrouter_reasoning_tool_order_cassette_result(
        "reasoning_tool_order_matrix/blocking_single",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    finish(S, cell, observed).await;
    Ok(())
}

#[tokio::test]
async fn blocking_parallel() -> Result<()> {
    const S: &str = "reasoning_tool_order_matrix/blocking_parallel";
    let cell = Cell {
        transport: Transport::Blocking,
        shape: Shape::Parallel,
    };
    let observed = SharedChoice::default();
    let capture = Arc::clone(&observed);
    with_openrouter_reasoning_tool_order_cassette_result(
        "reasoning_tool_order_matrix/blocking_parallel",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    finish(S, cell, observed).await;
    Ok(())
}

#[tokio::test]
async fn streaming_single() -> Result<()> {
    const S: &str = "reasoning_tool_order_matrix/streaming_single";
    let cell = Cell {
        transport: Transport::Streaming,
        shape: Shape::Single,
    };
    let observed = SharedChoice::default();
    let capture = Arc::clone(&observed);
    with_openrouter_reasoning_tool_order_cassette_result(
        "reasoning_tool_order_matrix/streaming_single",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    finish(S, cell, observed).await;
    Ok(())
}

#[tokio::test]
async fn streaming_parallel() -> Result<()> {
    const S: &str = "reasoning_tool_order_matrix/streaming_parallel";
    let cell = Cell {
        transport: Transport::Streaming,
        shape: Shape::Parallel,
    };
    let observed = SharedChoice::default();
    let capture = Arc::clone(&observed);
    with_openrouter_reasoning_tool_order_cassette_result(
        "reasoning_tool_order_matrix/streaming_parallel",
        |client| async move { run_cell(client, cell, capture).await },
    )
    .await?;
    finish(S, cell, observed).await;
    Ok(())
}

#[tokio::test]
async fn blocking_signed_agent_roundtrip() -> Result<()> {
    const S: &str = "reasoning_tool_order_matrix/blocking_signed_agent_roundtrip";
    let invocations = Arc::new(AtomicUsize::new(0));
    let capture = Arc::clone(&invocations);
    with_openrouter_reasoning_tool_order_cassette_result(
        "reasoning_tool_order_matrix/blocking_signed_agent_roundtrip",
        |client| async move { run_signed_agent(client, Transport::Blocking, capture).await },
    )
    .await?;
    ensure!(
        invocations.load(Ordering::SeqCst) == 1,
        "blocking signed-agent tool was not invoked exactly once"
    );
    assert_signed_agent_fixture(S, Transport::Blocking);
    Ok(())
}

#[tokio::test]
async fn streaming_signed_agent_roundtrip() -> Result<()> {
    const S: &str = "reasoning_tool_order_matrix/streaming_signed_agent_roundtrip";
    let invocations = Arc::new(AtomicUsize::new(0));
    let capture = Arc::clone(&invocations);
    with_openrouter_reasoning_tool_order_cassette_result(
        "reasoning_tool_order_matrix/streaming_signed_agent_roundtrip",
        |client| async move { run_signed_agent(client, Transport::Streaming, capture).await },
    )
    .await?;
    ensure!(
        invocations.load(Ordering::SeqCst) == 1,
        "streaming signed-agent tool was not invoked exactly once"
    );
    assert_signed_agent_fixture(S, Transport::Streaming);
    Ok(())
}
