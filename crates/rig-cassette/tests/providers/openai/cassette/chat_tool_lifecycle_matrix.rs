//! Live Chat Completions tool-call lifecycle and argument-integrity matrix.
//!
//! The call shapes are a zero-argument call, a nested object containing an
//! array and Unicode, and two parallel calls. Tool choice is fixed to
//! `required` so call shape, not model discretion, is the independent
//! variable. Model cells read the normalized reply the driver decodes from
//! the provider-native response, agent cells prove exactly-once invocation,
//! and streaming cells also reassemble every id, name and argument fragment
//! from their fixtures.
//!
//! Malformed call objects stay unit tests, since model output cannot
//! deterministically produce those wire defects.

use anyhow::Result;
use futures::StreamExt as _;
use rig::completion::{AssistantContent, CompletionRequest};
use rig::streaming::{Item, StreamEvent};
use serde_json::{Value, json};

use crate::matrix::tool_lifecycle_matrix::{
    Cell, LifecycleWire, Observation, PREAMBLE, Shape, Shape::*, Surface, Surface::*, Transport,
    Transport::*, cell, expected_names, prompt, tool_definition,
};

use super::super::support::{OpenAiCassette, with_openai_cassette_result};

const WIRE: LifecycleWire = LifecycleWire {
    provider: "openai",
    tool_choice: "required",
};

fn request(cell: Cell) -> rig::completion::CompletionRequest {
    let mut builder = CompletionRequest::new(prompt(cell.shape))
        .preamble(PREAMBLE.to_owned())
        .parallel_tool_calls(cell.shape == Shape::Parallel)
        .additional_params(json!({ "tool_choice": WIRE.tool_choice }))
        .max_tokens(128);
    for name in expected_names(cell.shape) {
        builder = builder.tool(tool_definition(name));
    }
    builder
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
            .map(String::from)
            .collect(),
        calls.iter().map(|call| call.id.to_string()).collect(),
        calls
            .iter()
            .map(|call| call.function.arguments_value())
            .collect(),
    )
}

async fn run_model(client: OpenAiCassette, cell: Cell) -> Observation {
    let model = client.openai.chat(cell.model);
    match cell.transport {
        // The provider-native reply and the normalized view are one call now:
        // the driver decodes the native response and hands back the
        // normalization, keeping the native value on `CompletionResponse::raw`.
        Transport::Blocking => match model.call(request(cell)).await {
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
        Transport::Streaming => {
            let mut stream = match model.stream(request(cell)) {
                Ok(stream) => stream,
                Err(error) => {
                    return Observation {
                        errors: vec![error.to_string()],
                        ..Default::default()
                    };
                }
            };
            let mut observation = Observation::default();
            while let Some(item) = stream.next().await {
                match item {
                    Ok(Item::Event(StreamEvent::End {
                        content: AssistantContent::ToolCall(tool_call),
                        ..
                    })) => {
                        observation
                            .names
                            .push(tool_call.function.name.clone().into());
                        observation.ids.push(tool_call.id.to_string());
                        observation
                            .arguments
                            .push(tool_call.function.arguments_value());
                    }
                    Ok(_) => {}
                    Err(error) => observation.errors.push(error.to_string()),
                }
            }
            if let Ok(terminal) = stream.finish().await {
                observation.finish_reason = terminal.finish_reason();
            }
            observation
        }
    }
}

async fn run_cell(client: OpenAiCassette, cell: Cell) -> Observation {
    match cell.surface {
        Surface::Model => run_model(client, cell).await,
        Surface::Agent => {
            WIRE.run_agent(
                rig::AgentBuilder::new(client.chat.completion(cell.model)),
                cell,
            )
            .await
        }
    }
}

fn assert_cell(scenario: &str, cell: Cell, observation: Observation) {
    WIRE.assert_cell(scenario, cell, observation);
}

// The literal wrapper calls below are intentionally explicit: cassette safety
// parses source rather than macro expansion.

crate::matrix::case_matrix! {
    wrapper: with_openai_cassette_result, family: tool_lifecycle_matrix_case;
    # [tokio :: test]
    blocking_gpt4o_zero_agent: ("chat_tool_lifecycle_matrix/blocking_gpt4o_zero_agent", configured, cell(Blocking, "gpt-4o-mini", Zero, Agent));
    # [tokio :: test]
    streaming_gpt4o_nested_model: ("chat_tool_lifecycle_matrix/streaming_gpt4o_nested_model", configured, cell(Streaming, "gpt-4o-mini", Nested, Model));
}
