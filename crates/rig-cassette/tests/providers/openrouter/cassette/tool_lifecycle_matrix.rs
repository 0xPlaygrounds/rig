//! Live Chat Completions tool-call lifecycle and argument-integrity matrix.
//!
//! The space is 2 transports × 2 models × 3 call shapes × 2
//! public surfaces = 24 cells, of which the two below are tests. The shapes
//! cover a deliberate zero-argument call, a nested object containing an array
//! and Unicode, and two parallel calls. Model cells deserialize the
//! provider-native response and normalize it; agent cells prove exact-once
//! invocation. Streaming cells additionally reassemble every id, name, and
//! argument fragment from their fixtures.
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
//! Each test maps to
//! `crates/rig-cassette/fixtures/cassettes/openrouter/tool_lifecycle_matrix/<test-name>.yaml`.
//! Both cheap mini routes are pinned to OpenAI with fallbacks disabled, making
//! the gateway wire stable across model families. Assertions span request
//! schemas, native blocking/streaming assembly, normalized ids, names,
//! arguments and order, and agent exact-once dispatch. Synthetic malformed call
//! objects remain shared unit tests because they cannot be requested live.
//!
//! | recorded cells | exact fixture set |
//! |---|---|
//! | 2 of 24 | `crates/rig-cassette/fixtures/cassettes/openrouter/tool_lifecycle_matrix/{blocking_gpt41_zero_agent,streaming_gpt4o_nested_agent}.yaml` |

use anyhow::Result;
use rig_test_support::cassette_models::OpenAiModels;
use serde_json::{Value, json};

use crate::matrix::tool_lifecycle_matrix::{
    Cell, LifecycleWire, Observation, Shape::*, Surface::*, Transport, Transport::*, cell,
};

use super::super::support::{pinned_order, with_openrouter_cassette_result};

const WIRE: LifecycleWire = LifecycleWire {
    provider: "openrouter",
    tool_choice: "required",
};

async fn run_cell(client: OpenAiModels, cell: Cell) -> Observation {
    let builder = rig::AgentBuilder::new(client.completion(cell.model))
        .provider_options(pinned_order(&["OpenAI"]));
    WIRE.run_agent(builder, cell).await
}

fn recorded_provider(scenario: &str, transport: Transport) -> Value {
    match transport {
        Transport::Blocking => WIRE.recorded_response(scenario)["provider"].clone(),
        Transport::Streaming => WIRE
            .recorded_chunks(scenario)
            .into_iter()
            .filter_map(|chunk| chunk.get("provider").cloned())
            .next_back()
            .unwrap_or(Value::Null),
    }
}

fn assert_cell(scenario: &str, cell: Cell, observation: Observation) {
    let request = WIRE.recorded_request(scenario);
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
    WIRE.assert_cell(scenario, cell, observation);
}

// The literal wrapper calls below are intentionally explicit: cassette safety
// parses source rather than macro expansion.

crate::matrix::case_matrix! {
    wrapper: with_openrouter_cassette_result, family: tool_lifecycle_matrix_case;
    # [tokio :: test]
    blocking_gpt41_zero_agent: ("tool_lifecycle_matrix/blocking_gpt41_zero_agent", configured, cell(Blocking, "openai/gpt-4.1-mini", Zero, Agent));
    # [tokio :: test]
    streaming_gpt4o_nested_agent: ("tool_lifecycle_matrix/streaming_gpt4o_nested_agent", configured, cell(Streaming, "openai/gpt-4o-mini", Nested, Agent));
}
