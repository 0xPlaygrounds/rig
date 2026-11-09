//! Live Chat Completions tool-call lifecycle and argument-integrity matrix.
//!
//! The space is transport, model (`openai/gpt-4o-mini` and `openai/gpt-4.1-mini`, pinned to
//! OpenAI with fallbacks disabled so the gateway wire is stable), call shape (zero arguments, a
//! nested object with an array and Unicode, a parallel pair) and surface (normalized model
//! reply or one-turn agent). Tool choice is fixed to `required` so call shape, not model
//! discretion, is the independent variable; the request-shape and content matrices own the
//! other choices and sequential history.
//!
//! Model cells deserialize the provider-native response and normalize it, agent cells prove
//! exact-once dispatch, and streaming cells reassemble every id, name and argument fragment
//! from their fixtures under
//! `crates/rig-cassette/fixtures/cassettes/openrouter/tool_lifecycle_matrix/`. Malformed call
//! objects stay in the shared unit tests because they cannot be requested live.

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
