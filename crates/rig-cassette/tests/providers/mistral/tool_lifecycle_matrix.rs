//! Live Chat Completions tool-call lifecycle and argument-integrity matrix.
//!
//! The space is transport (blocking, streaming), model (`mistral-small-latest`,
//! `ministral-3b-latest`), call shape (zero arguments, a nested object with an
//! array and Unicode, a parallel pair), and surface (model, one-turn agent).
//! Model cells assert on the normalized response the wire's decoder folds the
//! reply into; agent cells prove exact-once invocation. Streaming cells also
//! reassemble every id, name, and argument fragment from their fixtures.
//!
//! Tool choice is fixed to `any`, so call shape rather than model discretion is
//! the independent variable; tool-choice controls belong to the request-shape
//! matrix and sequential history to the content/history matrix. Malformed call
//! objects stay beside the deserializers as unit tests because model output
//! cannot produce them deterministically. Each test records
//! `crates/rig-cassette/fixtures/cassettes/mistral/tool_lifecycle_matrix/<test-name>.yaml`.

use anyhow::Result;
use rig_test_support::cassette_models::OpenAiModels;

use crate::matrix::tool_lifecycle_matrix::{
    Cell, LifecycleWire, Observation, Shape::*, Surface::*, Transport::*, cell,
};

use super::support::with_mistral_cassette_result;

const WIRE: LifecycleWire = LifecycleWire {
    provider: "mistral",
    tool_choice: "any",
};

async fn run_cell(client: OpenAiModels, cell: Cell) -> Observation {
    WIRE.run_agent(rig::AgentBuilder::new(client.completion(cell.model)), cell)
        .await
}

fn assert_cell(scenario: &str, cell: Cell, observation: Observation) {
    WIRE.assert_cell(scenario, cell, observation);
}

// The literal wrapper calls below are intentionally explicit: cassette safety
// parses source rather than macro expansion.

crate::matrix::case_matrix! {
    wrapper: with_mistral_cassette_result, family: tool_lifecycle_matrix_case;
    # [tokio :: test]
    blocking_ministral_3b_zero_agent: ("tool_lifecycle_matrix/blocking_ministral_3b_zero_agent", configured, cell(Blocking, "ministral-3b-latest", Zero, Agent));
    # [tokio :: test]
    streaming_mistral_small_nested_agent: ("tool_lifecycle_matrix/streaming_mistral_small_nested_agent", configured, cell(Streaming, "mistral-small-latest", Nested, Agent));
}
