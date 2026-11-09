//! Live Chat Completions tool-call lifecycle and argument-integrity matrix.
//!
//! The space is 2 transports × 2 models × 3 call shapes × 2
//! public surfaces = 24 cells, of which the two below are tests. The shapes
//! cover a deliberate zero-argument call, a nested object containing an array
//! and Unicode, and two parallel calls. Model cells assert on the normalized
//! response the wire's decoder folds the reply into; agent cells prove
//! exact-once invocation. Streaming cells additionally reassemble every id,
//! name, and argument fragment from their fixtures.
//!
//! Auto/none/specific tool-choice controls are assigned to the separate
//! request-shape matrix; this matrix fixes choice to `any` so call shape
//! rather than model discretion is the independent variable. Sequential
//! call/result/follow-up history is assigned to the content/history matrix.
//!
//! | dimension | values |
//! |---|---|
//! | transport | blocking, streaming |
//! | model | `mistral-small-latest`, `ministral-3b-latest` |
//! | call shape | zero arguments, nested Unicode/array object, parallel pair |
//! | surface | model, one-turn agent |
//!
//! Each test maps to
//! `crates/rig-cassette/fixtures/cassettes/mistral/tool_lifecycle_matrix/<test-name>.yaml`.
//! The small and 3B aliases are inexpensive, currently served, tool-capable
//! model families. Assertions span request schemas, native blocking/streaming
//! assembly, normalized ids/names/arguments/order, and exact-once agent
//! dispatch. Synthetic malformed call objects stay beside the deserializers as
//! units because model output cannot deterministically produce them.
//!
//! | recorded cells | exact fixture set |
//! |---|---|
//! | 2 of 24 | `crates/rig-cassette/fixtures/cassettes/mistral/tool_lifecycle_matrix/{blocking_ministral_3b_zero_agent,streaming_mistral_small_nested_agent}.yaml` |

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
