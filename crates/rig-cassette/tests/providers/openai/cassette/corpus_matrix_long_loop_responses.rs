//! The long tool loop's producer column on OpenAiResponses: gpt-4.1-mini (Responses).
//! One recording per live cell; the native twin (`ecs_matrix_long_loop_responses.rs`)
//! reuses each with strict matching. Programs, toolset and assertions are
//! `tests/common/ecs_matrix/long_loop.rs`'s; this file holds the scenario
//! literals and the wire's model.

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::ecs_matrix::{Wire, cells, long_loop};

fn wire(client: &OpenAiCassette) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: cells::ThinkingWire::OpenAiResponses,
        model: client.openai.completion("gpt-4.1-mini"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
        options: None,
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_openai_cassette, wire: wire, run: long_loop::run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    #[ignore = "a length-cut call is now kept with tolerant arguments and answered with an error result (change 2), so the cell's cut turn no longer ends the run and the shared long_loop assertions (every requested call dispatched, nothing dispatched under the cap) cannot hold; two live attempts failed on them; the cell needs redesigning for the new semantics"]
    output_cap_midway: ("long_loop_matrix_responses/output_cap_midway", long_loop::OUTPUT_CAP_MIDWAY, "openai_responses_long_loop_output_cap_midway");
}
