//! The ECS contract matrix on the DeepSeek wire (`deepseek-chat`, temperature 0; the route `deepseek-reasoner`): every cell of
//! `tests/common/ecs_matrix/cells.rs` as an agent graph in a Bevy `World`,
//! served by the real adapter over the same recording as its producer in
//! `corpus_matrix.rs`, and asserted against the cell,
//! then by its graph, its cut and its despawn (the driver is
//! `tests/common/ecs_matrix/world.rs`). This file holds the scenario
//! literals, the wire's models and the wire's `#[ignore]` reasons.

use crate::deepseek::support::with_deepseek_cassette;
use crate::ecs_matrix::{Wire, cells, world::run_world};
use rig_test_support::cassette_models::OpenAiModels;

fn wire(client: &OpenAiModels) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::DeepSeek,
        model: client.completion("deepseek-chat"),
        route: Some(client.completion("deepseek-reasoner")),
        temperature: Some(0.0),
        additional_params: None,
        options: None,
    }
}

crate::matrix::native_matrix! {
    wrapper: with_deepseek_cassette, wire: wire, run: run_world;
    #[tokio::test]
    causal_completion_serial: ("corpus_matrix/causal_completion_serial", cells::CAUSAL_COMPLETION_SERIAL, "deepseek_causal_completion_serial");
}

crate::matrix::case_matrix! {
    wrapper: with_deepseek_cassette, family: wire_matrix_case;
    #[ignore = "deepseek-flash rejects enabling thinking after a disabled tool turn with HTTP 400: prior reasoning_content required; record-deepseek-shaping-thinking-second-turn-attempt-1.log and https://api-docs.deepseek.com/guides/thinking_mode/; the model refuses this setting transition"]
    #[tokio::test]
    shaping_thinking_second_turn: ("corpus_matrix/shaping_thinking_second_turn", shaping_thinking_second_turn_18, "deepseek_matrix_shaping_thinking_second_turn");
}

// Reasoning matrix: the named thinking model, with the shared knob.
fn reasoning_wire(
    client: &OpenAiModels,
) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: cells::ThinkingWire::DeepSeek,
        model: client.completion("deepseek-flash"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
        options: None,
    }
}
