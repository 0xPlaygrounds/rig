//! The ECS contract matrix on the Venice wire (`mistral-small-3-2-24b-instruct`, the model the suite records tools under, temperature 0; the route is the same model under `golden/model:fast`: every other Venice model tried either thinks (its reasoning part in the history is refused by the Mistral tokenizer on the next turn) or re-calls the tool after its result, so the route is observable on the bus, not on the wire): every cell of
//! `tests/common/ecs_matrix/cells.rs` as an agent graph in a Bevy `World`,
//! served by the real adapter over the same recording as its producer in
//! `corpus_matrix.rs`, and asserted against the cell,
//! then by its graph, its cut and its despawn (the driver is
//! `tests/common/ecs_matrix/world.rs`). This file holds the scenario
//! literals, the wire's models and the wire's `#[ignore]` reasons.

use rig::providers::venice::MISTRAL_SMALL_3_2_24B;
use rig_test_support::cassette_models::OpenAiModels;

use super::super::support::with_venice_cassette;
use crate::ecs_matrix::{Wire, cells, world::run_world};

fn wire(client: &OpenAiModels) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Venice,
        model: client.completion(MISTRAL_SMALL_3_2_24B),
        route: Some(client.completion(MISTRAL_SMALL_3_2_24B)),
        temperature: Some(0.0),
        additional_params: None,
        options: None,
    }
}

crate::matrix::native_matrix! {
    wrapper: with_venice_cassette, wire: wire, run: run_world;
    #[tokio::test]
    #[ignore = "stale cassette: its request predates the rebuilt Chat history, and the model skipped the tool call in all three re-record attempts"]
    causal_completion_serial: ("corpus_matrix/causal_completion_serial", cells::CAUSAL_COMPLETION_SERIAL, "venice_causal_completion_serial");
    #[tokio::test]
    #[ignore = "stale cassette: its request predates item-shaped history, and the model skipped the tool call in every re-record attempt"]
    causal_completion_concurrent: ("corpus_matrix/causal_completion_concurrent", cells::CAUSAL_COMPLETION_CONCURRENT, "venice_causal_completion_concurrent");
    #[tokio::test]
    #[ignore = "stale cassette: its request predates the rebuilt Chat history, and the model skipped the tool call in all three re-record attempts"]
    causal_completion_streamed: ("corpus_matrix/causal_completion_streamed", cells::CAUSAL_COMPLETION_STREAMED, "venice_causal_completion_streamed");
}

#[ignore = "Venice's gateway never answers the two-turn output-tool program (a 2,200 s hang, then a 500 `cannot send request`); two recordings agreed"]
#[tokio::test]
async fn output_tool_with_real_tool() {
    crate::goldens::capture_world_programs(async {
        with_venice_cassette(
            "corpus_matrix/output_tool_with_real_tool",
            |client| async move {
                run_world(&wire(&client), &cells::OUTPUT_TOOL_WITH_REAL_TOOL, |log| {
                    crate::goldens::world_golden_effects(
                        "venice_matrix_output_tool_with_real_tool",
                        log,
                    )
                })
                .await;
            },
        )
        .await;
    })
    .await
}

crate::matrix::case_matrix! {
    wrapper: with_venice_cassette, family: wire_matrix_case;
    #[ignore = "Venice thinking disabled answered directly without the required first-turn add call in all three attempts; record-venice-shaping-thinking-second-turn-attempt-{1,2,3}.log; three attempts exhausted"]
    #[tokio::test]
    shaping_thinking_second_turn: ("corpus_matrix/shaping_thinking_second_turn", shaping_thinking_second_turn_18, "venice_matrix_shaping_thinking_second_turn");
}

// Reasoning matrix: the named thinking model, with the shared knob.
fn reasoning_wire(
    client: &OpenAiModels,
) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: cells::ThinkingWire::Venice,
        model: client.completion(rig::providers::venice::QWEN3_235B_A22B_THINKING),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
        options: None,
    }
}
