//! The ECS contract matrix on the OpenAI Responses wire (`gpt-5-mini`, no temperature: the gpt-5 family takes only its default; the route `gpt-5-nano`): every cell of
//! `tests/common/ecs_matrix/cells.rs` as an agent graph in a Bevy `World`,
//! served by the real adapter over the same recording as its producer in
//! `corpus_matrix_responses.rs`, and asserted against the cell,
//! then by its graph, its cut and its despawn (the driver is
//! `tests/common/ecs_matrix/world.rs`). This file holds the scenario
//! literals, the wire's models and the wire's `#[ignore]` reasons.

use rig::providers::openai::{GPT_5_MINI, GPT_5_NANO};

use super::super::support::{OpenAiCassette, stateless, with_openai_cassette};
use crate::ecs_matrix::{Wire, cells, world::run_world};

fn wire(client: &OpenAiCassette) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::OpenAiResponses,
        model: client.openai.completion(GPT_5_MINI),
        route: Some(client.openai.completion(GPT_5_NANO)),
        temperature: None,
        additional_params: None,
        options: Some(|| crate::ecs_matrix::corpus::TypedOptions::provider(stateless())),
    }
}

crate::matrix::native_matrix! {
    wrapper: with_openai_cassette, wire: reasoning_wire, run: run_world;
    #[tokio::test]
    #[ignore = "stale cassette: its request predates item-shaped history, and gpt-5-mini reported zero reasoning tokens in every re-record attempt"]
    reasoning_tool_streamed: ("reasoning_matrix_responses/tool_streamed", cells::REASONING_TOOL_STREAMED, "openai_responses_reasoning_tool_streamed");
}

#[ignore = "the Responses wire's `max_output_tokens` floor is 16; the corpus's second-turn cap is 5"]
#[tokio::test]
async fn shaping_max_tokens_second_turn() {
    crate::goldens::capture_world_programs(async {
        with_openai_cassette(
            "corpus_matrix_responses/shaping_max_tokens_second_turn",
            |client| async move {
                run_world(
                    &wire(&client),
                    &cells::SHAPING_MAX_TOKENS_SECOND_TURN,
                    |log| {
                        crate::goldens::world_golden_effects(
                            "openai_matrix_responses_shaping_max_tokens_second_turn",
                            log,
                        )
                    },
                )
                .await;
            },
        )
        .await;
    })
    .await
}

crate::matrix::case_matrix! {
    wrapper: with_openai_cassette, family: wire_matrix_case;
    #[ignore = "gpt-5-mini minimal returned an encrypted reasoning block on the first turn in all three attempts; record-openai-responses-shaping-thinking-second-turn-attempt-{1,2,3}.log; three attempts exhausted"]
    #[tokio::test]
    shaping_thinking_second_turn: ("corpus_matrix_responses/shaping_thinking_second_turn", shaping_thinking_second_turn_18, "openai_matrix_responses_shaping_thinking_second_turn");
    #[tokio::test]
    #[ignore = "gpt-5-mini minimal returned an encrypted reasoning part despite zero reasoning usage in all three attempts; record-openai-responses-off-attempt-{1,2,3}.log; three attempts exhausted"]
    reasoning_off: ("reasoning_matrix_responses/off", reasoning_off_21, "openai_matrix_responses_reasoning_off");
}

#[ignore = "gpt-5-mini on the Responses wire calls `lookup` with `leaf: true`, so the tool answers without nesting a completion; three recordings agreed"]
#[tokio::test]
async fn causal_completion_serial() {
    crate::goldens::capture_world_programs(async {
        with_openai_cassette(
            "corpus_matrix_responses/causal_completion_serial",
            |client| async move {
                run_world(&wire(&client), &cells::CAUSAL_COMPLETION_SERIAL, |log| {
                    crate::goldens::world_golden_effects(
                        "openai_matrix_responses_causal_completion_serial",
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

#[ignore = "gpt-5-mini on the Responses wire calls `lookup` with `leaf: true`, so the tool answers without nesting a completion; three recordings agreed"]
#[tokio::test]
async fn causal_completion_concurrent() {
    crate::goldens::capture_world_programs(async {
        with_openai_cassette(
            "corpus_matrix_responses/causal_completion_concurrent",
            |client| async move {
                run_world(
                    &wire(&client),
                    &cells::CAUSAL_COMPLETION_CONCURRENT,
                    |log| {
                        crate::goldens::world_golden_effects(
                            "openai_matrix_responses_causal_completion_concurrent",
                            log,
                        )
                    },
                )
                .await;
            },
        )
        .await;
    })
    .await
}

#[ignore = "gpt-5-mini on the Responses wire calls `lookup` with `leaf: true`, so the tool answers without nesting a completion; three recordings agreed"]
#[tokio::test]
async fn causal_completion_streamed() {
    crate::goldens::capture_world_programs(async {
        with_openai_cassette(
            "corpus_matrix_responses/causal_completion_streamed",
            |client| async move {
                run_world(&wire(&client), &cells::CAUSAL_COMPLETION_STREAMED, |log| {
                    crate::goldens::world_golden_effects(
                        "openai_matrix_responses_causal_completion_streamed",
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

// Reasoning matrix: the named thinking model, with the shared knob.
fn reasoning_wire(
    client: &OpenAiCassette,
) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: cells::ThinkingWire::OpenAiResponses,
        model: client.openai.completion(rig::providers::openai::GPT_5_MINI),
        route: None,
        temperature: None,
        additional_params: None,
        options: None,
    }
}

#[tokio::test]
#[ignore = "gpt-5-mini low: attempts 1 and 3 reported zero reasoning tokens on the tool turn; attempt 2 failed an over-strict final-turn assertion before cassette export; record-openai-responses-tool-unary-attempt-{1,2,3}.log; three attempts exhausted"]
async fn reasoning_tool_unary() {
    crate::goldens::capture_world_programs(async {
        with_openai_cassette(
            "reasoning_matrix_responses/tool_unary",
            |client| async move {
                run_world(
                    &reasoning_wire(&client),
                    &cells::REASONING_TOOL_UNARY,
                    |log| {
                        crate::goldens::world_golden_effects(
                            "openai_matrix_responses_reasoning_tool_unary",
                            log,
                        )
                    },
                )
                .await;
            },
        )
        .await;
    })
    .await
}
