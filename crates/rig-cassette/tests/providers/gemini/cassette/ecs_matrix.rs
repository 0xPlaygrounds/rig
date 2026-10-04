//! The ECS contract matrix on the Gemini REST wire (`gemini-3-flash-preview`, temperature 0; the route `gemini-3.1-flash-lite-preview`): every cell of
//! `tests/common/ecs_matrix/cells.rs` as an agent graph in a Bevy `World`,
//! served by the real adapter over the same recording as its producer in
//! `corpus_matrix.rs`, and asserted against the cell,
//! then by its graph, its cut and its despawn (the driver is
//! `tests/common/ecs_matrix/world.rs`). This file holds the scenario
//! literals, the wire's models and the wire's `#[ignore]` reasons.

use rig::providers::gemini::completion::{GEMINI_3_1_FLASH_LITE_PREVIEW, GEMINI_3_FLASH_PREVIEW};
use rig_test_support::cassette_models::GeminiModels;

use super::super::support::with_gemini_cassette;
use crate::ecs_matrix::{Wire, cells, world::run_world};

fn wire(
    client: &GeminiModels,
) -> Wire<
    rig::Model<
        rig::providers::gemini::completion::GenerateContent,
        rig::http_client::DynHttpClient,
    >,
> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Gemini,
        model: client.completion(GEMINI_3_FLASH_PREVIEW),
        route: Some(client.completion(GEMINI_3_1_FLASH_LITE_PREVIEW)),
        temperature: Some(0.0),
        additional_params: None,
    }
}

#[ignore = "the Gemini REST wire streams a function call as one whole part: no tool-call delta reaches the hook, the run answers"]
#[tokio::test]
async fn endings_tool_call_delta_stop() {
    crate::goldens::capture_world_programs(async {
        with_gemini_cassette(
            "corpus_matrix/endings_tool_call_delta_stop",
            |client| async move {
                run_world(
                    &wire(&client),
                    &cells::ENDINGS_TOOL_CALL_DELTA_STOP,
                    |log| {
                        crate::goldens::world_golden_effects(
                            "gemini_matrix_endings_tool_call_delta_stop",
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

#[ignore = "gemini-3-flash-preview returned a signed final_result call without visible reasoning in all three attempts; record-gemini-output-tool-thinking-attempt-{1,2,3}.log; three attempts exhausted"]
#[tokio::test]
async fn output_tool_thinking() {
    crate::goldens::capture_world_programs(async {
        with_gemini_cassette("corpus_matrix/output_tool_thinking", |client| async move {
            run_world(
                &reasoning_wire(&client),
                &cells::OUTPUT_TOOL_THINKING,
                |log| {
                    crate::goldens::world_golden_effects("gemini_matrix_output_tool_thinking", log)
                },
            )
            .await;
        })
        .await;
    })
    .await
}

#[ignore = "Gemini answers the output-tool call the model still makes under mode NONE with finish_reason MALFORMED_FUNCTION_CALL: the turn has no record"]
#[tokio::test]
async fn shaping_tool_choice_none_on_committed_output() {
    crate::goldens::capture_world_programs(async {
        with_gemini_cassette(
            "corpus_matrix/shaping_tool_choice_none_on_committed_output",
            |client| async move {
                run_world(
                    &wire(&client),
                    &cells::SHAPING_TOOL_CHOICE_NONE_ON_COMMITTED_OUTPUT,
                    |log| {
                        crate::goldens::world_golden_effects(
                            "gemini_matrix_shaping_tool_choice_none_on_committed_output",
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
    wrapper: with_gemini_cassette, family: wire_matrix_case;
    #[ignore = "gemini-3-flash-preview returned only text and a signature-only reasoning part with zero reasoning usage on the second turn in all three attempts; record-gemini-shaping-thinking-second-turn-attempt-{1,2,3}.log; three attempts exhausted"]
    #[tokio::test]
    shaping_thinking_second_turn: ("corpus_matrix/shaping_thinking_second_turn", shaping_thinking_second_turn_18, "gemini_matrix_shaping_thinking_second_turn");
    #[tokio::test]
    #[ignore = "gemini-3-flash-preview with thinkingBudget 128 returned a signed tool call without reasoning in all three attempts; record-gemini-tool-unary-attempt-{1,2,3}.log; three attempts exhausted"]
    reasoning_tool_unary: ("reasoning_matrix/tool_unary", reasoning_tool_unary_19, "gemini_matrix_reasoning_tool_unary");
    #[tokio::test]
    #[ignore = "gemini-3-flash-preview with thinkingBudget 128 returned a signed tool call without reasoning in all three attempts; record-gemini-tool-streamed-attempt-{1,2,3}.log; three attempts exhausted"]
    reasoning_tool_streamed: ("reasoning_matrix/tool_streamed", reasoning_tool_streamed_20, "gemini_matrix_reasoning_tool_streamed");
    #[tokio::test]
    #[ignore = "gemini-3-flash-preview accepted thinkingBudget 0 but returned a signature-only reasoning part despite zero reasoning usage in all three attempts; record-gemini-off-attempt-{1,2,3}.log; three attempts exhausted"]
    reasoning_off: ("reasoning_matrix/off", reasoning_off_21, "gemini_matrix_reasoning_off");
}

// Reasoning matrix: the named thinking model, with the shared knob.
fn reasoning_wire(
    client: &GeminiModels,
) -> Wire<
    rig::Model<
        rig::providers::gemini::completion::GenerateContent,
        rig::http_client::DynHttpClient,
    >,
> {
    Wire {
        thinking: cells::ThinkingWire::Gemini,
        model: client.completion("gemini-3-flash-preview"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}
