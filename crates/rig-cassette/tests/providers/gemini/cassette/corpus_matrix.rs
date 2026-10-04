//! The ECS contract matrix's producers on the Gemini REST wire (`gemini-3-flash-preview`, temperature 0; the route `gemini-3.1-flash-lite-preview`): every cell of
//! `tests/common/ecs_matrix/cells.rs` on rig-agent's builder, recorded once
//! and replayed as the golden `gemini_<cell>` the world cells in
//! `ecs_matrix.rs` are compared to. The driver is
//! `tests/common/ecs_matrix/agent.rs`; this file holds the scenario
//! literals, the wire's models and the wire's `#[ignore]` reasons.
//!
//! Every scenario here is a new recording under `corpus_matrix/`; a cell of
//! the grid missing from this file reuses a recording the corpus already
//! had, whose producer stays where it is.

use rig::providers::gemini::completion::{GEMINI_3_1_FLASH_LITE_PREVIEW, GEMINI_3_FLASH_PREVIEW};
use rig_test_support::cassette_models::GeminiModels;

use super::super::support::with_gemini_cassette;
use crate::ecs_matrix::{Wire, agent::run_agent, cells};

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

crate::matrix::golden_matrix! {
    wrapper: with_gemini_cassette, wire: wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    output_tool_under_none_degrades: ("corpus_matrix/output_tool_under_none_degrades", cells::OUTPUT_TOOL_UNDER_NONE_DEGRADES, "gemini_output_tool_under_none_degrades");
    #[tokio::test]
    shaping_max_tokens_second_turn: ("corpus_matrix/shaping_max_tokens_second_turn", cells::SHAPING_MAX_TOKENS_SECOND_TURN, "gemini_shaping_max_tokens_second_turn");
    #[tokio::test]
    shaping_preamble_second_turn: ("corpus_matrix/shaping_preamble_second_turn", cells::SHAPING_PREAMBLE_SECOND_TURN, "gemini_shaping_preamble_second_turn");
    #[tokio::test]
    causal_completion_concurrent: ("corpus_matrix/causal_completion_concurrent", cells::CAUSAL_COMPLETION_CONCURRENT, "gemini_causal_completion_concurrent");
}

crate::matrix::golden_matrix! {
    wrapper: with_gemini_cassette, wire: wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    output_tool_choice_required: ("corpus_matrix/output_tool_choice_required", cells::OUTPUT_TOOL_CHOICE_REQUIRED, "gemini_output_tool_choice_required");
    #[tokio::test]
    shaping_route_on_first_turn: ("corpus_matrix/shaping_route_on_first_turn", cells::SHAPING_ROUTE_ON_FIRST_TURN, "gemini_shaping_route_on_first_turn");
    #[tokio::test]
    shaping_active_tools_none_second_turn: ("corpus_matrix/shaping_active_tools_none_second_turn", cells::SHAPING_ACTIVE_TOOLS_NONE_SECOND_TURN, "gemini_shaping_active_tools_none_second_turn");
    #[tokio::test]
    causal_completion_streamed: ("corpus_matrix/causal_completion_streamed", cells::CAUSAL_COMPLETION_STREAMED, "gemini_causal_completion_streamed");
}

#[ignore = "the Gemini REST wire streams a function call as one whole part: no tool-call delta reaches the hook, the run answers"]
#[tokio::test]
async fn endings_tool_call_delta_stop() {
    with_gemini_cassette(
        "corpus_matrix/endings_tool_call_delta_stop",
        |client| async move {
            run_agent(&wire(&client), &cells::ENDINGS_TOOL_CALL_DELTA_STOP, |_| {}).await;
        },
    )
    .await;
}

#[ignore = "gemini-3-flash-preview returned a signed final_result call without visible reasoning in all three attempts; record-gemini-output-tool-thinking-attempt-{1,2,3}.log; three attempts exhausted"]
#[tokio::test]
async fn output_tool_thinking() {
    with_gemini_cassette("corpus_matrix/output_tool_thinking", |client| async move {
        run_agent(
            &reasoning_wire(&client),
            &cells::OUTPUT_TOOL_THINKING,
            |_| panic!("unrecorded reasoning scenario: see this test\'s ignore disposition"),
        )
        .await;
    })
    .await;
}

#[ignore = "Gemini answers the output-tool call the model still makes under mode NONE with finish_reason MALFORMED_FUNCTION_CALL: the turn has no record"]
#[tokio::test]
async fn shaping_tool_choice_none_on_committed_output() {
    with_gemini_cassette(
        "corpus_matrix/shaping_tool_choice_none_on_committed_output",
        |client| async move {
            run_agent(
                &wire(&client),
                &cells::SHAPING_TOOL_CHOICE_NONE_ON_COMMITTED_OUTPUT,
                |_| {},
            )
            .await;
        },
    )
    .await;
}

crate::matrix::case_matrix! {
    wrapper: with_gemini_cassette, family: wire_matrix_case;
    #[ignore = "gemini-3-flash-preview returned only text and a signature-only reasoning part with zero reasoning usage on the second turn in all three attempts; record-gemini-shaping-thinking-second-turn-attempt-{1,2,3}.log; three attempts exhausted"]
    #[tokio::test]
    shaping_thinking_second_turn: ("corpus_matrix/shaping_thinking_second_turn", shaping_thinking_second_turn_9);
    #[tokio::test]
    #[ignore = "gemini-3-flash-preview with thinkingBudget 128 returned a signed tool call without reasoning in all three attempts; record-gemini-tool-unary-attempt-{1,2,3}.log; three attempts exhausted"]
    reasoning_tool_unary: ("reasoning_matrix/tool_unary", reasoning_tool_unary_10);
    #[tokio::test]
    #[ignore = "gemini-3-flash-preview with thinkingBudget 128 returned a signed tool call without reasoning in all three attempts; record-gemini-tool-streamed-attempt-{1,2,3}.log; three attempts exhausted"]
    reasoning_tool_streamed: ("reasoning_matrix/tool_streamed", reasoning_tool_streamed_11);
    #[tokio::test]
    #[ignore = "gemini-3-flash-preview accepted thinkingBudget 0 but returned a signature-only reasoning part despite zero reasoning usage in all three attempts; record-gemini-off-attempt-{1,2,3}.log; three attempts exhausted"]
    reasoning_off: ("reasoning_matrix/off", reasoning_off_12);
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
