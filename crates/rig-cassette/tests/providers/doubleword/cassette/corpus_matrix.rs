//! The ECS contract matrix's producers on the Doubleword wire (`Qwen/Qwen3.5-397B-A17B-FP8`, the model the suite records tool scenarios under, `reasoning_effort: none` so a tiny cap cuts an answer and not hidden tokens, temperature 0; the route `Qwen/Qwen3.5-9B`): every cell of
//! `tests/common/ecs_matrix/cells.rs` on rig-agent's builder, recorded once
//! and replayed as the golden `doubleword_<cell>` the world cells in
//! `ecs_matrix.rs` are compared to. The driver is
//! `tests/common/ecs_matrix/agent.rs`; this file holds the scenario
//! literals, the wire's models and the wire's `#[ignore]` reasons.
//!
//! Every scenario here is a new recording under `corpus_matrix/`; a cell of
//! the grid missing from this file reuses a recording the corpus already
//! had, whose producer stays where it is.

use rig::providers::doubleword::{QWEN3_5_9B, QWEN3_5_397B_A17B};
use rig_test_support::cassette_models::OpenAiModels;

use super::super::support::with_doubleword_cassette;
use crate::ecs_matrix::{Wire, agent::run_agent, cells};

fn wire(client: &OpenAiModels) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Doubleword,
        model: client.completion(QWEN3_5_397B_A17B),
        route: Some(client.completion(QWEN3_5_9B)),
        temperature: Some(0.0),
        additional_params: Some(cells::reasoning_off),
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_doubleword_cassette, wire: wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    serving_serial_concurrency_one: ("corpus_matrix/serving_serial_concurrency_one", cells::SERVING_SERIAL_CONCURRENCY_ONE, "doubleword_serving_serial_concurrency_one");
    #[tokio::test]
    output_tool_choice_specific_output: ("corpus_matrix/output_tool_choice_specific_output", cells::OUTPUT_TOOL_CHOICE_SPECIFIC_OUTPUT, "doubleword_output_tool_choice_specific_output");
    #[tokio::test]
    output_tool_under_none_degrades: ("corpus_matrix/output_tool_under_none_degrades", cells::OUTPUT_TOOL_UNDER_NONE_DEGRADES, "doubleword_output_tool_under_none_degrades");
    #[tokio::test]
    shaping_tool_choice_none_on_committed_output: ("corpus_matrix/shaping_tool_choice_none_on_committed_output", cells::SHAPING_TOOL_CHOICE_NONE_ON_COMMITTED_OUTPUT, "doubleword_shaping_tool_choice_none_on_committed_output");
    #[tokio::test]
    shaping_extra_context_streamed: ("corpus_matrix/shaping_extra_context_streamed", cells::SHAPING_EXTRA_CONTEXT_STREAMED, "doubleword_shaping_extra_context_streamed");
    #[tokio::test]
    shaping_max_tokens_second_turn: ("corpus_matrix/shaping_max_tokens_second_turn", cells::SHAPING_MAX_TOKENS_SECOND_TURN, "doubleword_shaping_max_tokens_second_turn");
    #[tokio::test]
    shaping_preamble_second_turn: ("corpus_matrix/shaping_preamble_second_turn", cells::SHAPING_PREAMBLE_SECOND_TURN, "doubleword_shaping_preamble_second_turn");
    #[tokio::test]
    shaping_active_tools_none_second_turn: ("corpus_matrix/shaping_active_tools_none_second_turn", cells::SHAPING_ACTIVE_TOOLS_NONE_SECOND_TURN, "doubleword_shaping_active_tools_none_second_turn");
    #[tokio::test]
    causal_completion_streamed: ("corpus_matrix/causal_completion_streamed", cells::CAUSAL_COMPLETION_STREAMED, "doubleword_causal_completion_streamed");
}

crate::matrix::golden_matrix! {
    wrapper: with_doubleword_cassette, wire: reasoning_wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    output_tool_thinking: ("corpus_matrix/output_tool_thinking", cells::OUTPUT_TOOL_THINKING, "doubleword_output_tool_thinking");
    #[tokio::test]
    reasoning_capped: ("reasoning_matrix/capped", cells::REASONING_CAPPED, "doubleword_reasoning_capped");
}

// Reasoning matrix: the named thinking model, with the shared knob.
fn reasoning_wire(
    client: &OpenAiModels,
) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: cells::ThinkingWire::Doubleword,
        model: client.completion("Qwen/Qwen3.5-397B-A17B-FP8"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}
