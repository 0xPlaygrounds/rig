//! The corpus matrix's producers on the DeepSeek wire (`deepseek-chat`, temperature 0; the route `deepseek-reasoner`): every cell of
//! `tests/common/corpus_matrix/cells.rs` on rig-agent's builder, recorded once
//! and replayed as the golden `deepseek_<cell>`. The driver is
//! `tests/common/corpus_matrix/agent.rs`; this file holds the scenario
//! literals, the wire's models and the wire's `#[ignore]` reasons.
//!
//! Every scenario here is a new recording under `corpus_matrix/`; a cell of
//! the grid missing from this file reuses a recording the corpus already
//! had, whose producer stays where it is.

use crate::corpus_matrix::{Wire, agent::run_agent, cells};
use crate::deepseek::support::with_deepseek_cassette;
use rig_test_support::cassette_models::OpenAiModels;

fn wire(client: &OpenAiModels) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: crate::corpus_matrix::cells::ThinkingWire::DeepSeek,
        model: client.completion("deepseek-chat"),
        route: Some(client.completion("deepseek-reasoner")),
        temperature: Some(0.0),
        additional_params: None,
        options: None,
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_deepseek_cassette, wire: wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    serving_concurrent_concurrency_two: ("corpus_matrix/serving_concurrent_concurrency_two", cells::SERVING_CONCURRENT_CONCURRENCY_TWO, "deepseek_serving_concurrent_concurrency_two");
    #[tokio::test]
    output_tool_choice_specific_output: ("corpus_matrix/output_tool_choice_specific_output", cells::OUTPUT_TOOL_CHOICE_SPECIFIC_OUTPUT, "deepseek_output_tool_choice_specific_output");
    #[tokio::test]
    shaping_tool_choice_none_on_committed_output: ("corpus_matrix/shaping_tool_choice_none_on_committed_output", cells::SHAPING_TOOL_CHOICE_NONE_ON_COMMITTED_OUTPUT, "deepseek_shaping_tool_choice_none_on_committed_output");
    #[tokio::test]
    shaping_active_tools_none_second_turn: ("corpus_matrix/shaping_active_tools_none_second_turn", cells::SHAPING_ACTIVE_TOOLS_NONE_SECOND_TURN, "deepseek_shaping_active_tools_none_second_turn");
}

crate::matrix::golden_matrix! {
    wrapper: with_deepseek_cassette, wire: wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    host_custom_at_start_streamed: ("corpus_matrix/host_custom_at_start_streamed", cells::HOST_CUSTOM_AT_START_STREAMED, "deepseek_host_custom_at_start_streamed");
    #[tokio::test]
    shaping_preamble_second_turn: ("corpus_matrix/shaping_preamble_second_turn", cells::SHAPING_PREAMBLE_SECOND_TURN, "deepseek_shaping_preamble_second_turn");
}

crate::matrix::golden_matrix! {
    wrapper: with_deepseek_cassette, wire: reasoning_wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    reasoning_text_streamed: ("reasoning_matrix/text_streamed", cells::REASONING_TEXT_STREAMED, "deepseek_reasoning_text_streamed");
    #[tokio::test]
    reasoning_tool_streamed: ("reasoning_matrix/tool_streamed", cells::REASONING_TOOL_STREAMED, "deepseek_reasoning_tool_streamed");
    #[tokio::test]
    reasoning_capped: ("reasoning_matrix/capped", cells::REASONING_CAPPED, "deepseek_reasoning_capped");
}

crate::matrix::case_matrix! {
    wrapper: with_deepseek_cassette, family: wire_matrix_case;
    #[ignore = "deepseek-flash rejects enabling thinking after a disabled tool turn with HTTP 400: prior reasoning_content required; record-deepseek-shaping-thinking-second-turn-attempt-1.log and https://api-docs.deepseek.com/guides/thinking_mode/; the model refuses this setting transition"]
    #[tokio::test]
    shaping_thinking_second_turn: ("corpus_matrix/shaping_thinking_second_turn", shaping_thinking_second_turn_9);
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
