//! The corpus matrix's producers on the Venice wire (`mistral-small-3-2-24b-instruct`, the model the suite records tools under, temperature 0; the route is the same model under `golden/model:fast`: every other Venice model tried either thinks (its reasoning part in the history is refused by the Mistral tokenizer on the next turn) or re-calls the tool after its result, so the route is observable on the bus, not on the wire): every cell of
//! `tests/common/corpus_matrix/cells.rs` on rig-agent's builder, recorded once
//! and replayed as the golden `venice_<cell>`. The driver is
//! `tests/common/corpus_matrix/agent.rs`; this file holds the scenario
//! literals, the wire's models and the wire's `#[ignore]` reasons.
//!
//! Every scenario here is a new recording under `corpus_matrix/`; a cell of
//! the grid missing from this file reuses a recording the corpus already
//! had, whose producer stays where it is.

use rig::providers::venice::MISTRAL_SMALL_3_2_24B;
use rig_test_support::cassette_models::OpenAiModels;

use super::super::support::with_venice_cassette;
use crate::corpus_matrix::{Wire, agent::run_agent, cells};

fn wire(client: &OpenAiModels) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: crate::corpus_matrix::cells::ThinkingWire::Venice,
        model: client.completion(MISTRAL_SMALL_3_2_24B),
        route: Some(client.completion(MISTRAL_SMALL_3_2_24B)),
        temperature: Some(0.0),
        additional_params: None,
        options: None,
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_venice_cassette, wire: wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    host_custom_at_start_streamed: ("corpus_matrix/host_custom_at_start_streamed", cells::HOST_CUSTOM_AT_START_STREAMED, "venice_host_custom_at_start_streamed");
    #[tokio::test]
    output_tool_under_none_degrades: ("corpus_matrix/output_tool_under_none_degrades", cells::OUTPUT_TOOL_UNDER_NONE_DEGRADES, "venice_output_tool_under_none_degrades");
    #[tokio::test]
    shaping_max_tokens_second_turn: ("corpus_matrix/shaping_max_tokens_second_turn", cells::SHAPING_MAX_TOKENS_SECOND_TURN, "venice_shaping_max_tokens_second_turn");
    #[tokio::test]
    shaping_preamble_second_turn: ("corpus_matrix/shaping_preamble_second_turn", cells::SHAPING_PREAMBLE_SECOND_TURN, "venice_shaping_preamble_second_turn");
    #[tokio::test]
    #[ignore = "stale cassette: its request predates item-shaped history, and the model skipped the tool call in every re-record attempt"]
    causal_completion_concurrent: ("corpus_matrix/causal_completion_concurrent", cells::CAUSAL_COMPLETION_CONCURRENT, "venice_causal_completion_concurrent");
}

crate::matrix::golden_matrix! {
    wrapper: with_venice_cassette, wire: wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    memory_serial_two_tools: ("corpus_matrix/memory_serial_two_tools", cells::MEMORY_SERIAL_TWO_TOOLS, "venice_memory_serial_two_tools");
    #[tokio::test]
    output_tool_choice_specific_output: ("corpus_matrix/output_tool_choice_specific_output", cells::OUTPUT_TOOL_CHOICE_SPECIFIC_OUTPUT, "venice_output_tool_choice_specific_output");
    #[tokio::test]
    shaping_tool_choice_none_on_committed_output: ("corpus_matrix/shaping_tool_choice_none_on_committed_output", cells::SHAPING_TOOL_CHOICE_NONE_ON_COMMITTED_OUTPUT, "venice_shaping_tool_choice_none_on_committed_output");
    #[tokio::test]
    shaping_active_tools_none_second_turn: ("corpus_matrix/shaping_active_tools_none_second_turn", cells::SHAPING_ACTIVE_TOOLS_NONE_SECOND_TURN, "venice_shaping_active_tools_none_second_turn");
    #[tokio::test]
    #[ignore = "stale cassette: its request predates the rebuilt Chat history, and the model skipped the tool call in all three re-record attempts"]
    causal_completion_serial: ("corpus_matrix/causal_completion_serial", cells::CAUSAL_COMPLETION_SERIAL, "venice_causal_completion_serial");
    #[tokio::test]
    #[ignore = "stale cassette: its request predates the rebuilt Chat history, and the model skipped the tool call in all three re-record attempts"]
    causal_completion_streamed: ("corpus_matrix/causal_completion_streamed", cells::CAUSAL_COMPLETION_STREAMED, "venice_causal_completion_streamed");
}

#[ignore = "Venice's gateway never answers the two-turn output-tool program (a 2,200 s hang, then a 500 `cannot send request`); two recordings agreed"]
#[tokio::test]
async fn output_tool_with_real_tool() {
    with_venice_cassette(
        "corpus_matrix/output_tool_with_real_tool",
        |client| async move {
            run_agent(&wire(&client), &cells::OUTPUT_TOOL_WITH_REAL_TOOL, |_| {}).await;
        },
    )
    .await;
}

crate::matrix::golden_matrix! {
    wrapper: with_venice_cassette, wire: reasoning_wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    output_tool_thinking: ("corpus_matrix/output_tool_thinking", cells::OUTPUT_TOOL_THINKING, "venice_output_tool_thinking");
    #[tokio::test]
    reasoning_text_streamed: ("reasoning_matrix/text_streamed", cells::REASONING_TEXT_STREAMED, "venice_reasoning_text_streamed");
    #[tokio::test]
    reasoning_tool_unary: ("reasoning_matrix/tool_unary", cells::REASONING_TOOL_UNARY, "venice_reasoning_tool_unary");
    #[tokio::test]
    reasoning_tool_streamed: ("reasoning_matrix/tool_streamed", cells::REASONING_TOOL_STREAMED, "venice_reasoning_tool_streamed");
    #[tokio::test]
    reasoning_capped: ("reasoning_matrix/capped", cells::REASONING_CAPPED, "venice_reasoning_capped");
}

crate::matrix::case_matrix! {
    wrapper: with_venice_cassette, family: wire_matrix_case;
    #[ignore = "Venice thinking disabled answered directly without the required first-turn add call in all three attempts; record-venice-shaping-thinking-second-turn-attempt-{1,2,3}.log; three attempts exhausted"]
    #[tokio::test]
    shaping_thinking_second_turn: ("corpus_matrix/shaping_thinking_second_turn", shaping_thinking_second_turn_9);
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
