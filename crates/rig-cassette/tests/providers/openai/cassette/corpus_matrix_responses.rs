//! The ECS contract matrix's producers on the OpenAI Responses wire (`gpt-5-mini`, no temperature: the gpt-5 family takes only its default; the route `gpt-5-nano`): every cell of
//! `tests/common/ecs_matrix/cells.rs` on rig-agent's builder, recorded once
//! and replayed as the golden `openai_responses_<cell>` the world cells in
//! `ecs_matrix_responses.rs` are compared to. The driver is
//! `tests/common/ecs_matrix/agent.rs`; this file holds the scenario
//! literals, the wire's models and the wire's `#[ignore]` reasons.
//!
//! Every scenario here is a new recording under `corpus_matrix_responses/`; a cell of
//! the grid missing from this file reuses a recording the corpus already
//! had, whose producer stays where it is.

use rig::providers::openai::{GPT_5_MINI, GPT_5_NANO};

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::ecs_matrix::{Wire, agent::run_agent, cells};

fn wire(client: &OpenAiCassette) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::OpenAiResponses,
        model: client.openai.completion(GPT_5_MINI),
        route: Some(client.openai.completion(GPT_5_NANO)),
        temperature: None,
        additional_params: Some(crate::ecs_matrix::cells::openai_responses_stateless),
        options: None,
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_openai_cassette, wire: wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    serving_concurrent_concurrency_one: ("corpus_matrix_responses/serving_concurrent_concurrency_one", cells::SERVING_CONCURRENT_CONCURRENCY_ONE, "openai_responses_serving_concurrent_concurrency_one");
    #[tokio::test]
    memory_failing_append_streamed: ("corpus_matrix_responses/memory_failing_append_streamed", cells::MEMORY_FAILING_APPEND_STREAMED, "openai_responses_memory_failing_append_streamed");
    #[tokio::test]
    output_tool_choice_specific_output: ("corpus_matrix_responses/output_tool_choice_specific_output", cells::OUTPUT_TOOL_CHOICE_SPECIFIC_OUTPUT, "openai_responses_output_tool_choice_specific_output");
    #[tokio::test]
    output_tool_under_none_degrades: ("corpus_matrix_responses/output_tool_under_none_degrades", cells::OUTPUT_TOOL_UNDER_NONE_DEGRADES, "openai_responses_output_tool_under_none_degrades");
    #[tokio::test]
    shaping_preamble_second_turn: ("corpus_matrix_responses/shaping_preamble_second_turn", cells::SHAPING_PREAMBLE_SECOND_TURN, "openai_responses_shaping_preamble_second_turn");
    #[tokio::test]
    shaping_active_tools_none_second_turn: ("corpus_matrix_responses/shaping_active_tools_none_second_turn", cells::SHAPING_ACTIVE_TOOLS_NONE_SECOND_TURN, "openai_responses_shaping_active_tools_none_second_turn");
}

crate::matrix::golden_matrix! {
    wrapper: with_openai_cassette, wire: reasoning_wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    #[ignore = "stale cassette: its request predates item-shaped history, and gpt-5-mini reported zero reasoning tokens in every re-record attempt"]
    reasoning_tool_streamed: ("reasoning_matrix_responses/tool_streamed", cells::REASONING_TOOL_STREAMED, "openai_responses_reasoning_tool_streamed");
    #[tokio::test]
    reasoning_capped: ("reasoning_matrix_responses/capped", cells::REASONING_CAPPED, "openai_responses_reasoning_capped");
}

#[ignore = "the Responses wire's `max_output_tokens` floor is 16; the corpus's second-turn cap is 5"]
#[tokio::test]
async fn shaping_max_tokens_second_turn() {
    with_openai_cassette(
        "corpus_matrix_responses/shaping_max_tokens_second_turn",
        |client| async move {
            run_agent(
                &wire(&client),
                &cells::SHAPING_MAX_TOKENS_SECOND_TURN,
                |_| {},
            )
            .await;
        },
    )
    .await;
}

crate::matrix::case_matrix! {
    wrapper: with_openai_cassette, family: wire_matrix_case;
    #[ignore = "gpt-5-mini minimal returned an encrypted reasoning block on the first turn in all three attempts; record-openai-responses-shaping-thinking-second-turn-attempt-{1,2,3}.log; three attempts exhausted"]
    #[tokio::test]
    shaping_thinking_second_turn: ("corpus_matrix_responses/shaping_thinking_second_turn", shaping_thinking_second_turn_9);
    #[tokio::test]
    #[ignore = "gpt-5-mini minimal returned an encrypted reasoning part despite zero reasoning usage in all three attempts; record-openai-responses-off-attempt-{1,2,3}.log; three attempts exhausted"]
    reasoning_off: ("reasoning_matrix_responses/off", reasoning_off_12);
}

#[ignore = "gpt-5-mini on the Responses wire calls `lookup` with `leaf: true`, so the tool answers without nesting a completion; three recordings agreed"]
#[tokio::test]
async fn causal_completion_serial() {
    with_openai_cassette(
        "corpus_matrix_responses/causal_completion_serial",
        |client| async move {
            run_agent(&wire(&client), &cells::CAUSAL_COMPLETION_SERIAL, |_| {}).await;
        },
    )
    .await;
}

#[ignore = "gpt-5-mini on the Responses wire calls `lookup` with `leaf: true`, so the tool answers without nesting a completion; three recordings agreed"]
#[tokio::test]
async fn causal_completion_concurrent() {
    with_openai_cassette(
        "corpus_matrix_responses/causal_completion_concurrent",
        |client| async move {
            run_agent(&wire(&client), &cells::CAUSAL_COMPLETION_CONCURRENT, |_| {}).await;
        },
    )
    .await;
}

#[ignore = "gpt-5-mini on the Responses wire calls `lookup` with `leaf: true`, so the tool answers without nesting a completion; three recordings agreed"]
#[tokio::test]
async fn causal_completion_streamed() {
    with_openai_cassette(
        "corpus_matrix_responses/causal_completion_streamed",
        |client| async move {
            run_agent(&wire(&client), &cells::CAUSAL_COMPLETION_STREAMED, |_| {}).await;
        },
    )
    .await;
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
    with_openai_cassette(
        "reasoning_matrix_responses/tool_unary",
        |client| async move {
            run_agent(
                &reasoning_wire(&client),
                &cells::REASONING_TOOL_UNARY,
                |_| panic!("unrecorded reasoning scenario: see this test\'s ignore disposition"),
            )
            .await;
        },
    )
    .await;
}
