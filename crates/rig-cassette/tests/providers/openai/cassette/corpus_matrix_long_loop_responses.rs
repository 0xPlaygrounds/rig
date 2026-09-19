//! The long tool loop's producer column on OpenAiResponses: gpt-4.1-mini (Responses).
//! One recording per live cell; the native twin (`ecs_matrix_long_loop_responses.rs`)
//! reuses each with strict matching. Programs, toolset and assertions are
//! `tests/common/ecs_matrix/long_loop.rs`'s; this file holds the scenario
//! literals and the wire's model.

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::ecs_matrix::{Wire, cells, long_loop};
use rig::completion::CompletionModel;

fn wire(client: &OpenAiCassette) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::OpenAiResponses,
        model: client.openai.completion("gpt-4.1-mini"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_openai_cassette, wire: wire, run: long_loop::run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    long_unary: ("long_loop_matrix_responses/long_unary", long_loop::LONG_UNARY, "openai_responses_long_loop_long_unary");
    #[tokio::test]
    long_streamed: ("long_loop_matrix_responses/long_streamed", long_loop::LONG_STREAMED, "openai_responses_long_loop_long_streamed");
    #[tokio::test]
    parallel_calls: ("long_loop_matrix_responses/parallel_calls", long_loop::PARALLEL_CALLS, "openai_responses_long_loop_parallel_calls");
    #[tokio::test]
    big_result: ("long_loop_matrix_responses/big_result", long_loop::BIG_RESULT, "openai_responses_long_loop_big_result");
    #[tokio::test]
    tool_error_midway: ("long_loop_matrix_responses/tool_error_midway", long_loop::TOOL_ERROR_MIDWAY, "openai_responses_long_loop_tool_error_midway");
    #[tokio::test]
    max_turns_midway: ("long_loop_matrix_responses/max_turns_midway", long_loop::MAX_TURNS_MIDWAY, "openai_responses_long_loop_max_turns_midway");
}

crate::matrix::golden_matrix! {
    wrapper: with_openai_cassette, wire: wire, run: long_loop::run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    output_cap_midway: ("long_loop_matrix_responses/output_cap_midway", long_loop::OUTPUT_CAP_MIDWAY, "openai_responses_long_loop_output_cap_midway");
}
