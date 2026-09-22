//! The long tool loop's producer column on Gemini: gemini-2.5-flash.
//! One recording per live cell; the native twin (`ecs_matrix_long_loop.rs`)
//! reuses each with strict matching. Programs, toolset and assertions are
//! `tests/common/ecs_matrix/long_loop.rs`'s; this file holds the scenario
//! literals and the wire's model.

use super::super::support::with_gemini_cassette;
use crate::ecs_matrix::{Wire, cells, long_loop};
use rig::completion::CompletionModel;
use rig::driver::{Bound, Socket};
use rig::providers::gemini::Gemini;

// gemini-2.5-flash, not flash-lite: at temperature 0 flash-lite answered the
// `list_files` functionResponse with an empty candidate (no parts,
// finishReason STOP) on both endpoints, 3 attempts, first recording round.
fn wire<H: Socket>(client: &Bound<Gemini, H>) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::Gemini,
        model: client.completion("gemini-2.5-flash"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_gemini_cassette, wire: wire, run: long_loop::run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    long_unary: ("long_loop_matrix/long_unary", long_loop::LONG_UNARY, "gemini_long_loop_long_unary");
    #[tokio::test]
    long_streamed: ("long_loop_matrix/long_streamed", long_loop::LONG_STREAMED, "gemini_long_loop_long_streamed");
    #[tokio::test]
    parallel_calls: ("long_loop_matrix/parallel_calls", long_loop::PARALLEL_CALLS, "gemini_long_loop_parallel_calls");
    #[tokio::test]
    big_result: ("long_loop_matrix/big_result", long_loop::BIG_RESULT, "gemini_long_loop_big_result");
    #[tokio::test]
    tool_error_midway: ("long_loop_matrix/tool_error_midway", long_loop::TOOL_ERROR_MIDWAY, "gemini_long_loop_tool_error_midway");
    #[tokio::test]
    max_turns_midway: ("long_loop_matrix/max_turns_midway", long_loop::MAX_TURNS_MIDWAY, "gemini_long_loop_max_turns_midway");
}

crate::matrix::golden_matrix! {
    wrapper: with_gemini_cassette, wire: wire, run: long_loop::run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    output_cap_midway: ("long_loop_matrix/output_cap_midway", long_loop::OUTPUT_CAP_MIDWAY, "gemini_long_loop_output_cap_midway");
}
