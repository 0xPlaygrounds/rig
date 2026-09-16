//! The long tool loop's producer column on DeepSeek: deepseek-flash (thinking disabled).
//! One recording per live cell; the native twin (`ecs_matrix_long_loop.rs`)
//! reuses each with strict matching. Programs, toolset and assertions are
//! `tests/common/ecs_matrix/long_loop.rs`'s; this file holds the scenario
//! literals and the wire's model.

use crate::deepseek::support::{BoundDeepSeek, with_deepseek_cassette};
use crate::ecs_matrix::{Wire, cells, long_loop};
use rig::completion::CompletionModel;

fn wire(client: &BoundDeepSeek) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::DeepSeek,
        model: client.completion("deepseek-flash"),
        route: None,
        temperature: Some(0.0),
        additional_params: Some(|| serde_json::json!({"thinking":{"type":"disabled"}})),
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_deepseek_cassette, wire: wire, run: long_loop::run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    long_unary: ("long_loop_matrix/long_unary", long_loop::LONG_UNARY, "deepseek_long_loop_long_unary");
    #[tokio::test]
    long_streamed: ("long_loop_matrix/long_streamed", long_loop::LONG_STREAMED, "deepseek_long_loop_long_streamed");
    #[tokio::test]
    parallel_calls: ("long_loop_matrix/parallel_calls", long_loop::PARALLEL_CALLS, "deepseek_long_loop_parallel_calls");
    #[tokio::test]
    big_result: ("long_loop_matrix/big_result", long_loop::BIG_RESULT, "deepseek_long_loop_big_result");
    #[tokio::test]
    tool_error_midway: ("long_loop_matrix/tool_error_midway", long_loop::TOOL_ERROR_MIDWAY, "deepseek_long_loop_tool_error_midway");
    #[tokio::test]
    max_turns_midway: ("long_loop_matrix/max_turns_midway", long_loop::MAX_TURNS_MIDWAY, "deepseek_long_loop_max_turns_midway");
}

crate::matrix::golden_matrix! {
    wrapper: with_deepseek_cassette, wire: wire, run: long_loop::run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    output_cap_midway: ("long_loop_matrix/output_cap_midway", long_loop::OUTPUT_CAP_MIDWAY, "deepseek_long_loop_output_cap_midway");
}
