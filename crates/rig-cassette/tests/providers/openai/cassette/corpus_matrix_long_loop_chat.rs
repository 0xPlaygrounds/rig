//! The long tool loop's producer column on OpenAiChat: gpt-4.1-mini (Chat Completions).
//! One recording per live cell; the native twin (`ecs_matrix_long_loop_chat.rs`)
//! reuses each with strict matching. Programs, toolset and assertions are
//! `tests/common/ecs_matrix/long_loop.rs`'s; this file holds the scenario
//! literals and the wire's model.

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::ecs_matrix::{Wire, cells, long_loop};
use rig::completion::CompletionModel;

fn wire(client: &OpenAiCassette) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::OpenAiChat,
        model: client.openai.chat("gpt-4.1-mini"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_openai_cassette, wire: wire, run: long_loop::run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    long_unary: ("long_loop_matrix_chat/long_unary", long_loop::LONG_UNARY, "openai_chat_long_loop_long_unary");
    #[tokio::test]
    long_streamed: ("long_loop_matrix_chat/long_streamed", long_loop::LONG_STREAMED, "openai_chat_long_loop_long_streamed");
    #[tokio::test]
    parallel_calls: ("long_loop_matrix_chat/parallel_calls", long_loop::PARALLEL_CALLS, "openai_chat_long_loop_parallel_calls");
    #[tokio::test]
    big_result: ("long_loop_matrix_chat/big_result", long_loop::BIG_RESULT, "openai_chat_long_loop_big_result");
    #[tokio::test]
    tool_error_midway: ("long_loop_matrix_chat/tool_error_midway", long_loop::TOOL_ERROR_MIDWAY, "openai_chat_long_loop_tool_error_midway");
    #[tokio::test]
    max_turns_midway: ("long_loop_matrix_chat/max_turns_midway", long_loop::MAX_TURNS_MIDWAY, "openai_chat_long_loop_max_turns_midway");
}

crate::matrix::golden_matrix! {
    wrapper: with_openai_cassette, wire: wire, run: long_loop::run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    output_cap_midway: ("long_loop_matrix_chat/output_cap_midway", long_loop::OUTPUT_CAP_MIDWAY, "openai_chat_long_loop_output_cap_midway");
}
