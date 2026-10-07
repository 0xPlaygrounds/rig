//! The matrix's focused families once per cell: checkpoint programs and the
//! long tool loop (`tests/common/corpus_matrix/{checkpoint,long_loop}.rs`).
//! A row runs the family's producer on rig-agent's builder over the bank's
//! replies, with every assertion the producer makes. The rows rotate the
//! cells across the wires that recorded them.

use rig::http_client::DynHttpClient;
use rig_test_support::bank;

use crate::corpus_matrix::{Wire, cells::Cell, checkpoint, long_loop};

/// Which family's driver a row runs.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Family {
    Checkpoint,
    LongLoop,
}

/// The family's producer over `replies`.
pub(crate) async fn produce<W, T>(
    wire: fn(DynHttpClient) -> Wire<rig::driver::Model<W, T>>,
    replies: Vec<bank::Entry>,
    family: Family,
    cell: &Cell,
) where
    W: rig::wire::Wire<Op = rig::operation::Completion>,
    T: rig::driver::Transport<W>,
{
    match family {
        Family::Checkpoint => {
            checkpoint::run_agent(&wire(bank::client(&replies)), cell, |_| {}).await;
        }
        Family::LongLoop => {
            long_loop::run_agent(&wire(bank::client(&replies)), cell, |_| {}).await;
        }
    }
}

macro_rules! replies {
    (recorded $provider:literal, $scenario:literal) => {
        bank::recorded($provider, $scenario)
    };
    ($provider:literal, $scenario:literal) => {
        bank::script($provider, $scenario)
    };
}

macro_rules! rows {
    ($($name:ident: $($pinned:ident)? ($family:ident, $wire:ident, $provider:literal, $scenario:literal, $cell:expr);)*) => {
        $(
            #[tokio::test]
            async fn $name() {
                let replies = replies!($($pinned)? $provider, $scenario);
                produce(crate::wires::$wire, replies, Family::$family, &$cell).await;
            }
        )*
    };
}

rows! {
    checkpoint_multi_turn_unary: recorded (Checkpoint, anthropic, "anthropic", "checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY);
    checkpoint_multi_turn_streamed: recorded (Checkpoint, openai_responses_mini, "openai", "checkpoint_matrix_responses/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED);
    checkpoint_parallel_batch: recorded (Checkpoint, gemini_flash_lite, "gemini", "checkpoint_matrix/parallel_batch", checkpoint::PARALLEL_BATCH);
    checkpoint_large_result: recorded (Checkpoint, deepseek_flash, "deepseek", "checkpoint_matrix/large_result", checkpoint::LARGE_RESULT);
    long_unary: recorded (LongLoop, deepseek_flash, "deepseek", "long_loop_matrix/long_unary", long_loop::LONG_UNARY);
    long_streamed: recorded (LongLoop, openai_chat_mini, "openai", "long_loop_matrix_chat/long_streamed", long_loop::LONG_STREAMED);
    long_parallel_calls: recorded (LongLoop, gemini_flash, "gemini", "long_loop_matrix/parallel_calls", long_loop::PARALLEL_CALLS);
    long_big_result: recorded (LongLoop, anthropic, "anthropic", "long_loop_matrix/big_result", long_loop::BIG_RESULT);
    long_tool_error_midway: recorded (LongLoop, openai_responses_mini, "openai", "long_loop_matrix_responses/tool_error_midway", long_loop::TOOL_ERROR_MIDWAY);
    long_max_turns_midway: recorded (LongLoop, deepseek_flash, "deepseek", "long_loop_matrix/max_turns_midway", long_loop::MAX_TURNS_MIDWAY);
    long_output_cap_midway: recorded (LongLoop, openai_chat_mini, "openai", "long_loop_matrix_chat/output_cap_midway", long_loop::OUTPUT_CAP_MIDWAY_MAX_TURNS);
}
