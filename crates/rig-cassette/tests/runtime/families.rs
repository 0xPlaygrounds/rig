//! The matrix's focused families once per cell: checkpoint cuts, the long
//! tool loop and the long tasks (`tests/common/ecs_matrix/{checkpoint,
//! long_loop,long_tasks}.rs`). A checkpoint or long-loop row runs the
//! family's producer on rig-agent's builder and its world at one cut, each
//! over its own transport of the same replies, and compares the world's
//! whole log, head and tail, with the producer's. A long task is a world
//! program with no producer; it runs once with every assertion its driver
//! makes. The rows rotate the cells across the wires that recorded them.
//!
//! The long tasks' request checks (`long_tasks::assert_requests`) read the
//! recorded requests of the provider's own cassette; they pin what an
//! encoder sends, which the request snapshots and the acceptance index pin
//! for every provider, and they are not repeated here.

use rig::http_client::DynHttpClient;
use rig_test_support::bank;

use super::cells::agree_on_cuts;
use crate::ecs_matrix::corpus;
use crate::ecs_matrix::{
    Wire, cells::Cell, checkpoint, checkpoint_world, long_loop, long_loop_world, long_tasks,
};
use crate::goldens::capture_world_programs;

/// Which family's drivers a row runs.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Family {
    Checkpoint,
    LongLoop,
}

/// The producer, then the world cut after `cut` tool turns (`None` for no
/// cut, `usize::MAX` for after the last), and the two logs compared.
pub(crate) async fn cut<W, T>(
    wire: fn(DynHttpClient) -> Wire<rig::driver::Model<W, T>>,
    replies: Vec<bank::Entry>,
    family: Family,
    cell: &Cell,
    cut: Option<usize>,
) where
    W: rig::wire::Wire<Op = rig::operation::Completion>,
    T: rig::driver::Transport<W>,
{
    let mut cell = *cell;
    // A long loop's last cut is after its last tool turn: every reply but
    // the answer.
    cell.resume_after = match (family, cut) {
        (Family::LongLoop, Some(usize::MAX)) => Some(replies.len() - 1),
        _ => cut,
    };
    let producer = Cell {
        resume_after: None,
        ..cell
    };
    let mut agent = match family {
        Family::Checkpoint => {
            checkpoint::run_agent(&wire(bank::client(&replies)), &producer, |_| {}).await
        }
        Family::LongLoop => {
            long_loop::run_agent(&wire(bank::client(&replies)), &producer, |_| {}).await
        }
    };
    let world_wire = wire(bank::client(&replies));
    let mut world = capture_world_programs(async {
        match family {
            Family::Checkpoint => checkpoint_world::run_world(&world_wire, &cell, |_| {}).await,
            Family::LongLoop => long_loop_world::run_world(&world_wire, &cell, |_| {}).await,
        }
    })
    .await;
    agree_on_cuts(&mut world, &mut agent, cell.name);
    corpus::assert_same_records(&world, &agent, cell.name);
}

/// A long task in a world, saved and restored after `restore` tool turns
/// where given.
pub(crate) async fn task<W, T>(
    wire: fn(DynHttpClient) -> Wire<rig::driver::Model<W, T>>,
    replies: Vec<bank::Entry>,
    cell: &Cell,
    restore: Option<usize>,
) where
    W: rig::wire::Wire<Op = rig::operation::Completion>,
    T: rig::driver::Transport<W>,
{
    let mut cell = *cell;
    cell.resume_after = restore;
    let wire = wire(bank::client(&replies));
    capture_world_programs(long_tasks::run_world(&wire, &cell, |_| {})).await;
}

macro_rules! replies {
    (recorded $provider:literal, $scenario:literal) => {
        bank::recorded($provider, $scenario)
    };
    ($provider:literal, $scenario:literal) => {
        bank::script($provider, $scenario)
    };
}

macro_rules! cuts {
    ($($name:ident: $($pinned:ident)? ($family:ident, $wire:ident, $provider:literal, $scenario:literal, $cell:expr, $cut:expr);)*) => {
        $(
            #[tokio::test]
            async fn $name() {
                let replies = replies!($($pinned)? $provider, $scenario);
                cut(crate::wires::$wire, replies, Family::$family, &$cell, $cut).await;
            }
        )*
    };
}

macro_rules! tasks {
    ($($name:ident: $($pinned:ident)? ($wire:ident, $provider:literal, $scenario:literal, $cell:expr, $restore:expr);)*) => {
        $(
            #[tokio::test]
            async fn $name() {
                let replies = replies!($($pinned)? $provider, $scenario);
                task(crate::wires::$wire, replies, &$cell, $restore).await;
            }
        )*
    };
}

cuts! {
    checkpoint_multi_turn_unary: recorded (Checkpoint, anthropic, "anthropic", "checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, None);
    checkpoint_multi_turn_unary_cut_1: recorded (Checkpoint, anthropic, "anthropic", "checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(1));
    checkpoint_multi_turn_unary_cut_2: recorded (Checkpoint, anthropic, "anthropic", "checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(2));
    checkpoint_multi_turn_unary_cut_3: recorded (Checkpoint, anthropic, "anthropic", "checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(3));
    checkpoint_multi_turn_unary_cut_final: recorded (Checkpoint, anthropic, "anthropic", "checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(usize::MAX));
    checkpoint_multi_turn_streamed: recorded (Checkpoint, openai_responses_mini, "openai", "checkpoint_matrix_responses/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, None);
    checkpoint_multi_turn_streamed_cut_1: recorded (Checkpoint, openai_responses_mini, "openai", "checkpoint_matrix_responses/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(1));
    checkpoint_multi_turn_streamed_cut_2: recorded (Checkpoint, openai_responses_mini, "openai", "checkpoint_matrix_responses/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(2));
    checkpoint_multi_turn_streamed_cut_3: recorded (Checkpoint, openai_responses_mini, "openai", "checkpoint_matrix_responses/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(3));
    checkpoint_multi_turn_streamed_cut_final: recorded (Checkpoint, openai_responses_mini, "openai", "checkpoint_matrix_responses/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(usize::MAX));
    checkpoint_parallel_batch: recorded (Checkpoint, gemini_flash_lite, "gemini", "checkpoint_matrix/parallel_batch", checkpoint::PARALLEL_BATCH, None);
    checkpoint_parallel_batch_cut_final: recorded (Checkpoint, gemini_flash_lite, "gemini", "checkpoint_matrix/parallel_batch", checkpoint::PARALLEL_BATCH, Some(usize::MAX));
    checkpoint_large_result: recorded (Checkpoint, deepseek_flash, "deepseek", "checkpoint_matrix/large_result", checkpoint::LARGE_RESULT, None);
    checkpoint_large_result_cut_final: recorded (Checkpoint, deepseek_flash, "deepseek", "checkpoint_matrix/large_result", checkpoint::LARGE_RESULT, Some(usize::MAX));
    long_unary: recorded (LongLoop, deepseek_flash, "deepseek", "long_loop_matrix/long_unary", long_loop::LONG_UNARY, None);
    long_unary_cut_1: recorded (LongLoop, deepseek_flash, "deepseek", "long_loop_matrix/long_unary", long_loop::LONG_UNARY, Some(1));
    long_unary_cut_2: recorded (LongLoop, deepseek_flash, "deepseek", "long_loop_matrix/long_unary", long_loop::LONG_UNARY, Some(2));
    long_unary_cut_3: recorded (LongLoop, deepseek_flash, "deepseek", "long_loop_matrix/long_unary", long_loop::LONG_UNARY, Some(3));
    long_unary_cut_final: recorded (LongLoop, deepseek_flash, "deepseek", "long_loop_matrix/long_unary", long_loop::LONG_UNARY, Some(usize::MAX));
    long_streamed: recorded (LongLoop, openai_chat_mini, "openai", "long_loop_matrix_chat/long_streamed", long_loop::LONG_STREAMED, long_loop::LONG_STREAMED.resume_after);
    long_parallel_calls: recorded (LongLoop, gemini_flash, "gemini", "long_loop_matrix/parallel_calls", long_loop::PARALLEL_CALLS, long_loop::PARALLEL_CALLS.resume_after);
    long_big_result: recorded (LongLoop, anthropic, "anthropic", "long_loop_matrix/big_result", long_loop::BIG_RESULT, long_loop::BIG_RESULT.resume_after);
    long_tool_error_midway: recorded (LongLoop, openai_responses_mini, "openai", "long_loop_matrix_responses/tool_error_midway", long_loop::TOOL_ERROR_MIDWAY, long_loop::TOOL_ERROR_MIDWAY.resume_after);
    long_max_turns_midway: recorded (LongLoop, deepseek_flash, "deepseek", "long_loop_matrix/max_turns_midway", long_loop::MAX_TURNS_MIDWAY, long_loop::MAX_TURNS_MIDWAY.resume_after);
    long_output_cap_midway: recorded (LongLoop, openai_chat_mini, "openai", "long_loop_matrix_chat/output_cap_midway", long_loop::OUTPUT_CAP_MIDWAY_MAX_TURNS, long_loop::OUTPUT_CAP_MIDWAY_MAX_TURNS.resume_after);
}

tasks! {
    task_repair: recorded (deepseek_flash, "deepseek", "long_task_matrix/repair", long_tasks::REPAIR, None);
    task_repair_streamed: recorded (openai_chat_task, "openai", "long_task_matrix/chat_repair_streamed", long_tasks::REPAIR_STREAMED, None);
    task_reconcile: recorded (gemini_task, "gemini", "long_task_matrix/reconcile", long_tasks::RECONCILE, None);
    task_inventory: recorded (openai_responses_task, "openai", "long_task_matrix/responses_inventory", long_tasks::INVENTORY_WIDE_BATCH, None);
    task_inventory_restore: recorded (openai_responses_task, "openai", "long_task_matrix/responses_inventory", long_tasks::INVENTORY_WIDE_BATCH, Some(3));
}
