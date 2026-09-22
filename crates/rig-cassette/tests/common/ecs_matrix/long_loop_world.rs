//! The long tool loop's world entry: the shared native interpreter over the
//! cell's live recording with strict continuation, a cut (`resume_after`)
//! taken live at the #2514 hold after that many committed tool turns and
//! resumed in a fresh world over the same cassette; the tree leased for the
//! test so the restored world's tools see the head's writes.

#![allow(dead_code, reason = "long-loop cells run on five provider columns")]

use super::{Wire, cells::Cell, long_loop};

use rig_agent::completion::CompletionModel;

use rig_cassette::effect_log::EffectLog;

pub(crate) async fn run_world<M: CompletionModel + Clone + 'static>(
    wire: &Wire<M>,
    cell: &Cell,
    golden: impl FnOnce(&EffectLog),
) -> EffectLog {
    let mut cell = *cell;
    if cell.resume_after == Some(usize::MAX) {
        cell.resume_after = Some(long_loop::recorded_tool_turns(wire.thinking, &cell));
    }
    cell.live_resume = cell.resume_after.is_some();
    let lease = long_loop::lease(&cell).await;
    // `world::run_world` asserts the loop (`long_loop::assert_log`) and the
    // transcript beside the golden, while the world is still open.
    let log = super::world::run_world(wire, &cell, golden).await;
    drop(lease);
    log
}
