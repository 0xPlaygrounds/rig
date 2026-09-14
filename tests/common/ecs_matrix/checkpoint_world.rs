//! Checkpoint rows use the shared native interpreter and strict live-cassette
//! continuation, with a separate cassette session for each selected cut.

#![allow(dead_code, reason = "checkpoint cells run on five provider columns")]

use super::{Wire, cells::Cell};
use rig::{completion::CompletionModel, effect_log::EffectLog};

pub(crate) async fn run_world<M: CompletionModel + Clone + 'static>(
    wire: &Wire<M>,
    cell: &Cell,
    golden: impl FnOnce(&EffectLog),
) -> EffectLog {
    let mut cell = *cell;
    if cell.resume_after == Some(usize::MAX) {
        cell.resume_after = Some(super::checkpoint::tool_turns(&cell));
    }
    cell.live_resume = cell.resume_after.is_some();
    let log = super::world::run_world(wire, &cell, golden).await;
    super::checkpoint::assert_log(&cell, &log);
    log
}

use std::sync::Arc;

use bevy_app::App;
use bevy_ecs::prelude::*;
use rig::effect::{EffectKind, HandlerDescriptor};
use rig::serve::{Dispatch, Reply, Serve, adapters::ToolAdapter};
use rig_ecs::agent::{ToolCallSlot, checkpoint::ToolTurnCommit};
use rig_ecs::bus::{BusSet, EffectOutcome, RigSchedule};
use rig_ecs::systems::RigSet;
use tokio::sync::Semaphore;

/// Host scheduling state, intentionally outside the saved scene. A checkpoint
/// is taken only after all three tools have completed; a restored adapter has
/// fresh unused gates and must never be asked to execute the completed tools.
#[derive(Resource)]
struct ParallelGates {
    permits: [Arc<Semaphore>; 3],
    run: Option<Entity>,
    released: Option<usize>,
    observed: Vec<usize>,
}

impl Default for ParallelGates {
    fn default() -> Self {
        Self {
            permits: std::array::from_fn(|_| Arc::new(Semaphore::new(0))),
            run: None,
            released: None,
            observed: Vec::new(),
        }
    }
}

/// The genuine tool adapter with a host gate before each actual invocation.
/// Descriptors, tool results and error conversion are the ordinary adapter's.
pub(crate) struct GatedBatch {
    inner: ToolAdapter<super::checkpoint::CheckpointBatch>,
    permits: [Arc<Semaphore>; 3],
}

impl Serve for GatedBatch {
    type Family = rig::effect::family::Tool;

    fn descriptor(&self) -> HandlerDescriptor {
        self.inner.descriptor()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let slot = match &kind {
            EffectKind::ToolCall { name, args } => {
                assert_eq!(name, "checkpoint_batch");
                serde_json::from_str::<super::checkpoint::BatchArgs>(args)
                    .expect("recorded batch arguments")
                    .slot
            }
            other => panic!("the batch adapter received {other:?}"),
        };
        self.permits
            .get(slot)
            .expect("the recorded batch slot is 0, 1 or 2")
            .acquire()
            .await
            .expect("the host gate remains open")
            .forget();
        self.inner.serve(kind, dispatch).await
    }
}

/// Install once in each parallel-row world, including the fresh restored
/// world. Only explicitly bound runs can receive permits.
pub(crate) fn install_parallel(app: &mut App) {
    assert!(!app.world().contains_resource::<ParallelGates>());
    app.init_resource::<ParallelGates>().add_systems(
        RigSchedule,
        advance_parallel.after(BusSet::Collect).before(RigSet::Fold),
    );
}

pub(crate) fn parallel_adapter(world: &World) -> GatedBatch {
    GatedBatch {
        inner: ToolAdapter::new(super::checkpoint::CheckpointBatch),
        permits: world.resource::<ParallelGates>().permits.clone(),
    }
}

/// Bind the original run after spawning it. Restoration does not bind a new
/// batch: every tool outcome is already in the committed scene.
pub(crate) fn bind_parallel_run(world: &mut World, run: Entity) {
    let mut gates = world.resource_mut::<ParallelGates>();
    assert!(
        gates.run.replace(run).is_none(),
        "one run owns this fixture's gates"
    );
}

/// Release 2, then 1, then 0. A subsequent permit is issued only after the
/// previous outcome is visible in ECS, so timing cannot collapse the intended
/// partial-batch observations into one simultaneous collection.
fn advance_parallel(world: &mut World) {
    let Some(run) = world.resource::<ParallelGates>().run else {
        return;
    };
    if world.resource::<ParallelGates>().observed.len() == 3 {
        return;
    }
    let mut slots: Vec<_> = world
        .query::<(&ChildOf, &ToolCallSlot, Option<&EffectOutcome>)>()
        .iter(world)
        .filter(|(parent, slot, _)| {
            slot.name == "checkpoint_batch"
                && world
                    .get::<ChildOf>(parent.parent())
                    .is_some_and(|parent| parent.parent() == run)
        })
        .map(|(parent, slot, outcome)| (slot.index, parent.parent(), outcome.is_some()))
        .collect();
    slots.sort_by_key(|(slot, _, _)| *slot);
    let complete = slots.len() == 3 && slots.iter().all(|(_, _, ready)| *ready);
    if !complete {
        assert!(
            !world
                .query::<(&ChildOf, &ToolTurnCommit)>()
                .iter(world)
                .any(|(parent, _)| parent.parent() == run),
            "a partial batch cannot publish a tool-turn commit"
        );
    }
    if slots.is_empty() {
        return;
    }
    assert_eq!(
        slots.len(),
        3,
        "the provider supplied one exact three-call batch"
    );
    assert_eq!(
        slots.iter().map(|(slot, _, _)| *slot).collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert!(
        slots.iter().all(|(_, turn, _)| *turn == slots[0].1),
        "all three calls belong to one turn"
    );
    let mut gates = world.resource_mut::<ParallelGates>();
    // Only the currently released call, and previously observed calls, may
    // have outcomes. This assertion fails even if an unauthorized completion
    // arrives between consecutive schedule passes.
    for (slot, _, ready) in &slots {
        if *ready {
            assert!(
                gates.observed.contains(slot) || gates.released == Some(*slot),
                "slot {slot} completed before its host permit"
            );
        }
    }
    if let Some(slot) = gates.released
        && slots[slot].2
    {
        gates.observed.push(slot);
        gates.released = None;
    }
    assert_eq!(
        gates.observed,
        [2, 1, 0][..gates.observed.len()],
        "actual ECS-visible completion order"
    );
    if gates.released.is_none() && gates.observed.len() < 3 {
        let slot = [2, 1, 0][gates.observed.len()];
        gates.released = Some(slot);
        gates.permits[slot].add_permits(1);
    }
}

/// Called before destroying the original world at a cut, or after its normal
/// settlement. It cannot pass solely because the final model answer matches.
pub(crate) fn assert_parallel_complete(world: &World) {
    let gates = world.resource::<ParallelGates>();
    assert!(gates.run.is_some(), "the original run was explicitly bound");
    assert_eq!(
        gates.observed,
        [2, 1, 0],
        "all gated outcomes were observed in host order"
    );
    assert_eq!(gates.released, None);
    assert!(
        gates
            .permits
            .iter()
            .all(|permit| permit.available_permits() == 0),
        "each issued permit was consumed once"
    );
}
