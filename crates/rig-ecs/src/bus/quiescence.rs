//! Whether a pass of [`RigSchedule`](super::plugin::RigSchedule) advanced the
//! world.
//!
//! The runner loops the schedule while a pass still moves something. That
//! question used to be a `Progress(bool)` resource that every system took as
//! `ResMut` and set directly — ten systems, forty-five call sites — which
//! made every one of them conflict with every other, so the schedule was
//! serial whatever executor it ran under and its ambiguity detection had
//! nothing left to find.
//!
//! The answer is the same; only the way it is written changed. A system says
//! it advanced the world with [`AdvancedCommands::advanced`] —
//! `commands.advanced()` — which **queues** the write through `Commands`
//! instead of declaring it. `Commands` conflicts with nothing, so saying it
//! costs a system no access at all, and a host system can say it too.
//! Commands apply at the pass's sync points, and exclusive systems are sync
//! points, so a mid-pass reader sees what the systems before it said.
//!
//! ## Why the condition is not derived from the world
//!
//! An effect gaining an [`EffectOutcome`](super::EffectOutcome), or leaving
//! [`InFlight`](super::InFlight), *is* an advance, and a read-only predicate
//! over `Added`/`RemovedComponents` would need no cooperation from the system
//! that caused it. That was built first, and it is not equivalent:
//!
//! - `Added<T>` is relative to the reader's previous run, so on the first
//!   read of a world — or of one a scene has just loaded — every component
//!   already present reads as added. A world's existing state and this
//!   pass's work are indistinguishable to change detection, and a tick that
//!   should be one pass becomes two. `bus::collect::tests`'
//!   `ready_delivery_is_bounded_and_deltas_do_not_spin_quiescence` catches
//!   exactly this.
//! - Several advances have no component boundary at all: the stream fold
//!   landing in `Streamed::outcome`, a replay batch of cancelled steps that
//!   moves only its own cursor, the first `ReplayFailure`. Those need
//!   per-transition markers the runtime does not have.
//!
//! Deriving the condition is worth doing once those markers exist. Until
//! then this is exact, and it already buys the parallelism: the flag is
//! written, but nothing declares that it writes it.

use bevy_ecs::prelude::*;

/// Whether this pass advanced the world. Reset by the runner before each
/// pass and read after it.
///
/// Never taken as `ResMut` by a system: [`AdvancedCommands::advanced`] is
/// how it is written, and that queues the write.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct Advanced(pub bool);

/// Say that the world advanced, so the schedule should run another pass.
///
/// Implemented for `Commands` — the form a system uses, which declares no
/// access and so conflicts with nothing — and for `World`, for the exclusive
/// helpers the runtime runs outside a system.
pub trait AdvancedCommands {
    /// Say that this pass advanced the world.
    fn advanced(&mut self);
}

impl AdvancedCommands for Commands<'_, '_> {
    fn advanced(&mut self) {
        self.queue(|world: &mut World| {
            world.advanced();
        });
    }
}

impl AdvancedCommands for World {
    fn advanced(&mut self) {
        if let Some(mut advanced) = self.get_resource_mut::<Advanced>() {
            advanced.0 = true;
        }
    }
}

/// Whether anything has said it advanced since the runner last reset the
/// answer: the whole pass, when the runner asks after it, and the part of
/// the pass that already ran, when an exclusive system asks during it.
pub(super) fn advanced(world: &World) -> bool {
    world
        .get_resource::<Advanced>()
        .is_some_and(|advanced| advanced.0)
}

/// Start a pass with nothing said.
pub(super) fn reset(world: &mut World) {
    if let Some(mut advanced) = world.get_resource_mut::<Advanced>() {
        advanced.0 = false;
    }
}
