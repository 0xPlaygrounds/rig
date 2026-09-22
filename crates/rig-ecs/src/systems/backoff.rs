//! Provider-retry delays driven by the world's Bevy clock. Pausing virtual time
//! holds retries until the host advances it.
//!
//! ```
//! use std::time::Duration;
//! use rig_ecs::{agent::Backoff, systems::backoff::delay};
//! let backoff = Backoff { base: Duration::from_secs(1), max: Duration::from_secs(8) };
//! assert_eq!(delay(&backoff, 2), Duration::from_secs(2));
//! ```

use std::time::Duration;

use bevy_ecs::prelude::*;
use bevy_reflect::Reflect;
use bevy_time::DelayedCommandsExt;
use rig_core::observe::Emitter;
use serde::{Deserialize, Serialize};

use crate::{
    agent::{Backoff, RunOf},
    bus::{PendingEffect, acquire_hold, release_hold},
};

/// The owner name of a backoff's hold.
pub const BACKOFF_OWNER: &str = "rig-ecs/backoff";

/// The turn re-issues a completion a provider retry lost: which attempt
/// (1 for the first retry). Stamped by `advance` on the retry's turn.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct RetryAttempt(pub usize);

/// The delay before attempt `attempt` (1-based): `base × 2^(attempt-1)`,
/// capped at `max`.
pub fn delay(backoff: &Backoff, attempt: usize) -> Duration {
    let factor = 2u32.saturating_pow(attempt.saturating_sub(1).min(31) as u32);
    backoff.base.saturating_mul(factor).min(backoff.max)
}

/// After `RigSet::Patch`, before `RigSet::Release`: the completion effect of a retry turn under a
/// [`Backoff`] (the run's, else the agent's) is held as `rig-ecs/backoff`
/// the pass it is folded, and released by a delayed command when the delay
/// has elapsed on the world's clock.
pub fn hold_retries(
    mut commands: Commands,
    effects: Query<(Entity, &ChildOf), Added<PendingEffect>>,
    turns: Query<(&ChildOf, &RetryAttempt)>,
    runs: Query<&RunOf>,
    backoffs: Query<&Backoff>,
) {
    for (effect, turn_of) in &effects {
        let Ok((run_of, RetryAttempt(attempt))) = turns.get(turn_of.parent()) else {
            continue;
        };
        let run = run_of.parent();
        let Ok(RunOf(agent)) = runs.get(run) else {
            continue;
        };
        let Some(backoff) = backoffs.get(run).ok().or_else(|| backoffs.get(*agent).ok()) else {
            continue;
        };
        let wait = delay(backoff, *attempt);
        commands.queue(move |world: &mut World| {
            // A removed or issued effect no longer needs this retry barrier.
            let _ = acquire_hold(world, effect, Emitter::named(BACKOFF_OWNER));
        });
        commands
            .delayed()
            .duration(wait)
            .queue(move |world: &mut World| {
                let _ = release_hold(world, effect, BACKOFF_OWNER);
            });
    }
}
