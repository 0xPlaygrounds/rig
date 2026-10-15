//! Waking the loop that sleeps while nothing happens. Bevy's clock, which
//! timers and delayed commands run on, stops while the machine sleeps; a
//! [`WakeAt`] is a wall-clock instant instead. Once it passes, the entity
//! gets [`Due`] in `First`, and the headless loop sleeps no longer than
//! until the earliest one. A plugin reacts to `Add<Due>`, and arms the next
//! time by inserting a new `WakeAt`, which takes `Due` away. A
//! [`KeepAwake`] keeps the loop running at its pace while it lives.
//!
//! ```
//! use std::time::SystemTime;
//!
//! use rig_harness::prelude::*;
//!
//! let mut app = App::new();
//! app.add_plugins(rig_harness::clock::ClockPlugin);
//! let alarm = app.world_mut().spawn(WakeAt(SystemTime::UNIX_EPOCH)).id();
//! app.update();
//! assert!(app.world().entity(alarm).contains::<Due>());
//! ```

use std::time::{Duration, SystemTime};

use bevy_app::prelude::*;
use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::prelude::*;
use bevy_ecs::world::DeferredWorld;
use bevy_reflect::prelude::*;

/// When this entity is due, on the wall clock.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
#[component(on_insert = rearm)]
pub struct WakeAt(pub SystemTime);

impl WakeAt {
    /// A wake `after` from now.
    pub fn after(after: Duration) -> Self {
        Self(SystemTime::now() + after)
    }

    /// How long until it is due; zero once it is.
    pub fn left(&self, now: SystemTime) -> Duration {
        self.0.duration_since(now).unwrap_or_default()
    }
}

/// On an entity whose [`WakeAt`] passed, from the `First` of the frame
/// after it did.
#[derive(Component, Clone, Copy, Debug, Default)]
#[component(storage = "SparseSet")]
pub struct Due;

/// Keeps the loop that sleeps between frames running a frame at least
/// this often while the entity exists: a spinner, or a panel that polls.
/// Time itself is Bevy's: a system paced with
/// `.run_if(on_real_timer(interval))` runs on time while a `KeepAwake` of
/// that interval lives, and a one-off wait is a delayed command
/// (`commands.delayed().duration(wait)`), whose deadline the loop wakes for
/// on its own.
///
/// ```
/// use rig_harness::prelude::*;
///
/// const STEP: Duration = Duration::from_millis(200);
///
/// fn tick() {}
///
/// let mut app = App::new();
/// // `tick` runs every step while this entity lives; once it is despawned
/// // the loop sleeps until there is work.
/// app.world_mut().spawn((Name::new("spinner"), KeepAwake(STEP)));
/// app.add_systems(Update, tick.run_if(on_real_timer(STEP)));
/// ```
#[derive(Component, Reflect, Clone, Copy, Debug)]
#[reflect(Component, Clone, Debug)]
pub struct KeepAwake(pub Duration);

/// Marks what is [`Due`].
pub struct ClockPlugin;

impl Plugin for ClockPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(First, mark_due);
    }
}

/// A new [`WakeAt`] is not due yet.
fn rearm(mut world: DeferredWorld, context: HookContext) {
    world.commands().entity(context.entity).try_remove::<Due>();
}

fn mark_due(waits: Query<(Entity, &WakeAt), Without<Due>>, mut commands: Commands) {
    let now = SystemTime::now();
    for (entity, wake) in &waits {
        if wake.0 <= now {
            commands.entity(entity).try_insert(Due);
        }
    }
}

#[cfg(test)]
mod tests;
