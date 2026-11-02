//! Requests owned by what made them: an entity of work a plugin or the
//! kernel starts for an owner, such as a fork
//! ([`ForkAgent`](super::fork::ForkAgent)), a sign-in or a read, carries
//! [`RequestOf`] the owner, and its payload, task and result are components
//! on it. It goes when its owner goes, when its owner settles, and when a
//! stop that reaches below reaches its owner agent, idle or not, so no
//! plugin keeps a map of what is pending.
//!
//! ```
//! use rig_ecs::prelude::*;
//!
//! let mut world = World::new();
//! let agent = world.spawn_empty().id();
//! let request = world.spawn(RequestOf(agent)).id();
//! world.despawn(agent);
//! assert!(world.get_entity(request).is_err());
//! ```

use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::prelude::*;
use bevy_ecs::world::DeferredWorld;

use super::stage::Settled;

/// On a request: its owner, which it ends with. A request whose owner is
/// gone or [`Settled`] when it is inserted is despawned. Moved to a retry
/// with the marker a [`follow_successor`](super::stage::follow_successor)
/// follows.
#[derive(Component, Debug)]
#[component(on_insert = refuse_ended)]
#[relationship(relationship_target = Requests)]
pub struct RequestOf(pub Entity);

/// The requests an entity owns ([`RequestOf`]). Despawning it despawns them.
#[derive(Component, Debug)]
#[relationship_target(relationship = RequestOf, linked_spawn)]
pub struct Requests(Vec<Entity>);

impl Requests {
    /// Despawns each of these requests.
    pub(crate) fn end(&self, commands: &mut Commands) {
        for request in self.iter() {
            commands.entity(request).try_despawn();
        }
    }
}

/// Despawns a request inserted for an owner that is gone or settled.
fn refuse_ended(mut world: DeferredWorld, context: HookContext) {
    let Some(&RequestOf(owner)) = world.get::<RequestOf>(context.entity) else {
        return;
    };
    let ended = world
        .get_entity(owner)
        .map_or(true, |owner| owner.contains::<Settled>());
    if ended {
        world.commands().entity(context.entity).try_despawn();
    }
}

/// Ends the requests of each effect as it settles.
pub(crate) fn end_with_settled(
    settled: On<Add<Settled>>,
    owners: Query<&Requests>,
    mut commands: Commands,
) {
    if let Ok(requests) = owners.get(settled.entity) {
        requests.end(&mut commands);
    }
}

#[cfg(test)]
mod tests;
