//! Requests owned by what made them: an entity of work carries [`RequestOf`]
//! its owner, with its payload, task and result as components. It goes when
//! its owner goes or settles, and when a stop that reaches below reaches its
//! owner agent, idle or not.
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

use bevy_ecs::prelude::*;

use super::stage::Settled;

/// On a request: its owner, which it ends with. One inserted for an owner
/// gone or [`Settled`] is despawned. Moved to a retry with the marker a
/// [`follow_successor`](super::stage::follow_successor) follows.
#[derive(Component, Debug)]
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

/// Despawns a request inserted for an owner that is gone or settled. An
/// observer, so it runs after the relationship's own hook.
pub(crate) fn refuse_ended(
    inserted: On<Insert<RequestOf>>,
    requests: Query<&RequestOf>,
    owners: Query<Has<Settled>>,
    mut commands: Commands,
) {
    let Ok(&RequestOf(owner)) = requests.get(inserted.entity) else {
        return;
    };
    if owners.get(owner).unwrap_or(true) {
        commands.entity(inserted.entity).try_despawn();
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
