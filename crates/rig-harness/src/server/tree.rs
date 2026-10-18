//! The agent tree: child sessions, their grants, policies and budgets, and
//! stopping a tree leaves first.

use bevy_ecs::prelude::*;
use rig_protocol::methods::Start;
use rig_protocol::{Grant, Policy, RpcError, SessionId, codes};

/// On a child session: its parent.
#[derive(Component)]
#[relationship(relationship_target = ChildSessions)]
pub struct ParentSession(pub Entity);

/// On a session: its children. Not linked: stopping keeps their history.
#[derive(Component)]
#[relationship_target(relationship = ParentSession)]
pub struct ChildSessions(Vec<Entity>);

/// What a session may use.
#[derive(Component)]
pub struct SessionGrant(pub Grant);

/// Who answers a session's asks.
#[derive(Component)]
pub struct SessionPolicy(pub Policy);

/// On a root session: its tree's budget and what the tree has spent.
#[derive(Component)]
pub struct TreeBudget {
    pub tokens: Option<u64>,
    pub cost_usd: Option<f64>,
    pub spent_tokens: u64,
    pub spent_usd: f64,
}

/// A session's usage, rolling up to its root's [`TreeBudget`].
#[derive(EntityEvent)]
#[entity_event(propagate = &'static ParentSession, auto_propagate)]
pub struct Usage {
    #[event_target]
    pub session: Entity,
    pub tokens: u64,
    pub cost_usd: f64,
}

/// Stops `root` and every descendant, leaves first, and returns the
/// sessions told to stop.
pub fn stop_tree(world: &mut World, root: Entity) -> Vec<SessionId> {
    let _ = (world, root);
    Vec::new()
}

/// The grant a child of `parent` started with `req` runs under, or the
/// refusal (`-32015` wider, `-32019` too deep).
pub fn admit_child(world: &World, parent: Entity, req: &Start) -> Result<Grant, RpcError> {
    let _ = (world, parent, req);
    Err(RpcError::new(
        codes::INTERNAL,
        "child sessions are not built yet",
    ))
}
