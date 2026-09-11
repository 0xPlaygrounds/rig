//! Independent owners of a shared pre-dispatch barrier.

use std::collections::{BTreeMap, VecDeque};

use bevy_ecs::prelude::*;
use rig_core::observe::Emitter;
use serde::{Deserialize, Serialize};

use super::{EffectOutcome, Held, Issued};

/// Active hold owners, keyed by stable emitter name. Distinct policies must
/// use distinct names; reacquiring the same name is idempotent.
/// Mutate through [`acquire_hold`] and [`release_hold`], not by replacing Held.
#[derive(Component, Debug, Clone, Serialize, Deserialize)]
pub struct HoldOwners(BTreeMap<String, Emitter>);

impl HoldOwners {
    /// The owners currently preventing dispatch.
    pub fn owners(&self) -> impl Iterator<Item = &Emitter> {
        self.0.values()
    }
}

/// An ownership transition published after the corresponding hold mutation.
#[derive(EntityEvent)]
pub struct HoldTransition {
    /// The held call.
    pub entity: Entity,
    /// The policy or batch owner whose state changed.
    pub owner: Emitter,
    /// Whether this owner acquired rather than released its hold.
    pub acquired: bool,
}

// Component observers can synchronously call the ownership API again. Queue
// facts before mutation and publish after the outer mutation, preserving order.
#[derive(Resource, Default)]
struct Transitions {
    depth: usize,
    publishing: bool,
    pending: VecDeque<HoldTransition>,
}

fn begin(world: &mut World, event: HoldTransition) {
    let mut transitions = world.get_resource_or_init::<Transitions>();
    transitions.depth += 1;
    transitions.pending.push_back(event);
}

fn finish(world: &mut World) {
    let mut transitions = world.resource_mut::<Transitions>();
    transitions.depth -= 1;
    if transitions.depth != 0 || transitions.publishing {
        return;
    }
    transitions.publishing = true;
    loop {
        let event = world.resource_mut::<Transitions>().pending.pop_front();
        let Some(event) = event else { break };
        // Despawn is terminal cleanup, not a release; attribution can no
        // longer be resolved once lifecycle observers removed the entity.
        if world.get_entity(event.entity).is_ok() {
            world.trigger(event);
        }
    }
    world.resource_mut::<Transitions>().publishing = false;
}

/// Acquire one owner's hold before dispatch. Returns whether it was new.
/// A pre-existing bare Held is retained as an independent unknown owner.
pub fn acquire_hold(world: &mut World, entity: Entity, owner: Emitter) -> bool {
    let Ok(effect) = world.get_entity(entity) else {
        return false;
    };
    if effect.contains::<EffectOutcome>() || effect.contains::<Issued>() {
        return false;
    }
    let mut owners = effect.get::<HoldOwners>().cloned().unwrap_or_else(|| {
        let mut owners = BTreeMap::new();
        if effect.contains::<Held>() {
            let unknown = Emitter::unknown();
            owners.insert(unknown.name.clone(), unknown);
        }
        HoldOwners(owners)
    });
    if owners.0.contains_key(&owner.name) {
        return false;
    }
    owners.0.insert(owner.name.clone(), owner.clone());
    begin(
        world,
        HoldTransition {
            entity,
            owner,
            acquired: true,
        },
    );
    world.entity_mut(entity).insert(owners);
    if world
        .get::<HoldOwners>(entity)
        .is_some_and(|owners| !owners.0.is_empty())
    {
        world.entity_mut(entity).insert(Held);
    }
    finish(world);
    true
}

/// Release only the named owner's hold. The barrier remains while any other
/// owner holds it. Returns false when that owner did not hold the effect.
/// Denial/despawn cleanup is not an ordinary release observation.
pub fn release_hold(world: &mut World, entity: Entity, owner: &str) -> bool {
    let mut owners = match world.get::<HoldOwners>(entity).cloned() {
        Some(owners) => owners,
        None if world.get::<Held>(entity).is_some() && owner == Emitter::unknown().name => {
            let unknown = Emitter::unknown();
            HoldOwners(BTreeMap::from([(unknown.name.clone(), unknown)]))
        }
        None => return false,
    };
    let Some(owner) = owners.0.remove(owner) else {
        return false;
    };
    // Keep the ownership component present during Held removal: the legacy
    // aggregate observer must not duplicate this explicit owner transition.
    begin(
        world,
        HoldTransition {
            entity,
            owner,
            acquired: false,
        },
    );
    world.entity_mut(entity).insert(owners);
    if world
        .get::<HoldOwners>(entity)
        .is_some_and(|owners| owners.0.is_empty())
    {
        world.entity_mut(entity).remove::<Held>();
        // Removal observers may acquire a new owner. Never erase that state.
        if world
            .get::<HoldOwners>(entity)
            .is_some_and(|owners| owners.0.is_empty())
        {
            world.entity_mut(entity).remove::<HoldOwners>();
        }
    }
    finish(world);
    true
}
