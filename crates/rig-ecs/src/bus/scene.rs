//! A scene is a checkpoint: the effect entities as data, and the bound
//! descriptors, so a world rebuilt from it re-issues exactly what was not
//! answered.

use bevy_ecs::prelude::*;
use rig_core::{
    effect::{EffectId, EffectKind, HandlerDescriptor, HandlerKey, Outcome},
    error::ErrorReport,
};
use serde::{Deserialize, Serialize};

use super::{
    effect::{EffectOutcome, Held, Issued, PendingEffect, Reserved, Scope, Seq},
    handlers::Bound,
};

/// One effect entity as data: intent, order, id, answer, causality (the
/// index of its parent in the scene), scope, and whether it was held.
/// Never an `Entity`: a scene remaps them by position. In-flight state is
/// not saved — an effect taken but not answered is saved as intent and
/// re-issued under its saved id at load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneEffect {
    /// The saved dispatch order.
    pub seq: Seq,
    /// The handler key.
    pub key: HandlerKey,
    /// The effect.
    pub kind: EffectKind,
    /// The id it was (or, once loaded, will be) dispatched under.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<EffectId>,
    /// The answer, if it had one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<Result<Outcome, ErrorReport>>,
    /// The index in [`Scene::effects`] of the effect it descends from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<usize>,
    /// The entity's own scope, if it carried one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Whether a `Gate` system was holding it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub held: bool,
}

/// The bus half of a scene: what served each key, and every effect entity
/// as data. The crate's own serde form; the `reflect` feature (later)
/// carries a host's other components beside it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Scene {
    /// The bound descriptors, by key.
    pub handlers: Vec<HandlerDescriptor>,
    /// The effect entities, in `Seq` order; parents before children.
    pub effects: Vec<SceneEffect>,
}

impl Scene {
    /// Take the scene of `world`: every entity with a [`PendingEffect`],
    /// in `Seq` order, and every [`Bound`] descriptor.
    pub fn save(world: &mut World) -> Self {
        let mut handlers: Vec<HandlerDescriptor> = world
            .query::<&Bound>()
            .iter(world)
            .map(|bound| bound.descriptor.clone())
            .collect();
        handlers.sort_by(|a, b| a.key.cmp(&b.key));

        let mut rows: Vec<(Entity, Seq)> = world
            .query::<(Entity, &Seq, &PendingEffect)>()
            .iter(world)
            .map(|(entity, seq, _)| (entity, *seq))
            .collect();
        rows.sort_by_key(|(_, seq)| *seq);
        let index_of = |entity: Entity| rows.iter().position(|(e, _)| *e == entity);

        let mut effects = Vec::with_capacity(rows.len());
        for (entity, seq) in &rows {
            let entity_ref = world.entity(*entity);
            let Some(effect) = entity_ref.get::<PendingEffect>().cloned() else {
                continue;
            };
            let parent = entity_ref
                .get::<ChildOf>()
                .and_then(|child_of| index_of(child_of.parent()));
            effects.push(SceneEffect {
                seq: *seq,
                key: effect.key,
                kind: effect.kind,
                id: entity_ref
                    .get::<Issued>()
                    .map(|issued| issued.0)
                    .or_else(|| entity_ref.get::<Reserved>().map(|reserved| reserved.0)),
                outcome: entity_ref
                    .get::<EffectOutcome>()
                    .map(|outcome| outcome.0.clone()),
                parent,
                scope: entity_ref.get::<Scope>().map(|scope| scope.0.clone()),
                held: entity_ref.contains::<Held>(),
            });
        }
        Self { handlers, effects }
    }

    /// Spawn the scene's effects into `world`, in saved order: an answered
    /// effect is spawned answered (never re-dispatched), an unanswered one
    /// as intent with its saved id [`Reserved`], a child `ChildOf` its
    /// parent. Handlers are the host's to bind; the scene's descriptors
    /// say what each key needs. Returns the spawned entities, by scene
    /// index.
    pub fn load(&self, world: &mut World) -> Vec<Entity> {
        let mut spawned: Vec<Entity> = Vec::with_capacity(self.effects.len());
        for effect in &self.effects {
            let mut entity = world.spawn(PendingEffect {
                key: effect.key.clone(),
                kind: effect.kind.clone(),
            });
            if let Some(id) = effect.id {
                match &effect.outcome {
                    Some(outcome) => {
                        entity.insert((Issued(id), EffectOutcome(outcome.clone())));
                    }
                    None => {
                        entity.insert(Reserved(id));
                    }
                }
            } else if let Some(outcome) = &effect.outcome {
                entity.insert(EffectOutcome(outcome.clone()));
            }
            if let Some(scope) = &effect.scope {
                entity.insert(Scope(scope.clone()));
            }
            if effect.held {
                entity.insert(Held);
            }
            spawned.push(entity.id());
        }
        // Parents in a second pass, so a child saved before its parent (a
        // re-parented entity) keeps its causality.
        for (index, effect) in self.effects.iter().enumerate() {
            if let (Some(parent), Some(child)) = (
                effect.parent.and_then(|at| spawned.get(at).copied()),
                spawned.get(index).copied(),
            ) {
                world.entity_mut(child).insert(ChildOf(parent));
            }
        }
        spawned
    }

    /// The first key the scene needs that `handlers` does not describe, or
    /// describes for another family.
    pub fn first_gap(&self, handlers: &[HandlerDescriptor]) -> Option<&HandlerDescriptor> {
        self.handlers.iter().find(|needed| {
            !handlers.iter().any(|have| {
                have.key == needed.key && have.family.family() == needed.family.family()
            })
        })
    }
}
