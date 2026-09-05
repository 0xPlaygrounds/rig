//! A scene is a checkpoint: the effect entities as data, and the bound
//! descriptors. Safe unanswered intents are re-issued under saved ids;
//! completed answers and streams are restored. A partial unfinished stream
//! cannot resume without a cursor and is refused before entities are spawned.

use bevy_ecs::prelude::*;
use rig_core::{
    effect::{EffectId, EffectKind, HandlerDescriptor, HandlerKey, Outcome},
    error::ErrorReport,
};
use serde::{Deserialize, Serialize};

use super::{
    effect::{
        EffectOutcome, Held, IdCounter, Issued, PendingEffect, Reserved, Scope, Seq, Streamed,
        ToolInputs, ToolOutputs,
    },
    handlers::Bound,
};
use rig_core::tool::ToolContext;

/// One effect entity as data: intent, order, id, answer, causality (the
/// index of its parent in the scene), scope, and whether it was held.
/// Never an `Entity`: a scene remaps them by position. In-flight state is
/// not saved — an effect taken but not answered is saved as intent and
/// re-issued under its saved id at load. Completed streams retain their
/// collected state. Unfinished streams with observed progress cannot resume:
/// no provider cursor exists to prevent their prefix reaching policy twice.
/// Unanswered intents without stream progress restart under their saved id.
/// Submitted `WorldOutcome` inboxes are transient: collect before saving to
/// retain their results, just as for a ready handler task.
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
    /// Collected stream state, including unfinished progress. Required on the
    /// wire: `null` establishes that there was no stream state to preserve.
    /// Omitting it cannot distinguish a safe restart from a lost prefix.
    #[serde(deserialize_with = "Option::deserialize")]
    pub streamed: Option<Streamed>,
    /// The index in [`Scene::effects`] of the effect it descends from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<usize>,
    /// The index, in a sibling scene the host keeps beside this one, of
    /// the entity the effect descends from when that entity is not an
    /// effect of this scene — a host's own entity the effect was spawned
    /// `ChildOf`. Written by the host's resolver at
    /// [`Scene::save_with`], read back by its resolver at
    /// [`Scene::load_with`]; the plain [`Scene::save`] writes none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_ref: Option<usize>,
    /// The entity's own scope, if it carried one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Whether a `Gate` system was holding it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub held: bool,
    /// The context a tool call runs with ([`ToolInputs`]), when it carried one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_inputs: Option<ToolContext>,
    /// What the tool published ([`ToolOutputs`]), when it had answered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_outputs: Option<ToolContext>,
}

/// The bus half of a scene: what served each key, and every effect entity
/// as data. This serde form owns only library effect components. Application
/// components on effect entities remain the host's responsibility;
/// `agent::scene::SceneExtensions` applies to the supported graph entities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Scene {
    /// The bound descriptors, by key.
    pub handlers: Vec<HandlerDescriptor>,
    /// The effect entities, in `Seq` order; parents before children.
    pub effects: Vec<SceneEffect>,
    /// Additional allocator history not derivable from surviving effect IDs.
    /// Absent when the saved IDs already determine the next unused ID. MAX
    /// preserves exhaustion; it is never allocated to a fresh effect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_id: Option<u64>,
}

impl Scene {
    /// Take the scene of `world`: every entity with a [`PendingEffect`],
    /// in `Seq` order, and every [`Bound`] descriptor. An effect `ChildOf`
    /// an entity that is not an effect keeps no parent; see
    /// [`Scene::save_with`].
    pub fn save(world: &mut World) -> Self {
        Self::save_with(world, |_| None)
    }

    /// [`Scene::save`], with `sibling` naming — by an index into a scene the
    /// host saves beside this one — every parent entity that is not an
    /// effect of this scene, so a host's own entity an effect descends from
    /// is kept as [`SceneEffect::parent_ref`] and restored by
    /// [`Scene::load_with`].
    pub fn save_with(world: &mut World, sibling: impl Fn(Entity) -> Option<usize>) -> Self {
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
            let parent_entity = entity_ref.get::<ChildOf>().map(ChildOf::parent);
            let parent = parent_entity.and_then(index_of);
            let parent_ref = match parent {
                Some(_) => None,
                None => parent_entity.and_then(&sibling),
            };
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
                streamed: entity_ref.get::<Streamed>().cloned(),
                parent,
                parent_ref,
                scope: entity_ref.get::<Scope>().map(|scope| scope.0.clone()),
                held: entity_ref.contains::<Held>(),
                tool_inputs: entity_ref
                    .get::<ToolInputs>()
                    .map(|inputs| inputs.0.clone()),
                tool_outputs: entity_ref
                    .get::<ToolOutputs>()
                    .map(|outputs| outputs.0.clone()),
            });
        }
        let derived = effects
            .iter()
            .filter_map(|effect| effect.id)
            .map(|id| id.as_u64().saturating_add(1))
            .max()
            .unwrap_or(0);
        let counter = world
            .get_resource::<IdCounter>()
            .map_or(0, |counter| counter.0);
        Self {
            handlers,
            effects,
            next_id: (counter > derived).then_some(counter),
        }
    }

    /// Spawn the scene's effects into `world`, in saved order: an answered
    /// effect is spawned answered (never re-dispatched), an unanswered one
    /// as intent with its saved id [`Reserved`], a child `ChildOf` its
    /// parent. Handlers are the host's to bind; the scene's descriptors
    /// say what each key needs. Returns the spawned entities, by scene
    /// index. A [`SceneEffect::parent_ref`] is not resolved here; see
    /// [`Scene::load_with`].
    ///
    /// Refuses unfinished streams with delivered progress before spawning
    /// anything. Load invokes insertion observers and change detection;
    /// install application observers afterward or guard rehydration explicitly.
    pub fn load(&self, world: &mut World) -> Result<Vec<Entity>, ErrorReport> {
        self.load_with(world, |_| None)
    }

    /// [`Scene::load`], with `sibling` resolving every
    /// [`SceneEffect::parent_ref`] to the entity the host loaded from its
    /// own scene, so the effect is `ChildOf` it again.
    pub fn load_with(
        &self,
        world: &mut World,
        sibling: impl Fn(usize) -> Option<Entity>,
    ) -> Result<Vec<Entity>, ErrorReport> {
        self.validate_resume()?;
        let next_id = self.id_floor()?;
        world.init_resource::<IdCounter>();
        let mut ids = world.resource_mut::<IdCounter>();
        ids.0 = ids.0.max(next_id);
        let mut spawned: Vec<Entity> = Vec::with_capacity(self.effects.len());
        for effect in &self.effects {
            let mut entity = world.spawn(PendingEffect {
                key: effect.key.clone(),
                kind: effect.kind.clone(),
            });
            // Restore stream data before the answer so answer observers can
            // inspect the complete snapshot. Unanswered intents always restart.
            if effect.outcome.is_some()
                && let Some(streamed) = &effect.streamed
            {
                entity.insert(streamed.clone());
            }
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
            if let Some(inputs) = &effect.tool_inputs {
                entity.insert(ToolInputs(inputs.clone()));
            }
            if let Some(outputs) = &effect.tool_outputs {
                entity.insert(ToolOutputs(outputs.clone()));
            }
            spawned.push(entity.id());
        }
        // Parents in a second pass, so a child saved before its parent (a
        // re-parented entity) keeps its causality.
        for (index, effect) in self.effects.iter().enumerate() {
            let Some(child) = spawned.get(index).copied() else {
                continue;
            };
            let parent = effect
                .parent
                .and_then(|at| spawned.get(at).copied())
                .or_else(|| effect.parent_ref.and_then(&sibling));
            if let Some(parent) = parent {
                world.entity_mut(child).insert(ChildOf(parent));
            }
        }
        Ok(spawned)
    }

    /// Validate the allocator and stream cut before spawning anything. A stream that has
    /// delivered a prefix but has not closed needs a provider cursor or host
    /// reconciliation, neither of which a generic scene can supply. Save
    /// before stream progress or after completion for automatic restoration.
    pub fn validate_resume(&self) -> Result<(), ErrorReport> {
        self.id_floor()?;
        for effect in &self.effects {
            if effect.outcome.is_none()
                && effect.streamed.as_ref().is_some_and(|streamed| {
                    !streamed.events.is_empty()
                        || !streamed.text.is_empty()
                        || streamed.outcome.is_some()
                })
            {
                return Err(ErrorReport::new(
                    rig_core::error::ErrorKind::Request,
                    format!(
                        "scene resume refused: unfinished stream for `{}` ({:?}) already delivered progress; no restart cursor is recorded",
                        effect.key, effect.id
                    ),
                ));
            }
        }
        Ok(())
    }

    fn id_floor(&self) -> Result<u64, ErrorReport> {
        let derived = self
            .effects
            .iter()
            .filter_map(|effect| effect.id)
            .try_fold(0, |floor: u64, id| {
                id.as_u64()
                    .checked_add(1)
                    .map(|next| floor.max(next))
                    .ok_or_else(|| {
                        ErrorReport::new(
                            rig_core::error::ErrorKind::Request,
                            "scene contains an effect ID beyond the allocator range",
                        )
                    })
            })?;
        if self.next_id.is_some_and(|next| next < derived) {
            return Err(ErrorReport::new(
                rig_core::error::ErrorKind::Request,
                "scene next_id contradicts its saved effect IDs",
            ));
        }
        Ok(self.next_id.unwrap_or(derived))
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
