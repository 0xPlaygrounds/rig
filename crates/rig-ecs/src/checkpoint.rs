//! A checkpoint is the world as reflected data: every entity with a
//! registered reflected component, each component under its type path, an
//! `Entity` in a component as the index of that entity in the checkpoint.
//! [`save_world`] takes it; [`load_world`] spawns it into a world — the
//! host's handlers kept where the checkpoint's keys meet them, in-flight
//! effects re-issued under their saved ids, the binary store merged — after
//! validating the whole of it in a scratch world, so a refused checkpoint
//! leaves the destination untouched.
//!
//! What is saved is what is registered with the world's [`AppTypeRegistry`]
//! as a component or a resource: the crate registers its own
//! ([`register_types`]); a host registers its own types the same way and
//! they travel with the entities they sit on. In-flight state (`Serving`,
//! `Streaming`, `Handler`, the cache views) is not reflected and so never
//! saved; relationship targets (`Children`, `Serves`, …) are rebuilt by the
//! hooks of their sources at load, in checkpoint order — which is the
//! world's `Children` order, parents before children.

use std::{any::TypeId, collections::HashMap};

use bevy_ecs::{
    entity_disabling::Disabled,
    prelude::*,
    reflect::{AppTypeRegistry, ReflectComponent},
    resource::IsResource,
};
use bevy_reflect::{
    PartialReflect, TypeRegistration, TypeRegistry,
    serde::{
        ReflectDeserializerProcessor, ReflectSerializer, ReflectSerializerProcessor,
        TypedReflectDeserializer,
    },
};
use rig_core::error::{ErrorKind, ErrorReport};
use serde::{Deserialize, Serialize, de::DeserializeSeed};

use crate::{
    agent::{
        self, Utterance,
        content::{binary::BinaryAssets, parts::*},
    },
    bus::{
        self, Bound, EffectOutcome, HandlerIndex, IdCounter, InFlight, Issued, PendingEffect,
        Reserved, Seq, SeqCounter, Streamed,
    },
};

/// One entity of a [`Checkpoint`]: its components by type path.
pub type CheckpointEntity = serde_json::Map<String, serde_json::Value>;

/// An entity's reflected components, by type path.
type Reflected<'a> = Vec<(&'a str, Box<dyn PartialReflect>)>;

/// The world as reflected data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Every entity with a reflected component, parents before children,
    /// siblings in `Children` order.
    pub entities: Vec<CheckpointEntity>,
    /// The world's counters: the next run number (the id and sequence
    /// counters bump themselves from what is loaded).
    #[serde(default)]
    pub counters: Counters,
    /// The binary store: every retained payload, once per content hash.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub binaries: Vec<crate::agent::content::binary::BinaryRecord>,
}

/// The world's counters, so nothing loaded collides with what comes after.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Counters {
    /// [`agent::RunCounter`].
    pub next_run: u64,
    /// [`IdCounter`].
    pub next_id: u64,
}

/// What [`load_world`] spawned, by checkpoint index; an index the load
/// merged into an existing handler entity names that entity.
#[derive(Debug, Clone, Default)]
pub struct Loaded {
    /// The entities, by [`Checkpoint::entities`] index.
    pub entities: Vec<Entity>,
}

impl Loaded {
    /// The loaded entities that carry `C`.
    pub fn with<C: Component>(&self, world: &World) -> Vec<Entity> {
        self.entities
            .iter()
            .copied()
            .filter(|entity| world.get::<C>(*entity).is_some())
            .collect()
    }
}

fn refused(message: impl Into<String>) -> ErrorReport {
    ErrorReport::new(ErrorKind::Request, message)
}

/// The relationship targets: rebuilt by their sources' hooks, never saved.
fn is_relationship_target(id: TypeId) -> bool {
    [
        TypeId::of::<bevy_ecs::hierarchy::Children>(),
        TypeId::of::<bus::Serves>(),
        TypeId::of::<agent::ModelOf>(),
        TypeId::of::<agent::RememberedBy>(),
        TypeId::of::<agent::RetrievedBy>(),
        TypeId::of::<agent::RoutedTo>(),
        TypeId::of::<agent::Grants>(),
        TypeId::of::<agent::ContextOf>(),
        TypeId::of::<agent::AttachedTo>(),
        TypeId::of::<agent::Runs>(),
        TypeId::of::<agent::AdvertisedOn>(),
        TypeId::of::<agent::content::parts::EditedBy>(),
        TypeId::of::<agent::checkpoint::AssistantForTurns>(),
        TypeId::of::<agent::checkpoint::ResultsForTurns>(),
    ]
    .contains(&id)
}

/// The world's entities in checkpoint order: roots by spawn index (an
/// `Entity`'s own `Ord` is over its opaque bits, which invert the index),
/// then each root's subtree depth-first in `Children` order.
fn ordered_entities(world: &mut World) -> Vec<Entity> {
    // A host may disable an ended run to keep it out of its hot queries; it
    // is still the world's, and a checkpoint keeps it.
    let mut roots: Vec<Entity> = world
        .query_filtered::<Entity, (Without<ChildOf>, Without<IsResource>, Allow<Disabled>)>()
        .iter(world)
        .collect();
    roots.sort_by_key(|entity| (entity.index().index(), entity.generation().to_bits()));
    let mut order = Vec::with_capacity(roots.len());
    let mut stack: Vec<Entity> = roots.into_iter().rev().collect();
    while let Some(entity) = stack.pop() {
        order.push(entity);
        if let Some(children) = world.get::<Children>(entity) {
            stack.extend(children.iter().rev());
        }
    }
    order
}

/// Take the checkpoint of `world`.
#[must_use = "saving a checkpoint does not remove anything from the world"]
pub fn save_world(world: &mut World) -> Result<Checkpoint, ErrorReport> {
    let registry = world
        .get_resource::<AppTypeRegistry>()
        .ok_or_else(|| refused("the world has no type registry: install RigPlugin first"))?
        .clone();
    let registry = registry.read();
    let mut components: Vec<(&str, &ReflectComponent)> = registry
        .iter_with_data::<ReflectComponent>()
        .filter(|(registration, _)| !is_relationship_target(registration.type_id()))
        .map(|(registration, data)| (registration.type_info().type_path(), data))
        .collect();
    components.sort_by_key(|(path, _)| *path);
    let order = ordered_entities(world);
    let mut rows: Vec<(Entity, Reflected<'_>)> = Vec::new();
    for entity in order {
        let entity_ref = world.entity(entity);
        let reflected: Reflected<'_> = components
            .iter()
            .filter_map(|(path, data)| data.reflect(entity_ref).map(|c| (*path, c.to_dynamic())))
            .collect();
        if !reflected.is_empty() {
            rows.push((entity, reflected));
        }
    }
    let index: HashMap<Entity, usize> = rows
        .iter()
        .enumerate()
        .map(|(index, (entity, _))| (*entity, index))
        .collect();
    let indexed = Indexed { index };
    let mut entities = Vec::with_capacity(rows.len());
    for (_, components) in &rows {
        let mut object = CheckpointEntity::new();
        for (path, component) in components {
            object.insert(
                (*path).to_owned(),
                reflect_to_json(component.as_ref(), &registry, &indexed)?,
            );
        }
        entities.push(object);
    }
    Ok(Checkpoint {
        entities,
        counters: Counters {
            next_run: world.get_resource::<agent::RunCounter>().map_or(0, |c| c.0),
            next_id: world.get_resource::<IdCounter>().map_or(0, |c| c.0),
        },
        binaries: world
            .get_resource::<BinaryAssets>()
            .map(BinaryAssets::snapshot)
            .unwrap_or_default(),
    })
}

fn reflect_to_json(
    value: &dyn PartialReflect,
    registry: &TypeRegistry,
    indexed: &Indexed,
) -> Result<serde_json::Value, ErrorReport> {
    let path = value
        .get_represented_type_info()
        .map(|info| info.type_path().to_owned())
        .unwrap_or_default();
    let json = serde_json::to_value(ReflectSerializer::with_processor(value, registry, indexed))
        .map_err(|error| refused(format!("{path}: {error}")))?;
    // `ReflectSerializer` wraps the value in a one-key map by type path.
    Ok(match json {
        serde_json::Value::Object(mut map) if map.len() == 1 => {
            map.remove(&path).unwrap_or(serde_json::Value::Null)
        }
        other => other,
    })
}

/// Serializes every `Entity` as the index the checkpoint gives it.
struct Indexed {
    index: HashMap<Entity, usize>,
}

impl ReflectSerializerProcessor for Indexed {
    fn try_serialize<S>(
        &self,
        value: &dyn PartialReflect,
        _registry: &TypeRegistry,
        serializer: S,
    ) -> Result<Result<S::Ok, S>, S::Error>
    where
        S: serde::Serializer,
    {
        match value.try_downcast_ref::<Entity>() {
            Some(entity) => match self.index.get(entity) {
                Some(index) => serde::Serialize::serialize(index, serializer).map(Ok),
                None => serializer.serialize_none().map(Ok),
            },
            None => Ok(Err(serializer)),
        }
    }
}

/// Deserializes every `Entity` from its checkpoint index.
struct Remapped<'a> {
    entities: &'a [Entity],
}

impl ReflectDeserializerProcessor for Remapped<'_> {
    fn try_deserialize<'de, D>(
        &mut self,
        registration: &TypeRegistration,
        _registry: &TypeRegistry,
        deserializer: D,
    ) -> Result<Result<Box<dyn PartialReflect>, D>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if registration.type_id() != TypeId::of::<Entity>() {
            return Ok(Err(deserializer));
        }
        let index: usize = serde::Deserialize::deserialize(deserializer)?;
        let entity = self.entities.get(index).copied().ok_or_else(|| {
            serde::de::Error::custom(format!("entity {index} is not in the checkpoint"))
        })?;
        Ok(Ok(Box::new(entity)))
    }
}

/// A checkpoint entity that carries a `Bound` the destination already
/// serves under the same key merges into that entity: the existing handler
/// wins, and every link to the checkpoint's handler resolves to it.
fn aliases(checkpoint: &Checkpoint, world: &World) -> Result<HashMap<usize, Entity>, ErrorReport> {
    let Some(index) = world.get_resource::<HandlerIndex>() else {
        return Ok(HashMap::new());
    };
    let mut aliases = HashMap::new();
    for (row, entity) in checkpoint.entities.iter().enumerate() {
        let Some(bound) = entity.get(std::any::type_name::<Bound>()) else {
            continue;
        };
        let bound: Bound = serde_json::from_value(bound.clone())
            .map_err(|error| refused(format!("checkpoint entity {row}: Bound: {error}")))?;
        if let Some(existing) = index.entity(&bound.key) {
            if let Some(served) = world.get::<Bound>(existing)
                && served.family() != bound.family()
            {
                return Err(refused(format!(
                    "`{}` is bound to a {} handler here; the checkpoint's is a {}",
                    bound.key,
                    served.family(),
                    bound.family()
                )));
            }
            aliases.insert(row, existing);
        }
    }
    Ok(aliases)
}

/// Spawn `checkpoint` into `world` (no validation, no merging of the
/// binary store): the mechanics of a load.
fn spawn_into(
    checkpoint: &Checkpoint,
    world: &mut World,
    aliases: &HashMap<usize, Entity>,
) -> Result<Vec<Entity>, ErrorReport> {
    let registry = world
        .get_resource::<AppTypeRegistry>()
        .ok_or_else(|| refused("the world has no type registry: install RigPlugin first"))?
        .clone();
    let registry = registry.read();
    let entities: Vec<Entity> = (0..checkpoint.entities.len())
        .map(|row| match aliases.get(&row) {
            Some(existing) => *existing,
            None => world.spawn_empty().id(),
        })
        .collect();
    let ordered: Vec<Entity> = (0..checkpoint.entities.len())
        .map(|row| entities.get(row).copied().unwrap_or(Entity::PLACEHOLDER))
        .collect();
    let entities = ordered;
    let mut remap = Remapped {
        entities: &entities,
    };
    // Effects: their saved `Seq`, re-stamped afterwards as the saved value
    // offset by the counter as it stood before the load — the saved order
    // and gaps kept, after everything already in the world, and in a
    // fresh world the saved values themselves.
    let base = world.resource::<SeqCounter>().0;
    let mut effects: Vec<(u64, Entity)> = Vec::new();
    for (row, components) in checkpoint.entities.iter().enumerate() {
        let Some(&entity) = entities.get(row) else {
            continue;
        };
        let merged = aliases.contains_key(&row);
        for (path, value) in components {
            let registration = registry.get_with_type_path(path).ok_or_else(|| {
                refused(format!(
                    "checkpoint entity {row}: `{path}` is not a registered type"
                ))
            })?;
            if is_relationship_target(registration.type_id()) {
                continue;
            }
            if merged && registration.type_id() == TypeId::of::<Bound>() {
                continue;
            }
            let component = registration.data::<ReflectComponent>().ok_or_else(|| {
                refused(format!(
                    "checkpoint entity {row}: `{path}` is not a component"
                ))
            })?;
            let reflected =
                TypedReflectDeserializer::with_processor(registration, &registry, &mut remap)
                    .deserialize(value)
                    .map_err(|error| {
                        refused(format!("checkpoint entity {row}: `{path}`: {error}"))
                    })?;
            if registration.type_id() == TypeId::of::<Seq>() {
                let seq: Seq = serde_json::from_value(value.clone())
                    .map_err(|error| refused(format!("checkpoint entity {row}: Seq: {error}")))?;
                effects.push((seq.0, entity));
                continue;
            }
            component.insert(&mut world.entity_mut(entity), reflected.as_ref(), &registry);
        }
    }
    // Every `PendingEffect` was stamped a fresh `Seq` on add; overwrite it.
    let mut next = base;
    for (saved, entity) in effects {
        let seq = base.saturating_add(saved);
        next = next.max(seq.saturating_add(1));
        world.entity_mut(entity).insert(Seq(seq));
    }
    let mut counter = world.resource_mut::<SeqCounter>();
    counter.0 = counter.0.max(next);
    // An effect taken but not answered is re-issued under its saved id.
    let taken: Vec<(Entity, Issued)> = world
        .query_filtered::<(Entity, &Issued), (With<InFlight>, Without<EffectOutcome>)>()
        .iter(world)
        .filter(|(entity, _)| entities.contains(entity))
        .map(|(entity, issued)| (entity, *issued))
        .collect();
    for (entity, Issued(id)) in taken {
        world
            .entity_mut(entity)
            .remove::<(InFlight, Issued)>()
            .insert(Reserved(id));
    }
    // The world's counters only ever move forward: a saved counter behind
    // the world's is the world's.
    let mut ids = world.resource_mut::<IdCounter>();
    ids.0 = ids.0.max(checkpoint.counters.next_id);
    let mut runs = world.get_resource_or_init::<agent::RunCounter>();
    runs.0 = runs.0.max(checkpoint.counters.next_run);
    Ok(entities)
}

/// Every invariant a loaded graph must hold, checked in the scratch world.
fn validate(world: &mut World, entities: &[Entity]) -> Result<(), ErrorReport> {
    for &entity in entities {
        if world
            .get::<Streamed>(entity)
            .is_some_and(|streamed| !streamed.events.is_empty() || !streamed.errors.is_empty())
            && world.get::<EffectOutcome>(entity).is_none()
        {
            return Err(refused(
                "an unfinished stream with delivered progress cannot resume: no provider cursor",
            ));
        }
        if world.get::<Utterance>(entity).is_some() {
            read_message(world, entity).map_err(|error| refused(error.to_string()))?;
        }
        if world.get::<ContentPart>(entity).is_some() {
            let parent = world
                .get::<ChildOf>(entity)
                .ok_or_else(|| refused("content part has no parent"))?
                .parent();
            if world.get::<Utterance>(parent).is_none()
                && world.get::<ToolResultPart>(parent).is_none()
            {
                return Err(refused("content part has an invalid parent"));
            }
        }
        if world.get::<ToolResultStatus>(entity).is_some()
            && world.get::<ToolResultPart>(entity).is_none()
        {
            return Err(refused("tool result status is not on a tool result part"));
        }
        if world.get::<agent::Run>(entity).is_none() {
            if world.get::<agent::Ready>(entity).is_some() {
                return Err(refused("ready is not on a run"));
            }
            if world.get::<agent::Prompt>(entity).is_some() {
                return Err(refused("prompt is not on a run"));
            }
        }
        if world.get::<RequestPartEdit>(entity).is_some() {
            let parent = world
                .get::<ChildOf>(entity)
                .ok_or_else(|| refused("request edit has no turn"))?
                .parent();
            let target = world
                .get::<EditTarget>(entity)
                .ok_or_else(|| refused("request edit has no target"))?
                .0;
            if world.get::<agent::Turn>(parent).is_none()
                || world.get::<ContentPart>(target).is_none()
            {
                return Err(refused("invalid request edit relationship"));
            }
        }
        if world.get::<PendingEffect>(entity).is_some()
            && world
                .get::<bus::HoldOwners>(entity)
                .is_some_and(|owners| owners.owners().any(|owner| owner.name == "rig-ecs/batch"))
            && world.get::<crate::systems::BatchHeld>(entity).is_none()
        {
            return Err(refused("batch owner is missing its runtime hold marker"));
        }
        if world.get::<crate::systems::BatchHeld>(entity).is_some()
            && (world.get::<bus::Held>(entity).is_none()
                || world.get::<agent::ToolCallSlot>(entity).is_none()
                || !world.get::<bus::HoldOwners>(entity).is_some_and(|owners| {
                    owners.owners().any(|owner| owner.name == "rig-ecs/batch")
                }))
        {
            return Err(refused(
                "batch hold is missing its barrier, owner or tool slot",
            ));
        }
        crate::agent::checkpoint::validate(world, entity)?;
    }
    wire_expansion(world)?;
    Ok(())
}

/// Every binary reference in the graph resolves in the store.
fn wire_expansion(world: &mut World) -> Result<(), ErrorReport> {
    let ids: Vec<Entity> = world
        .query_filtered::<Entity, With<Utterance>>()
        .iter(world)
        .collect();
    for entity in ids {
        read_message(world, entity).map_err(|error| refused(error.to_string()))?;
    }
    Ok(())
}

/// Load `checkpoint` into `world`: validated in a scratch world first, so a
/// refusal leaves `world` untouched; the binary store merged; a handler the
/// world already serves under a checkpoint key kept, the checkpoint's links
/// resolving to it; every effect taken but unanswered re-issued under its
/// saved id. Install application observers after loading: an insertion
/// observer would otherwise see a partially restored entity.
pub fn load_world(checkpoint: &Checkpoint, world: &mut World) -> Result<Loaded, ErrorReport> {
    let registry = world
        .get_resource::<AppTypeRegistry>()
        .ok_or_else(|| refused("the world has no type registry: install RigPlugin first"))?
        .clone();
    let empty = BinaryAssets::default();
    let assets = world
        .get_resource::<BinaryAssets>()
        .unwrap_or(&empty)
        .merged(&checkpoint.binaries)
        .map_err(|error| refused(error.to_string()))?;
    let aliases = aliases(checkpoint, world)?;
    // The scratch world: the same registry and store, a placeholder for
    // every merged handler, and nothing else.
    let mut scratch = World::new();
    scratch.insert_resource(registry);
    scratch.insert_resource(assets);
    scratch.init_resource::<SeqCounter>();
    scratch.init_resource::<IdCounter>();
    scratch.init_resource::<HandlerIndex>();
    let scratch_aliases: HashMap<usize, Entity> = aliases
        .keys()
        .map(|row| (*row, scratch.spawn_empty().id()))
        .collect();
    let scratch_entities = spawn_into(checkpoint, &mut scratch, &scratch_aliases)?;
    validate(&mut scratch, &scratch_entities)?;
    let assets = scratch
        .remove_resource::<BinaryAssets>()
        .ok_or_else(|| refused("validated asset store missing"))?;
    world.insert_resource(assets);
    let entities = spawn_into(checkpoint, world, &aliases)?;
    Ok(Loaded { entities })
}

impl Checkpoint {
    /// The checkpoint as JSON text.
    pub fn to_json(&self) -> Result<String, ErrorReport> {
        serde_json::to_string(self).map_err(|error| refused(error.to_string()))
    }

    /// A checkpoint from JSON text.
    pub fn from_json(json: &str) -> Result<Self, ErrorReport> {
        serde_json::from_str(json).map_err(|error| refused(error.to_string()))
    }
}

/// Register every component and resource of the bus and the graph, and every
/// remote wrapper, with the world's [`AppTypeRegistry`] (created if absent).
pub fn register_types(world: &mut World) {
    crate::reflect::install_reflect(world);
}
