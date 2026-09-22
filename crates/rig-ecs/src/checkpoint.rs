//! Save and restore reflected execution graphs with host-supplied handlers.
//!
//! Registered components, binary assets, and execution counters are persisted;
//! live handlers and provider launch settings are not. Restoration validates
//! original saved contracts before aliasing and reissues unfinished effects with
//! saved dispatch IDs. Relationship targets are rebuilt in checkpoint order.
//!
//! ```
//! use rig_ecs::{RigPlugin, checkpoint::{save_world, load_world, RestoreMode}};
//! let mut app = bevy_app::App::new();
//! app.add_plugins(RigPlugin::default());
//! let checkpoint = save_world(app.world_mut())?;
//! load_world(&checkpoint, app.world_mut(), RestoreMode::Strict, [])?;
//! # Ok::<(), rig_core::error::ErrorReport>(())
//! ```

mod restore;
pub use restore::{RestoreMode, load_world};

use std::{any::TypeId, collections::HashMap};

use bevy_ecs::{
    entity_disabling::Disabled,
    prelude::*,
    reflect::{AppTypeRegistry, ReflectComponent},
    resource::IsResource,
};
use bevy_reflect::{
    PartialReflect, TypePath, TypeRegistration, TypeRegistry,
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

/// The [`Checkpoint`] envelope format this crate writes and reads.
pub const CHECKPOINT_FORMAT: u32 = 2;

/// Reflected execution state with binary assets and counters.
/// Deserialization rejects unknown envelope fields and formats other than
/// [`CHECKPOINT_FORMAT`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    /// The envelope format ([`CHECKPOINT_FORMAT`]).
    #[serde(deserialize_with = "checkpoint_format")]
    pub format: u32,
    /// Every entity with a reflected component, parents before children,
    /// siblings in `Children` order.
    pub entities: Vec<CheckpointEntity>,
    /// The world's counters: the next run number (the id and sequence
    /// counters bump themselves from what is loaded).
    pub counters: Counters,
    /// The binary store: every retained payload, once per content hash.
    pub binaries: Vec<crate::agent::content::binary::BinaryRecord>,
}

impl Default for Checkpoint {
    fn default() -> Self {
        Self {
            format: CHECKPOINT_FORMAT,
            entities: Vec::new(),
            counters: Counters::default(),
            binaries: Vec::new(),
        }
    }
}

/// Deserialize the envelope's `format`, refusing any other than
/// [`CHECKPOINT_FORMAT`] by name.
fn checkpoint_format<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<u32, D::Error> {
    let format = u32::deserialize(deserializer)?;
    if format == CHECKPOINT_FORMAT {
        Ok(format)
    } else {
        Err(serde::de::Error::custom(format!(
            "load refused: the checkpoint is format {format}, this rig reads format {CHECKPOINT_FORMAT}"
        )))
    }
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
    // Disabled runs must survive checkpoints even though ordinary queries omit them.
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

/// Save registered reflected components, binary assets, and execution counters.
/// Returns an error if the type registry is absent or reflection serialization fails.
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
        format: CHECKPOINT_FORMAT,
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

/// Reuse destination entities for matching dispatch keys. Implementation
/// selection and original-descriptor validation belong to restoration preflight.
fn aliases(checkpoint: &Checkpoint, world: &World) -> Result<HashMap<usize, Entity>, ErrorReport> {
    let Some(index) = world.get_resource::<HandlerIndex>() else {
        return Ok(HashMap::new());
    };
    let mut aliases = HashMap::new();
    for (row, entity) in checkpoint.entities.iter().enumerate() {
        let Some(bound) = entity.get(Bound::type_path()) else {
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
    // Offset saved sequences so loaded effects follow existing ones without
    // changing their relative order or gaps.
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
    // Insertion hooks assign fresh sequences, which must not replace saved ordering.
    let mut next = base;
    for (saved, entity) in effects {
        let seq = base.saturating_add(saved);
        next = next.max(seq.saturating_add(1));
        world.entity_mut(entity).insert(Seq(seq));
    }
    let mut counter = world.resource_mut::<SeqCounter>();
    counter.0 = counter.0.max(next);
    // Saved IDs let handlers recognize operations retried after restoration.
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
    // Never reuse an ID or run number already allocated by the destination.
    let mut ids = world.resource_mut::<IdCounter>();
    ids.0 = ids.0.max(checkpoint.counters.next_id);
    let mut runs = world.get_resource_or_init::<agent::RunCounter>();
    runs.0 = runs.0.max(checkpoint.counters.next_run);
    Ok(entities)
}

/// Every invariant a loaded graph must hold, checked in the scratch world.
fn validate(world: &mut World, entities: &[Entity]) -> Result<(), ErrorReport> {
    for &entity in entities {
        // Resuming requires a saved handler contract, but advertised families
        // cannot restrict world handlers that accept arbitrary effects.
        if let Some(pending) = world.get::<PendingEffect>(entity)
            && world.get::<EffectOutcome>(entity).is_none()
        {
            world
                .resource::<HandlerIndex>()
                .entity(&pending.key)
                .and_then(|handler| world.get::<Bound>(handler))
                .ok_or_else(|| {
                    refused(format!(
                        "unfinished effect requires missing saved handler `{}`",
                        pending.key
                    ))
                })?;
        }
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
                && !matches!(
                    world.get::<ContentPart>(parent),
                    Some(ContentPart::ToolResult { .. })
                )
            {
                return Err(refused("content part has an invalid parent"));
            }
        }
        if world.get::<ToolResultStatus>(entity).is_some()
            && !matches!(
                world.get::<ContentPart>(entity),
                Some(ContentPart::ToolResult { .. })
            )
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

/// Preflight the original reflected graph and binary store without mutating
/// the destination. Handler completeness is validated separately.
fn validated_state(
    checkpoint: &Checkpoint,
    world: &World,
) -> Result<(BinaryAssets, HashMap<usize, Entity>), ErrorReport> {
    if checkpoint.format != CHECKPOINT_FORMAT {
        return Err(refused(format!(
            "load refused: the checkpoint is format {}, this rig reads format {CHECKPOINT_FORMAT}",
            checkpoint.format
        )));
    }
    // These rejected wire identifiers must stay fixed across Rust module renames.
    for entity in &checkpoint.entities {
        if let Some(path) = entity.keys().find(|path| {
            matches!(
                path.as_str(),
                "rig_ecs::bus::binding::ProviderBinding" | "rig_ecs::bus::binding::CredentialRef"
            )
        }) {
            return Err(refused(format!(
                "format-{} checkpoint contains removed `{path}`: migrate provider launch settings to the host, then validate the execution-only checkpoint",
                checkpoint.format
            )));
        }
    }
    if !world.contains_resource::<SeqCounter>() || !world.contains_resource::<IdCounter>() {
        return Err(refused(
            "the world has no execution counters: install RigPlugin first",
        ));
    }
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
    // Validate the original graph, including every saved descriptor. Aliasing
    // in the destination must not hide malformed or inconsistent saved data.
    let mut scratch = World::new();
    scratch.insert_resource(registry);
    scratch.insert_resource(assets);
    scratch.init_resource::<SeqCounter>();
    scratch.init_resource::<IdCounter>();
    scratch.init_resource::<HandlerIndex>();
    let scratch_entities = spawn_into(checkpoint, &mut scratch, &HashMap::new())?;
    validate(&mut scratch, &scratch_entities)?;
    let assets = scratch
        .remove_resource::<BinaryAssets>()
        .ok_or_else(|| refused("validated asset store missing"))?;
    Ok((assets, aliases))
}

fn load_state(
    checkpoint: &Checkpoint,
    world: &mut World,
    assets: BinaryAssets,
    aliases: HashMap<usize, Entity>,
) -> Result<Loaded, ErrorReport> {
    world.insert_resource(assets);
    let entities = spawn_into(checkpoint, world, &aliases)?;
    Ok(Loaded { entities })
}

impl Checkpoint {
    /// Validate saved execution data without installing state or constructing
    /// implementations. Hosts can call this before side-effectful assembly.
    /// Returns an error for invalid saved contracts, graphs, or binaries, or missing
    /// destination registry/counters. [`load_world`] checks handler compatibility
    /// and completeness.
    pub fn validate(&self, world: &World) -> Result<(), ErrorReport> {
        self.requirements()?;
        validated_state(self, world).map(|_| ())
    }

    /// Serialize the checkpoint to JSON, returning serialization failures as reports.
    pub fn to_json(&self) -> Result<String, ErrorReport> {
        serde_json::to_string(self).map_err(|error| refused(error.to_string()))
    }

    /// Parse checkpoint JSON, rejecting invalid data, unknown fields, or unsupported formats.
    pub fn from_json(json: &str) -> Result<Self, ErrorReport> {
        serde_json::from_str(json).map_err(|error| refused(error.to_string()))
    }
}

/// Register every component and resource of the bus and the graph, and every
/// remote wrapper, with the world's [`AppTypeRegistry`] (created if absent).
pub fn register_types(world: &mut World) {
    crate::reflect::install_reflect(world);
}
