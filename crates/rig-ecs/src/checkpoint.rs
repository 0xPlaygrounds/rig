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
//! # Ok::<(), rig_ecs::checkpoint::CheckpointError>(())
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
use rig_core::effect::{EffectFamily, HandlerKey};
use serde::{Deserialize, Serialize, de::DeserializeSeed};

use crate::{
    agent::{
        self, Utterance,
        content::{
            binary::{BinaryAssets, BinaryError},
            parts::*,
        },
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
pub const CHECKPOINT_FORMAT: u32 = 3;

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

/// Why a checkpoint cannot be saved, parsed, validated, or restored.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CheckpointError {
    /// The world lacks a resource that [`RigPlugin`](crate::RigPlugin) installs.
    #[error("the world has no {0}: install RigPlugin first")]
    NotInstalled(&'static str),
    /// The envelope is a format other than [`CHECKPOINT_FORMAT`].
    #[error("the checkpoint is format {found}, this rig reads format {CHECKPOINT_FORMAT}")]
    UnsupportedFormat {
        /// The envelope's format.
        found: u32,
    },
    /// The checkpoint carries a component this format no longer persists.
    #[error(
        "the checkpoint contains removed `{path}`: migrate provider launch settings to the host"
    )]
    RemovedComponent {
        /// The removed component's type path.
        path: String,
    },
    /// The checkpoint envelope does not serialize to or parse from JSON.
    #[error("checkpoint JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// A live component does not serialize through reflection.
    #[error("`{path}` does not serialize: {source}")]
    Serialize {
        /// The component's type path.
        path: String,
        /// The serializer's error.
        #[source]
        source: serde_json::Error,
    },
    /// A saved component does not deserialize as its registered type.
    #[error("checkpoint entity {entity}: `{path}`: {source}")]
    Deserialize {
        /// The entity's index in [`Checkpoint::entities`].
        entity: usize,
        /// The component's type path.
        path: String,
        /// The deserializer's error.
        #[source]
        source: serde_json::Error,
    },
    /// A saved type path is not a component registered in the destination.
    #[error("checkpoint entity {entity}: `{path}` is not a registered component")]
    Unregistered {
        /// The entity's index in [`Checkpoint::entities`].
        entity: usize,
        /// The unknown type path.
        path: String,
    },
    /// A saved handler's binding key differs from its descriptor's key.
    #[error("saved handler `{key}` has a different descriptor key")]
    HandlerKeyMismatch {
        /// The binding key.
        key: HandlerKey,
    },
    /// Two saved handlers share a key.
    #[error("duplicate saved handler `{key}`")]
    DuplicateSavedHandler {
        /// The repeated key.
        key: HandlerKey,
    },
    /// The host supplied two handlers for one key.
    #[error("duplicate supplied handler `{key}`")]
    DuplicateSuppliedHandler {
        /// The repeated key.
        key: HandlerKey,
    },
    /// The host supplied a handler the checkpoint does not require.
    #[error("supplied handler `{key}` is not required by this checkpoint")]
    UnrequiredHandler {
        /// The supplied key.
        key: HandlerKey,
    },
    /// A required handler is neither supplied nor installed in the destination.
    #[error("no implementation supplied for `{key}`")]
    MissingHandler {
        /// The required key.
        key: HandlerKey,
    },
    /// An unfinished effect names a handler the checkpoint did not save.
    #[error("unfinished effect requires missing saved handler `{key}`")]
    MissingSavedHandler {
        /// The effect's handler key.
        key: HandlerKey,
    },
    /// A handler serves a different effect family than the saved one.
    #[error("handler `{key}` serves {found}; the checkpoint's serves {saved}")]
    FamilyChanged {
        /// The handler key.
        key: HandlerKey,
        /// The saved family.
        saved: EffectFamily,
        /// The destination's family.
        found: EffectFamily,
    },
    /// Under [`RestoreMode::Strict`], a handler's descriptor differs from the
    /// saved one. [`RestoreMode::Replace`] accepts the change.
    #[error("handler `{key}` differs from its original saved descriptor")]
    DescriptorChanged {
        /// The handler key.
        key: HandlerKey,
    },
    /// An unfinished stream already delivered progress and has no provider cursor.
    #[error("an unfinished stream with delivered progress cannot resume")]
    UnresumableStream,
    /// A saved utterance does not convert to a message.
    #[error("invalid saved content: {0}")]
    Content(#[from] ContentError),
    /// The saved binary store is invalid or conflicts with the destination's.
    #[error("invalid saved binary store: {0}")]
    Binary(#[from] BinaryError),
    /// The saved graph violates a structural invariant.
    #[error("invalid checkpoint graph: {0}")]
    InvalidGraph(&'static str),
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
pub fn save_world(world: &mut World) -> Result<Checkpoint, CheckpointError> {
    let registry = world
        .get_resource::<AppTypeRegistry>()
        .ok_or(CheckpointError::NotInstalled("type registry"))?
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
) -> Result<serde_json::Value, CheckpointError> {
    let path = value
        .get_represented_type_info()
        .map(|info| info.type_path().to_owned())
        .unwrap_or_default();
    let json = serde_json::to_value(ReflectSerializer::with_processor(value, registry, indexed))
        .map_err(|source| CheckpointError::Serialize {
            path: path.clone(),
            source,
        })?;
    // `ReflectSerializer` wraps the value in a one-key map by type path.
    Ok(match json {
        serde_json::Value::Object(mut map) if map.len() == 1 => {
            map.shift_remove(&path).unwrap_or(serde_json::Value::Null)
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
fn aliases(
    checkpoint: &Checkpoint,
    world: &World,
) -> Result<HashMap<usize, Entity>, CheckpointError> {
    let Some(index) = world.get_resource::<HandlerIndex>() else {
        return Ok(HashMap::new());
    };
    let mut aliases = HashMap::new();
    for (row, entity) in checkpoint.entities.iter().enumerate() {
        let Some(bound) = entity.get(Bound::type_path()) else {
            continue;
        };
        let bound = saved_bound(row, bound)?;
        if let Some(existing) = index.entity(&bound.key) {
            if let Some(served) = world.get::<Bound>(existing)
                && served.family() != bound.family()
            {
                return Err(CheckpointError::FamilyChanged {
                    saved: bound.family(),
                    found: served.family(),
                    key: bound.key,
                });
            }
            aliases.insert(row, existing);
        }
    }
    Ok(aliases)
}

/// Parse a saved [`Bound`] component of checkpoint entity `row`.
fn saved_bound(row: usize, value: &serde_json::Value) -> Result<Bound, CheckpointError> {
    serde_json::from_value(value.clone()).map_err(|source| CheckpointError::Deserialize {
        entity: row,
        path: Bound::type_path().to_owned(),
        source,
    })
}

/// Spawn `checkpoint` into `world` (no validation, no merging of the
/// binary store): the mechanics of a load.
fn spawn_into(
    checkpoint: &Checkpoint,
    world: &mut World,
    aliases: &HashMap<usize, Entity>,
) -> Result<Vec<Entity>, CheckpointError> {
    let registry = world
        .get_resource::<AppTypeRegistry>()
        .ok_or(CheckpointError::NotInstalled("type registry"))?
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
            let unregistered = || CheckpointError::Unregistered {
                entity: row,
                path: path.clone(),
            };
            let registration = registry.get_with_type_path(path).ok_or_else(unregistered)?;
            if is_relationship_target(registration.type_id()) {
                continue;
            }
            if merged && registration.type_id() == TypeId::of::<Bound>() {
                continue;
            }
            let component = registration
                .data::<ReflectComponent>()
                .ok_or_else(unregistered)?;
            let malformed = |source| CheckpointError::Deserialize {
                entity: row,
                path: path.clone(),
                source,
            };
            let reflected =
                TypedReflectDeserializer::with_processor(registration, &registry, &mut remap)
                    .deserialize(value)
                    .map_err(malformed)?;
            if registration.type_id() == TypeId::of::<Seq>() {
                let seq: Seq = serde_json::from_value(value.clone()).map_err(malformed)?;
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
fn validate(world: &mut World, entities: &[Entity]) -> Result<(), CheckpointError> {
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
                .ok_or_else(|| CheckpointError::MissingSavedHandler {
                    key: pending.key.clone(),
                })?;
        }
        if world
            .get::<Streamed>(entity)
            .is_some_and(|streamed| !streamed.events.is_empty() || !streamed.errors.is_empty())
            && world.get::<EffectOutcome>(entity).is_none()
        {
            return Err(CheckpointError::UnresumableStream);
        }
        if world.get::<Utterance>(entity).is_some() {
            read_message(world, entity)?;
        }
        if world.get::<ContentPart>(entity).is_some() {
            let parent = world
                .get::<ChildOf>(entity)
                .ok_or(CheckpointError::InvalidGraph("content part has no parent"))?
                .parent();
            if world.get::<Utterance>(parent).is_none()
                && !matches!(
                    world.get::<ContentPart>(parent),
                    Some(ContentPart::ToolResult { .. })
                )
            {
                return Err(CheckpointError::InvalidGraph(
                    "content part has an invalid parent",
                ));
            }
        }
        if world.get::<ToolResultStatus>(entity).is_some()
            && !matches!(
                world.get::<ContentPart>(entity),
                Some(ContentPart::ToolResult { .. })
            )
        {
            return Err(CheckpointError::InvalidGraph(
                "tool result status is not on a tool result part",
            ));
        }
        if world.get::<agent::Run>(entity).is_none() {
            if world.get::<agent::Ready>(entity).is_some() {
                return Err(CheckpointError::InvalidGraph("ready is not on a run"));
            }
            if world.get::<agent::Prompt>(entity).is_some() {
                return Err(CheckpointError::InvalidGraph("prompt is not on a run"));
            }
        }
        if world.get::<RequestPartEdit>(entity).is_some() {
            let parent = world
                .get::<ChildOf>(entity)
                .ok_or(CheckpointError::InvalidGraph("request edit has no turn"))?
                .parent();
            let target = world
                .get::<EditTarget>(entity)
                .ok_or(CheckpointError::InvalidGraph("request edit has no target"))?
                .0;
            if world.get::<agent::Turn>(parent).is_none()
                || world.get::<ContentPart>(target).is_none()
            {
                return Err(CheckpointError::InvalidGraph(
                    "invalid request edit relationship",
                ));
            }
        }
        if world.get::<PendingEffect>(entity).is_some()
            && world
                .get::<bus::HoldOwners>(entity)
                .is_some_and(|owners| owners.owners().any(|owner| owner.name == "rig-ecs/batch"))
            && world.get::<crate::systems::BatchHeld>(entity).is_none()
        {
            return Err(CheckpointError::InvalidGraph(
                "batch owner is missing its runtime hold marker",
            ));
        }
        if world.get::<crate::systems::BatchHeld>(entity).is_some()
            && (world.get::<bus::Held>(entity).is_none()
                || world.get::<agent::ToolCallSlot>(entity).is_none()
                || !world.get::<bus::HoldOwners>(entity).is_some_and(|owners| {
                    owners.owners().any(|owner| owner.name == "rig-ecs/batch")
                }))
        {
            return Err(CheckpointError::InvalidGraph(
                "batch hold is missing its barrier, owner or tool slot",
            ));
        }
        crate::agent::checkpoint::validate(world, entity)?;
    }
    wire_expansion(world)?;
    Ok(())
}

/// Every binary reference in the graph resolves in the store.
fn wire_expansion(world: &mut World) -> Result<(), CheckpointError> {
    let ids: Vec<Entity> = world
        .query_filtered::<Entity, With<Utterance>>()
        .iter(world)
        .collect();
    for entity in ids {
        read_message(world, entity)?;
    }
    Ok(())
}

/// Preflight the original reflected graph and binary store without mutating
/// the destination. Handler completeness is validated separately.
fn validated_state(
    checkpoint: &Checkpoint,
    world: &World,
) -> Result<(BinaryAssets, HashMap<usize, Entity>), CheckpointError> {
    if checkpoint.format != CHECKPOINT_FORMAT {
        return Err(CheckpointError::UnsupportedFormat {
            found: checkpoint.format,
        });
    }
    // These rejected wire identifiers must stay fixed across Rust module renames.
    for entity in &checkpoint.entities {
        if let Some(path) = entity.keys().find(|path| {
            matches!(
                path.as_str(),
                "rig_ecs::bus::binding::ProviderBinding" | "rig_ecs::bus::binding::CredentialRef"
            )
        }) {
            return Err(CheckpointError::RemovedComponent { path: path.clone() });
        }
    }
    if !world.contains_resource::<SeqCounter>() || !world.contains_resource::<IdCounter>() {
        return Err(CheckpointError::NotInstalled("execution counters"));
    }
    let registry = world
        .get_resource::<AppTypeRegistry>()
        .ok_or(CheckpointError::NotInstalled("type registry"))?
        .clone();
    let empty = BinaryAssets::default();
    let assets = world
        .get_resource::<BinaryAssets>()
        .unwrap_or(&empty)
        .merged(&checkpoint.binaries)?;
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
        .ok_or(CheckpointError::InvalidGraph(
            "validated binary store missing",
        ))?;
    Ok((assets, aliases))
}

fn load_state(
    checkpoint: &Checkpoint,
    world: &mut World,
    assets: BinaryAssets,
    aliases: HashMap<usize, Entity>,
) -> Result<Loaded, CheckpointError> {
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
    pub fn validate(&self, world: &World) -> Result<(), CheckpointError> {
        self.requirements()?;
        validated_state(self, world).map(|_| ())
    }

    /// Serialize the checkpoint to JSON.
    pub fn to_json(&self) -> Result<String, CheckpointError> {
        Ok(serde_json::to_string(self)?)
    }

    /// Parse checkpoint JSON. Returns [`CheckpointError::UnsupportedFormat`]
    /// for another envelope format and [`CheckpointError::Json`] for invalid
    /// data or unknown fields.
    pub fn from_json(json: &str) -> Result<Self, CheckpointError> {
        #[derive(Deserialize)]
        struct Envelope {
            format: u32,
        }
        let Envelope { format } = serde_json::from_str(json)?;
        if format != CHECKPOINT_FORMAT {
            return Err(CheckpointError::UnsupportedFormat { found: format });
        }
        Ok(serde_json::from_str(json)?)
    }
}

/// Register every component and resource of the bus and the graph, and every
/// remote wrapper, with the world's [`AppTypeRegistry`] (created if absent).
pub fn register_types(world: &mut World) {
    crate::reflect::install_reflect(world);
}
