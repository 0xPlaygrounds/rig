//! The inspector's view (the `reflect` feature): every component of the
//! graph and the bus derives `Reflect`, the rig-core values they hold
//! reflect through opaque remote wrappers ([`crate::bus::reflect`],
//! [`crate::agent::reflect`]), [`ReflectPlugin`] registers them all, and
//! [`ReflectedScene`] is the world as reflected data — what an inspector
//! walks, and a second export beside the serde scene
//! ([`crate::agent::scene::WorldScene`], which stays the format).
//!
//! A reflected scene is canonical: its entities are ordered by their
//! reflected content, and an `Entity` in a component (a relationship, a
//! link) serializes as that entity's index in the scene, so two worlds
//! holding the same graph export the same JSON whatever their entity ids —
//! the equality `tests/reflect_scene.rs` checks between a world and the
//! world its serde scene loads into.

use std::{any::TypeId, collections::HashMap};

use bevy_app::{App, Plugin};
use bevy_ecs::{
    prelude::*,
    reflect::{AppTypeRegistry, ReflectComponent},
};
use bevy_reflect::{
    PartialReflect, TypeRegistry,
    serde::{ReflectSerializer, ReflectSerializerProcessor},
};

pub use crate::{agent::reflect::*, bus::reflect::*};

/// Registers every component of the bus and the graph, and every remote
/// wrapper, with the app's type registry.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReflectPlugin;

impl Plugin for ReflectPlugin {
    fn build(&self, app: &mut App) {
        register(app);
    }
}

macro_rules! register_all {
    ($app:expr, [$($ty:ty),* $(,)?]) => {
        $( $app.register_type::<$ty>(); )*
    };
}

/// Register every reflected type of this crate with `app`.
pub fn register(app: &mut App) {
    use crate::{agent, bus, systems};
    register_all!(
        app,
        [
            bevy_ecs::hierarchy::ChildOf,
            bevy_ecs::hierarchy::Children,
            // The bus.
            bus::PendingEffect,
            bus::Seq,
            bus::SeqCounter,
            bus::IdCounter,
            bus::Reserved,
            bus::Issued,
            bus::Held,
            bus::InFlight,
            bus::Streamed,
            bus::EffectOutcome,
            bus::ToolInputs,
            bus::ToolOutputs,
            bus::Scope,
            bus::Bound,
            HandlerKeyReflect,
            EffectKindReflect,
            EffectIdReflect,
            HandlerDescriptorReflect,
            ToolContextReflect,
            OutcomeReflect,
            StreamedOutcomeReflect,
            StreamEventsReflect,
            // The graph.
            agent::Owner,
            agent::Preamble,
            agent::Temperature,
            agent::MaxTokens,
            agent::AdditionalParams,
            agent::ToolChoiceSpec,
            agent::OutputKind,
            agent::Output,
            agent::MaxTurns,
            agent::Unhandled,
            agent::InvalidCalls,
            agent::ToolPolicy,
            agent::ToolContextSpec,
            agent::DefaultMaxTurns,
            agent::UsesModel,
            agent::ModelOf,
            agent::Remembers,
            agent::RememberedBy,
            agent::Conversation,
            agent::Remembered,
            agent::Remembering,
            agent::MemoryAppendScheduled,
            agent::PolicyVersion,
            agent::LoadingMemory,
            agent::Retrieves,
            agent::RetrievedBy,
            agent::Retrieval,
            agent::RetrievalKind,
            agent::Retrievable,
            agent::Retrieving,
            agent::Route,
            agent::RoutedTo,
            agent::Grant,
            agent::Grants,
            agent::Context,
            agent::ContextOf,
            agent::Order,
            agent::OrderCounter,
            agent::DocumentId,
            agent::DocumentText,
            agent::DocumentProps,
            agent::Attachment,
            agent::AttachedTo,
            agent::Utterance,
            agent::Role,
            agent::Parts,
            agent::MessageParts,
            agent::Run,
            agent::RunOf,
            agent::Runs,
            agent::RunSeq,
            agent::RunCounter,
            agent::Streamed,
            agent::Cursor,
            agent::Assembling,
            agent::AwaitingModel,
            agent::ResolvingTools,
            agent::Batch,
            agent::ToolCallSlot,
            agent::Settled,
            agent::Failed,
            agent::Failure,
            agent::RunResult,
            agent::Usage,
            agent::OutputRetries,
            agent::InvalidRetries,
            agent::OutputToolName,
            agent::Turn,
            agent::Advert,
            agent::AdvertisedOn,
            agent::Outputs,
            agent::Cancelled,
            agent::Retry,
            agent::RequestPatch,
            agent::Reprompt,
            agent::InvalidCall,
            agent::Resolution,
            systems::Fresh,
            systems::Folded,
            systems::Materialised,
            JsonReflect,
            OptionalJsonReflect,
            ToolChoiceReflect,
            UsageReflect,
            ToolCallIdReflect,
            ProviderCallIdReflect,
            AssistantContentsReflect,
            MessageReflect,
        ]
    );
}

/// One entity of a [`ReflectedScene`]: its reflected components, in type
/// path order.
pub struct ReflectedEntity {
    /// The entity, in the world the scene was taken from.
    pub entity: Entity,
    /// The components, each a reflected clone, ordered by type path.
    pub components: Vec<Box<dyn PartialReflect>>,
}

/// The world as reflected data: every entity with at least one registered
/// reflected component, in canonical order.
pub struct ReflectedScene {
    /// The entities.
    pub entities: Vec<ReflectedEntity>,
}

impl ReflectedScene {
    /// Take the reflected view of `world`: every component with a
    /// [`ReflectComponent`] registration, reflected off each entity. The
    /// order is canonical — by the entity's content with entity references
    /// masked, then by the world's order — so two worlds holding the same
    /// graph give the same scene.
    pub fn from_world(world: &mut World) -> Self {
        let registry = world.resource::<AppTypeRegistry>().clone();
        let registry = registry.read();
        let mut components: Vec<(&str, &ReflectComponent)> = registry
            .iter_with_data::<ReflectComponent>()
            .map(|(registration, data)| (registration.type_info().type_path(), data))
            .collect();
        components.sort_by_key(|(path, _)| *path);
        let entities: Vec<Entity> = world.query::<Entity>().iter(world).collect();
        let mut rows: Vec<ReflectedEntity> = Vec::new();
        for entity in entities {
            let entity_ref = world.entity(entity);
            let reflected: Vec<Box<dyn PartialReflect>> = components
                .iter()
                .filter_map(|(_, data)| data.reflect(entity_ref))
                .map(|component| component.to_dynamic())
                .collect();
            if !reflected.is_empty() {
                rows.push(ReflectedEntity {
                    entity,
                    components: reflected,
                });
            }
        }
        // Canonical order: by the masked content, ties by the world's order
        // (a stable sort keeps it).
        let mut keyed: Vec<(String, ReflectedEntity)> = rows
            .into_iter()
            .map(|row| {
                let key =
                    serde_json::to_string(&row.to_json(&registry, &Masked)).unwrap_or_default();
                (key, row)
            })
            .collect();
        keyed.sort_by(|(a, _), (b, _)| a.cmp(b));
        Self {
            entities: keyed.into_iter().map(|(_, row)| row).collect(),
        }
    }

    /// The scene as JSON: one object per entity, each component under its
    /// type path, an `Entity` in a component as the index of that entity
    /// in this scene (`null` for one the scene does not hold).
    pub fn to_json(&self, registry: &TypeRegistry) -> serde_json::Value {
        let index: HashMap<Entity, usize> = self
            .entities
            .iter()
            .enumerate()
            .map(|(index, row)| (row.entity, index))
            .collect();
        let indexer = Indexed { index };
        serde_json::Value::Array(
            self.entities
                .iter()
                .map(|row| row.to_json(registry, &indexer))
                .collect(),
        )
    }
}

impl ReflectedEntity {
    fn to_json<P: ReflectSerializerProcessor>(
        &self,
        registry: &TypeRegistry,
        processor: &P,
    ) -> serde_json::Value {
        let mut object = serde_json::Map::new();
        for component in &self.components {
            let path = component
                .get_represented_type_info()
                .map(|info| info.type_path().to_owned())
                .unwrap_or_default();
            let value = serde_json::to_value(ReflectSerializer::with_processor(
                component.as_ref(),
                registry,
                processor,
            ))
            .unwrap_or(serde_json::Value::Null);
            // `ReflectSerializer` wraps the value in a one-key map by type
            // path; the path is the key here.
            let inner = match value {
                serde_json::Value::Object(mut map) if map.len() == 1 => {
                    map.remove(&path).unwrap_or(serde_json::Value::Null)
                }
                other => other,
            };
            // Only the graph's known inverse relationships are sets here:
            // semantic order lives in Order / Seq. Preserve user components'
            // order even when their serialized payload is a numeric array.
            let unordered = component
                .get_represented_type_info()
                .is_some_and(|info| unordered_relationship(info.type_id()));
            let inner = match inner {
                serde_json::Value::Array(items)
                    if unordered && items.iter().all(serde_json::Value::is_u64) =>
                {
                    let mut items = items;
                    items.sort_by_key(|item| item.as_u64());
                    serde_json::Value::Array(items)
                }
                other => other,
            };
            object.insert(path, inner);
        }
        serde_json::Value::Object(object)
    }
}

// Restrict canonicalization to the inverse links owned by this graph. A
// reflected Vec or tuple has no implied set semantics, including user-defined
// relationship targets whose order the application may observe.
fn unordered_relationship(id: TypeId) -> bool {
    use crate::agent;
    [
        TypeId::of::<bevy_ecs::hierarchy::Children>(),
        TypeId::of::<agent::ModelOf>(),
        TypeId::of::<agent::RememberedBy>(),
        TypeId::of::<agent::RetrievedBy>(),
        TypeId::of::<agent::RoutedTo>(),
        TypeId::of::<agent::Grants>(),
        TypeId::of::<agent::ContextOf>(),
        TypeId::of::<agent::AttachedTo>(),
        TypeId::of::<agent::Runs>(),
        TypeId::of::<agent::AdvertisedOn>(),
    ]
    .contains(&id)
}

/// Serializes every `Entity` as the index the scene gives it.
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

/// Serializes every `Entity` as `null`: the content without the references,
/// for the canonical order.
struct Masked;

impl ReflectSerializerProcessor for Masked {
    fn try_serialize<S>(
        &self,
        value: &dyn PartialReflect,
        _registry: &TypeRegistry,
        serializer: S,
    ) -> Result<Result<S::Ok, S>, S::Error>
    where
        S: serde::Serializer,
    {
        if value.try_downcast_ref::<Entity>().is_some() {
            serializer.serialize_none().map(Ok)
        } else {
            Ok(Err(serializer))
        }
    }
}
