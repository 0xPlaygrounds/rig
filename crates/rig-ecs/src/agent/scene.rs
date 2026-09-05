//! The run graph as a scene: every agent, document, link, utterance, run,
//! turn and invalid call as serde, relationships as indices into the scene
//! (or, for a handler entity, its bound key), so a fresh world rebuilds the
//! graph exactly and the driver re-issues what has no outcome. Saved
//! beside the bus module's `Scene`, which carries the effects: the pair is
//! [`WorldScene`], and an effect `ChildOf` a turn keeps that parent across
//! the two by index ([`save_world`], [`load_world`]).

use bevy_ecs::prelude::*;
use rig_core::effect::HandlerKey;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::PolicyVersion;
use super::{
    AdditionalParams, Advert, Assembling, Attachment, AwaitingModel, Batch, Cancelled, Context,
    Conversation, Cursor, DefaultMaxTurns, DocumentId, DocumentProps, DocumentText, Failed, Grant,
    InvalidCall, InvalidCalls, InvalidRetries, LoadingMemory, MaxTokens, MaxTurns,
    MemoryAppendScheduled, Order, OrderCounter, Output, OutputRetries, OutputToolName, Outputs,
    Owner, Parts, Preamble, Remembered, Remembering, Remembers, Reprompt, RequestPatch, Resolution,
    ResolvingTools, Retrievable, Retrieval, Retrieves, Retrieving, Retry, Role, Route, Run,
    RunCounter, RunOf, RunResult, RunSeq, Settled, Streamed, Temperature, ToolCallSlot,
    ToolChoiceSpec, ToolContextSpec, ToolPolicy, Turn, Usage, UsesModel, Utterance,
};
use crate::bus::{Bound, Scope};

/// What a scene entity is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneKind {
    /// An agent.
    Agent,
    /// A document.
    Document,
    /// A grant, context, attachment or advert link.
    Link,
    /// An utterance.
    Utterance,
    /// A run.
    Run,
    /// A turn.
    Turn,
    /// An invalid call awaiting or holding its resolution.
    InvalidCall,
}

/// A link's target: another scene entity, or a handler by its bound key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "target", rename_all = "snake_case")]
pub enum Target {
    /// Another entity of the scene, by index.
    Scene {
        /// The index in [`RunScene::entities`].
        index: usize,
    },
    /// A handler entity, by its key: bound again by the host at load.
    Handler {
        /// The bound key.
        key: HandlerKey,
    },
}

/// One entity of the graph: its kind, its serde components by name, its
/// parent by index, and its relationships by name and target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneEntity {
    /// What it is.
    pub kind: SceneKind,
    /// Its components, each under its name.
    pub components: serde_json::Map<String, serde_json::Value>,
    /// Its `ChildOf`, by index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<usize>,
    /// Its relationships: `uses_model`, `run_of`, `grant`, `route`,
    /// `context`, `attachment`, `advert`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relations: Vec<(String, Target)>,
}

/// The run graph as data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunScene {
    /// The entities, parents before children.
    pub entities: Vec<SceneEntity>,
    /// The order counter, so loaded orders never collide with new ones.
    pub next_order: u64,
    /// The run counter, likewise.
    pub next_run: u64,
}

macro_rules! take {
    ($world:expr, $entity:expr, $components:expr, $errors:expr, $($ty:ty => $name:literal),* $(,)?) => {
        $(
            if let Some(value) = $world.get::<$ty>($entity) {
                match serde_json::to_value(value) {
                    Ok(json) => {
                        $components.insert($name.to_owned(), json);
                    }
                    Err(error) => $errors.push(format!("{}: {error}", $name)),
                }
            }
        )*
    };
}

macro_rules! give {
    ($world:expr, $entity:expr, $components:expr, $errors:expr, $($ty:ty => $name:literal),* $(,)?) => {
        $(
            if let Some(value) = $components.get($name) {
                match serde_json::from_value::<$ty>(value.clone()) {
                    Ok(component) => {
                        $world.entity_mut($entity).insert(component);
                    }
                    Err(error) => $errors.push(format!("{}: {error}", $name)),
                }
            }
        )*
    };
}

/// The supported persistent state of the agent runtime: the run graph and the
/// bus module's effects, saved together so an effect `ChildOf` a turn is
/// `ChildOf` it again after a load — which is what lets a run saved with
/// its model call in flight resume: the effect is re-issued under its saved
/// id, answered, and read by the turn it belongs to. Completed streams retain
/// their collected state. An unfinished stream that already delivered progress
/// is refused before graph restoration; generic scenes have no restart cursor.
///
/// Only the library's listed components and components explicitly registered
/// with [`SceneExtensions`] are captured. Resources, arbitrary entities,
/// system-local state, tasks and live handles remain the host's responsibility.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorldScene {
    /// The graph.
    pub graph: RunScene,
    /// The effects, with [`crate::bus::SceneEffect::parent_ref`] indexing
    /// `graph.entities`.
    pub effects: crate::bus::Scene,
    /// Which call each tool effect is ([`ToolCallSlot`]), by index into
    /// `effects.effects`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub slots: Vec<(usize, ToolCallSlot)>,
    /// Which index each retrieval effect asks for ([`Retrieval`]), by
    /// index into `effects.effects`: a run cut while retrieving resumes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub retrievals: Vec<(usize, Retrieval)>,
    /// Registered extension components on graph entities, keyed by graph index
    /// and the application's versioned component name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<usize, BTreeMap<String, serde_json::Value>>,
}

/// Explicit persistence registration for application components on entities
/// already captured by [`WorldScene::graph`]. Install the same registrations
/// in the source and destination worlds before saving or loading.
///
/// Payloads must be entity-independent serde data with pure, deterministic
/// serialization and deserialization (loading validates then deserializes).
/// Entity references inside a
/// payload are **not** remapped; use the library's graph relationships for
/// ownership and links. Registrations do not capture resources, additional
/// entities, system-local state or effects' custom components. Version names
/// when the payload's schema or meaning changes. Unregistered components are
/// outside the contract; a saved name missing at load is an error.
#[derive(Resource, Default, Clone)]
pub struct SceneExtensions {
    components: BTreeMap<String, ComponentCodec>,
}

#[derive(Clone)]
struct ComponentCodec {
    save: fn(&World, Entity) -> Result<Option<serde_json::Value>, serde_json::Error>,
    validate: fn(serde_json::Value) -> Result<(), serde_json::Error>,
    load: fn(&mut World, Entity, serde_json::Value) -> Result<(), serde_json::Error>,
}

impl SceneExtensions {
    /// Register one component under a nonempty, unique, application-owned name
    /// such as `acme/retry-budget/v1`. Duplicate names are rejected, including
    /// repeated registration of the same type. The host must use one name per
    /// type and register the same type for that name in both worlds.
    pub fn register_component<T>(
        &mut self,
        name: impl Into<String>,
    ) -> Result<(), rig_core::error::ErrorReport>
    where
        T: Component + Serialize + serde::de::DeserializeOwned,
    {
        let name = name.into();
        if name.trim().is_empty() || self.components.contains_key(&name) {
            return Err(extension_error("empty or duplicate extension name"));
        }
        self.components.insert(
            name,
            ComponentCodec {
                save: |world, entity| world.get::<T>(entity).map(serde_json::to_value).transpose(),
                validate: |value| serde_json::from_value::<T>(value).map(|_| ()),
                load: |world, entity, value| {
                    world
                        .entity_mut(entity)
                        .insert(serde_json::from_value::<T>(value)?);
                    Ok(())
                },
            },
        );
        Ok(())
    }
}

fn extension_error(message: impl Into<String>) -> rig_core::error::ErrorReport {
    rig_core::error::ErrorReport::new(rig_core::error::ErrorKind::Request, message)
}

/// What [`load_world`] spawned, by scene index.
#[derive(Debug, Clone, Default)]
pub struct Loaded {
    /// The graph's entities, by [`RunScene::entities`] index.
    pub graph: Vec<Entity>,
    /// The effect entities, by [`crate::bus::Scene::effects`] index.
    pub effects: Vec<Entity>,
}

/// Save the graph and the effects of `world` as one [`WorldScene`].
pub fn save_world(world: &mut World) -> Result<WorldScene, rig_core::error::ErrorReport> {
    let (graph, entities) = RunScene::take(world)?;
    let mut extensions = BTreeMap::<usize, BTreeMap<String, serde_json::Value>>::new();
    if let Some(registry) = world.get_resource::<SceneExtensions>() {
        for (index, entity) in entities.iter().enumerate() {
            for (name, codec) in &registry.components {
                if let Some(value) = (codec.save)(world, *entity)
                    .map_err(|error| extension_error(format!("extension {name}: {error}")))?
                {
                    extensions
                        .entry(index)
                        .or_default()
                        .insert(name.clone(), value);
                }
            }
        }
    }
    let effects = crate::bus::Scene::save_with(world, |parent| {
        entities.iter().position(|entity| *entity == parent)
    });
    // The effects in the scene's order, to pair each tool effect's slot.
    let mut rows: Vec<(Entity, crate::bus::Seq)> = world
        .query::<(Entity, &crate::bus::Seq, &crate::bus::PendingEffect)>()
        .iter(world)
        .map(|(entity, seq, _)| (entity, *seq))
        .collect();
    rows.sort_by_key(|(_, seq)| *seq);
    let slots = rows
        .iter()
        .enumerate()
        .filter_map(|(index, (entity, _))| {
            world
                .get::<ToolCallSlot>(*entity)
                .map(|slot| (index, slot.clone()))
        })
        .collect();
    let retrievals = rows
        .iter()
        .enumerate()
        .filter_map(|(index, (entity, _))| {
            world
                .get::<Retrieval>(*entity)
                .map(|retrieval| (index, *retrieval))
        })
        .collect();
    Ok(WorldScene {
        graph,
        effects,
        slots,
        retrievals,
        extensions,
    })
}

/// Load `scene` into `world`: the graph first, then the effects, each
/// effect `ChildOf` the graph entity its `parent_ref` names. Handlers are
/// the host's to bind first, as for [`RunScene::load`].
/// Registered extensions are validated before spawning and inserted after the
/// graph and effects have loaded. Install application observers after loading:
/// insertion observers can otherwise see a partially restored entity. Loading
/// is not transactional if a graph error, extension insertion/deserialization
/// or application observer fails.
pub fn load_world(
    scene: &WorldScene,
    world: &mut World,
) -> Result<Loaded, rig_core::error::ErrorReport> {
    scene.effects.validate_resume()?;
    let registry = world
        .get_resource::<SceneExtensions>()
        .cloned()
        .unwrap_or_default();
    for (index, components) in &scene.extensions {
        if *index >= scene.graph.entities.len() {
            return Err(extension_error("extension graph index is out of bounds"));
        }
        for (name, value) in components {
            let codec = registry
                .components
                .get(name)
                .ok_or_else(|| extension_error(format!("unregistered scene extension {name}")))?;
            (codec.validate)(value.clone())
                .map_err(|error| extension_error(format!("extension {name}: {error}")))?;
        }
    }
    let graph = scene.graph.load(world)?;
    let effects = scene
        .effects
        .load_with(world, |index| graph.get(index).copied())?;
    for (index, slot) in &scene.slots {
        if let Some(effect) = effects.get(*index).copied() {
            world.entity_mut(effect).insert(slot.clone());
        }
    }
    for (index, retrieval) in &scene.retrievals {
        if let Some(effect) = effects.get(*index).copied() {
            world.entity_mut(effect).insert(*retrieval);
        }
    }
    for (index, components) in &scene.extensions {
        if let Some(entity) = graph.get(*index).copied() {
            for (name, value) in components {
                if let Some(codec) = registry.components.get(name) {
                    (codec.load)(world, entity, value.clone())
                        .map_err(|error| extension_error(format!("extension {name}: {error}")))?;
                }
            }
        }
    }
    Ok(Loaded { graph, effects })
}

impl RunScene {
    /// Take the graph of `world`: agents and documents first, then their
    /// links, then runs, then utterances, turns and invalid calls, each
    /// after its parent.
    pub fn save(world: &mut World) -> Result<Self, rig_core::error::ErrorReport> {
        Self::take(world).map(|(scene, _)| scene)
    }

    /// [`RunScene::save`], with the entity each scene index was taken from.
    pub fn take(world: &mut World) -> Result<(Self, Vec<Entity>), rig_core::error::ErrorReport> {
        let mut order: Vec<(u8, Entity)> = Vec::new();
        for (entity, _) in world.query::<(Entity, &Owner)>().iter(world) {
            order.push((0, entity));
        }
        for (entity, _) in world.query::<(Entity, &DocumentId)>().iter(world) {
            order.push((1, entity));
        }
        for entity in world
            .query_filtered::<Entity, Or<(With<Grant>, With<Context>, With<Route>, With<Retrieves>)>>()
            .iter(world)
        {
            order.push((2, entity));
        }
        for (entity, _) in world.query::<(Entity, &Run)>().iter(world) {
            order.push((3, entity));
        }
        for (entity, _) in world.query::<(Entity, &Utterance)>().iter(world) {
            order.push((4, entity));
        }
        for (entity, _) in world.query::<(Entity, &Turn)>().iter(world) {
            order.push((5, entity));
        }
        for entity in world
            .query_filtered::<Entity, Or<(With<Attachment>, With<Advert>)>>()
            .iter(world)
        {
            order.push((6, entity));
        }
        for (entity, _) in world.query::<(Entity, &InvalidCall)>().iter(world) {
            order.push((7, entity));
        }
        order.sort_by_key(|(rank, entity)| (*rank, entity.index()));
        let entities: Vec<Entity> = order.iter().map(|(_, entity)| *entity).collect();
        let index_of = |entity: Entity| entities.iter().position(|e| *e == entity);
        let target_of = |world: &World, entity: Entity| -> Option<Target> {
            if let Some(index) = index_of(entity) {
                return Some(Target::Scene { index });
            }
            world.get::<Bound>(entity).map(|bound| Target::Handler {
                key: bound.key.clone(),
            })
        };

        let mut saved = Vec::with_capacity(entities.len());
        for (rank, entity) in &order {
            let entity = *entity;
            let kind = match rank {
                0 => SceneKind::Agent,
                1 => SceneKind::Document,
                2 | 6 => SceneKind::Link,
                3 => SceneKind::Run,
                4 => SceneKind::Utterance,
                5 => SceneKind::Turn,
                _ => SceneKind::InvalidCall,
            };
            let mut components = serde_json::Map::new();
            let mut errors: Vec<String> = Vec::new();
            take!(world, entity, components, errors,
                Owner => "owner", Preamble => "preamble", Temperature => "temperature",
                MaxTokens => "max_tokens", AdditionalParams => "additional_params",
                ToolChoiceSpec => "tool_choice", Output => "output", MaxTurns => "max_turns",
                InvalidCalls => "invalid_calls", DefaultMaxTurns => "default_max_turns",
                DocumentId => "document_id", DocumentText => "document_text",
                DocumentProps => "document_props", Order => "order",
                Utterance => "utterance", Role => "role", Parts => "parts",
                Run => "run", RunSeq => "run_seq", Streamed => "streamed", Cursor => "cursor",
                Assembling => "assembling", AwaitingModel => "awaiting_model",
                Settled => "settled", Failed => "failed", RunResult => "run_result",
                Usage => "usage", OutputRetries => "output_retries",
                InvalidRetries => "invalid_retries", OutputToolName => "output_tool_name",
                Scope => "scope", Turn => "turn", Outputs => "outputs", Reprompt => "reprompt",
                InvalidCall => "invalid_call", Resolution => "resolution",
                ToolPolicy => "tool_policy", ToolContextSpec => "tool_context",
                ResolvingTools => "resolving_tools", Batch => "batch",
                Cancelled => "cancelled", Retry => "retry", RequestPatch => "request_patch",
                Conversation => "conversation", Remembered => "remembered",
                Remembering => "remembering", LoadingMemory => "loading_memory",
                MemoryAppendScheduled => "memory_append_scheduled",
                PolicyVersion => "policy_version",
                Retrieval => "retrieval", Retrievable => "retrievable", Retrieving => "retrieving",
            );
            if !errors.is_empty() {
                return Err(rig_core::error::ErrorReport::new(
                    rig_core::error::ErrorKind::Internal,
                    format!(
                        "a component of entity {} did not serialize: {}",
                        saved.len(),
                        errors.join("; ")
                    ),
                ));
            }
            if world.get::<crate::systems::Fresh>(entity).is_some() {
                components.insert("fresh".to_owned(), serde_json::Value::Bool(true));
            }
            if let Some(crate::systems::Folded(mode)) = world.get::<crate::systems::Folded>(entity)
            {
                components.insert(
                    "folded".to_owned(),
                    serde_json::to_value(mode).unwrap_or(serde_json::Value::Null),
                );
            }
            if world.get::<crate::systems::Materialised>(entity).is_some() {
                components.insert("materialised".to_owned(), serde_json::Value::Bool(true));
            }
            let parent = world
                .get::<ChildOf>(entity)
                .and_then(|child_of| index_of(child_of.parent()));
            let mut relations = Vec::new();
            if let Some(UsesModel(model)) = world.get::<UsesModel>(entity)
                && let Some(target) = target_of(world, *model)
            {
                relations.push(("uses_model".to_owned(), target));
            }
            if let Some(RunOf(agent)) = world.get::<RunOf>(entity)
                && let Some(target) = target_of(world, *agent)
            {
                relations.push(("run_of".to_owned(), target));
            }
            if let Some(Grant(tool)) = world.get::<Grant>(entity)
                && let Some(target) = target_of(world, *tool)
            {
                relations.push(("grant".to_owned(), target));
            }
            if let Some(Route(model)) = world.get::<Route>(entity)
                && let Some(target) = target_of(world, *model)
            {
                relations.push(("route".to_owned(), target));
            }
            if let Some(Remembers(memory)) = world.get::<Remembers>(entity)
                && let Some(target) = target_of(world, *memory)
            {
                relations.push(("remembers".to_owned(), target));
            }
            if let Some(Retrieves(index)) = world.get::<Retrieves>(entity)
                && let Some(target) = target_of(world, *index)
            {
                relations.push(("retrieves".to_owned(), target));
            }
            if let Some(Context(document)) = world.get::<Context>(entity)
                && let Some(target) = target_of(world, *document)
            {
                relations.push(("context".to_owned(), target));
            }
            if let Some(Attachment(document)) = world.get::<Attachment>(entity)
                && let Some(target) = target_of(world, *document)
            {
                relations.push(("attachment".to_owned(), target));
            }
            if let Some(Advert(tool)) = world.get::<Advert>(entity)
                && let Some(target) = target_of(world, *tool)
            {
                relations.push(("advert".to_owned(), target));
            }
            saved.push(SceneEntity {
                kind,
                components,
                parent,
                relations,
            });
        }
        Ok((
            Self {
                entities: saved,
                next_order: world.get_resource::<OrderCounter>().map_or(0, |c| c.0),
                next_run: world.get_resource::<RunCounter>().map_or(0, |c| c.0),
            },
            entities,
        ))
    }

    /// Spawn the graph into `world`. Handlers are the host's to bind first:
    /// a relationship to a handler key nothing is bound to is an error
    /// naming the key. Returns the spawned entities by scene index.
    pub fn load(&self, world: &mut World) -> Result<Vec<Entity>, rig_core::error::ErrorReport> {
        let mut spawned: Vec<Entity> = Vec::with_capacity(self.entities.len());
        for _ in &self.entities {
            spawned.push(world.spawn_empty().id());
        }
        let handler =
            |world: &mut World, key: &HandlerKey| -> Result<Entity, rig_core::error::ErrorReport> {
                world
                    .query::<(Entity, &Bound)>()
                    .iter(world)
                    .find(|(_, bound)| &bound.key == key)
                    .map(|(entity, _)| entity)
                    .ok_or_else(|| {
                        rig_core::error::ErrorReport::new(
                            rig_core::error::ErrorKind::HandlerUnavailable,
                            format!("the scene needs `{key}` bound before it loads"),
                        )
                    })
            };
        for (index, saved) in self.entities.iter().enumerate() {
            let Some(entity) = spawned.get(index).copied() else {
                continue;
            };
            let components = &saved.components;
            let mut errors: Vec<String> = Vec::new();
            give!(world, entity, components, errors,
                Owner => "owner", Preamble => "preamble", Temperature => "temperature",
                MaxTokens => "max_tokens", AdditionalParams => "additional_params",
                ToolChoiceSpec => "tool_choice", Output => "output", MaxTurns => "max_turns",
                InvalidCalls => "invalid_calls", DefaultMaxTurns => "default_max_turns",
                DocumentId => "document_id", DocumentText => "document_text",
                DocumentProps => "document_props", Order => "order",
                Utterance => "utterance", Role => "role", Parts => "parts",
                Run => "run", RunSeq => "run_seq", Streamed => "streamed", Cursor => "cursor",
                Assembling => "assembling", AwaitingModel => "awaiting_model",
                Settled => "settled", Failed => "failed", RunResult => "run_result",
                Usage => "usage", OutputRetries => "output_retries",
                InvalidRetries => "invalid_retries", OutputToolName => "output_tool_name",
                Scope => "scope", Turn => "turn", Outputs => "outputs", Reprompt => "reprompt",
                InvalidCall => "invalid_call", Resolution => "resolution",
                ToolPolicy => "tool_policy", ToolContextSpec => "tool_context",
                ResolvingTools => "resolving_tools", Batch => "batch",
                Cancelled => "cancelled", Retry => "retry", RequestPatch => "request_patch",
                Conversation => "conversation", Remembered => "remembered",
                Remembering => "remembering", LoadingMemory => "loading_memory",
                MemoryAppendScheduled => "memory_append_scheduled",
                PolicyVersion => "policy_version",
                Retrieval => "retrieval", Retrievable => "retrievable", Retrieving => "retrieving",
            );
            if !errors.is_empty() {
                return Err(rig_core::error::ErrorReport::new(
                    rig_core::error::ErrorKind::Internal,
                    format!(
                        "a component of scene entity {index} did not deserialize: {}",
                        errors.join("; ")
                    ),
                ));
            }
            if components.get("fresh").is_some() {
                world.entity_mut(entity).insert(crate::systems::Fresh);
            }
            if let Some(mode) = components.get("folded")
                && let Ok(mode) = serde_json::from_value(mode.clone())
            {
                world
                    .entity_mut(entity)
                    .insert(crate::systems::Folded(mode));
            }
            if components.get("materialised").is_some() {
                world
                    .entity_mut(entity)
                    .insert(crate::systems::Materialised);
            }
            if let Some(parent) = saved.parent.and_then(|at| spawned.get(at).copied()) {
                world.entity_mut(entity).insert(ChildOf(parent));
            }
            for (name, target) in &saved.relations {
                let to = match target {
                    Target::Scene { index } => spawned.get(*index).copied().ok_or_else(|| {
                        rig_core::error::ErrorReport::new(
                            rig_core::error::ErrorKind::Internal,
                            format!("the scene's `{name}` names entity {index}, which it lacks"),
                        )
                    })?,
                    Target::Handler { key } => handler(world, key)?,
                };
                let mut entity = world.entity_mut(entity);
                match name.as_str() {
                    "uses_model" => {
                        entity.insert(UsesModel(to));
                    }
                    "run_of" => {
                        entity.insert(RunOf(to));
                    }
                    "grant" => {
                        entity.insert(Grant(to));
                    }
                    "route" => {
                        entity.insert(Route(to));
                    }
                    "remembers" => {
                        entity.insert(Remembers(to));
                    }
                    "retrieves" => {
                        entity.insert(Retrieves(to));
                    }
                    "context" => {
                        entity.insert(Context(to));
                    }
                    "attachment" => {
                        entity.insert(Attachment(to));
                    }
                    "advert" => {
                        entity.insert(Advert(to));
                    }
                    other => {
                        return Err(rig_core::error::ErrorReport::new(
                            rig_core::error::ErrorKind::Internal,
                            format!("the scene names a relationship `{other}` this crate has not"),
                        ));
                    }
                }
            }
        }
        if let Some(mut counter) = world.get_resource_mut::<OrderCounter>() {
            counter.0 = counter.0.max(self.next_order);
        }
        if let Some(mut counter) = world.get_resource_mut::<RunCounter>() {
            counter.0 = counter.0.max(self.next_run);
        }
        Ok(spawned)
    }
}
