//! The run graph as a scene: every agent, document, link, utterance, run,
//! turn and invalid call as serde, relationships as indices into the scene
//! (or, for a handler entity, its bound key), so a fresh world rebuilds the
//! graph exactly and the driver re-issues what has no outcome. Saved
//! beside the bus module's `Scene`, which carries the effects.

use bevy_ecs::prelude::*;
use rig_core::effect::HandlerKey;
use serde::{Deserialize, Serialize};

use super::{
    AdditionalParams, Advert, Assembling, Attachment, AwaitingModel, Context, Cursor,
    DefaultMaxTurns, DocumentId, DocumentProps, DocumentText, Failed, Grant, InvalidCall,
    InvalidCalls, InvalidRetries, MaxTokens, MaxTurns, Order, OrderCounter, Output, OutputRetries,
    OutputToolName, Outputs, Owner, Parts, Preamble, Reprompt, Resolution, Role, Run, RunCounter,
    RunOf, RunResult, RunSeq, Settled, Streamed, Temperature, ToolChoiceSpec, Turn, Usage,
    UsesModel, Utterance,
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
    /// Its relationships: `uses_model`, `run_of`, `grant`, `context`,
    /// `attachment`, `advert`.
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
    ($world:expr, $entity:expr, $components:expr, $($ty:ty => $name:literal),* $(,)?) => {
        $(
            if let Some(value) = $world.get::<$ty>($entity) {
                $components.insert(
                    $name.to_owned(),
                    serde_json::to_value(value).unwrap_or(serde_json::Value::Null),
                );
            }
        )*
    };
}

macro_rules! give {
    ($world:expr, $entity:expr, $components:expr, $($ty:ty => $name:literal),* $(,)?) => {
        $(
            if let Some(value) = $components.get($name)
                && let Ok(component) = serde_json::from_value::<$ty>(value.clone())
            {
                $world.entity_mut($entity).insert(component);
            }
        )*
    };
}

impl RunScene {
    /// Take the graph of `world`: agents and documents first, then their
    /// links, then runs, then utterances, turns and invalid calls, each
    /// after its parent.
    pub fn save(world: &mut World) -> Self {
        let mut order: Vec<(u8, Entity)> = Vec::new();
        for (entity, _) in world.query::<(Entity, &Owner)>().iter(world) {
            order.push((0, entity));
        }
        for (entity, _) in world.query::<(Entity, &DocumentId)>().iter(world) {
            order.push((1, entity));
        }
        for entity in world
            .query_filtered::<Entity, Or<(With<Grant>, With<Context>)>>()
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
            take!(world, entity, components,
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
            );
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
        Self {
            entities: saved,
            next_order: world.get_resource::<OrderCounter>().map_or(0, |c| c.0),
            next_run: world.get_resource::<RunCounter>().map_or(0, |c| c.0),
        }
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
            give!(world, entity, components,
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
            );
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
