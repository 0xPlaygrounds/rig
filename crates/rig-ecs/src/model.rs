//! An agent's model: the catalog model it chose, the reasoning setting it
//! sends, and how the choice is connected. The catalog and how its models
//! are reached are rig-core's [`Connector`].

use std::fmt;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_core::catalog::{Connector, ModelSpec};
use rig_core::completion::Reasoning;
use serde::{Deserialize, Serialize};

use super::agent::{ActiveTurn, Agent, Notice};
use super::effects::Handler;
use super::journal::ReflectSaved;
use super::turn::NO_MODEL;

/// The models agents can use and how each is reached: rig's built-in
/// catalog by default. A plugin adds a sign-in with
/// [`Connector::add_sign_in`], or replaces the catalog.
#[derive(Resource, Clone, Default)]
pub struct Models(pub Connector);

/// The chosen catalog model, as `vendor/model`. It never changes in place:
/// choosing another model inserts a new one, which rebuilds the agent's
/// [`Connection`]. Saved with the session.
#[derive(Component, Reflect, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[component(immutable)]
#[reflect(Component, Saved, Clone)]
#[serde(transparent)]
pub struct ModelChoice(pub String);

impl ModelChoice {
    /// The model and reasoning setting of an agent spawned by an agent with
    /// `parent`'s: `model`, a reference to a catalog model that calls tools,
    /// when given, else the parent's; the reasoning setting named `effort`
    /// when given, else the parent's with the parent's model.
    pub fn inherit(
        models: &Connector,
        parent: (Option<&ModelChoice>, Effort),
        model: Option<&str>,
        effort: Option<&str>,
    ) -> Result<(Option<ModelChoice>, Effort), String> {
        fn asked(text: Option<&str>) -> Option<&str> {
            text.map(str::trim).filter(|text| !text.is_empty())
        }
        let resolve = |text: &str| models.catalog().resolve(text).map(|found| found.spec);
        let model = match asked(model) {
            Some(asked) => {
                let spec = resolve(asked).map_err(|why| format!("{why}; use vendor/model"))?;
                if !spec.tools {
                    return Err(format!("{asked} cannot call tools"));
                }
                Some(ModelChoice(spec.reference()))
            }
            None => parent.0.cloned(),
        };
        let effort = match (asked(effort), &model) {
            (Some(name), Some(model)) => {
                let spec = resolve(&model.0).map_err(|why| why.to_string())?;
                let named = spec.reasoning.named(name);
                Effort(
                    named
                        .map_err(|why| format!("{}: {why}", spec.display_name))?
                        .reasoning,
                )
            }
            _ if model.is_some() && model.as_ref() == parent.0 => parent.1,
            _ => Effort::default(),
        };
        Ok((model, effort))
    }
}

/// The connected model of an agent with a [`ModelChoice`]: its catalog
/// entry and the effect handler every model call is dispatched to. Built
/// from the environment once per choice, and not saved: restoring the
/// choice rebuilds it.
#[derive(Component, Clone)]
pub struct Connection {
    /// The model's catalog entry.
    pub spec: Arc<ModelSpec>,
    /// The model as an effect handler.
    pub handler: Handler,
}

/// The reasoning setting sent with each request, or `None` for the
/// provider's default. It never changes in place. Saved with the session.
#[derive(
    Component, Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize,
)]
#[component(immutable)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
pub struct Effort(pub Option<Reasoning>);

/// `default`, `off`, the level's word, or `1024 tokens`.
impl fmt::Display for Effort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            None => f.write_str("default"),
            Some(Reasoning::Off) => f.write_str("off"),
            Some(Reasoning::Effort(level)) => f.write_str(level.as_str()),
            Some(Reasoning::Budget { tokens }) => write!(f, "{tokens} tokens"),
            Some(_) => f.write_str("custom"),
        }
    }
}

/// Choose the agent's model by catalog reference (`vendor/model`).
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct SetModel {
    /// The agent.
    pub entity: Entity,
    /// The catalog reference.
    pub model: String,
}

/// Choose the agent's reasoning setting; `None` is the provider default.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct SetEffort {
    /// The agent.
    pub entity: Entity,
    /// The setting.
    pub effort: Effort,
}

/// Chooses the agent's model: a known catalog model replaces the agent's
/// [`ModelChoice`], which [`connect`] connects.
pub(crate) fn on_set_model(
    set: On<SetModel>,
    agents: Query<Has<ActiveTurn>, With<Agent>>,
    models: Res<Models>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(busy) = agents.get(set.entity) else {
        return;
    };
    if refused_mid_turn(set.entity, busy, &mut notices) {
        return;
    }
    match models.0.catalog().resolve(&set.model) {
        Ok(found) => {
            commands
                .entity(set.entity)
                .insert(ModelChoice(found.spec.reference()));
        }
        Err(why) => {
            notices.write(Notice::error(set.entity, format!("Not picked: {why}.")));
        }
    }
}

/// Connects an agent whose [`ModelChoice`] was inserted, by [`SetModel`],
/// by restoring a session or by a plugin, unless it is connected to that
/// model already: each choice is connected, or refused, once, so requests
/// never re-resolve the provider. Only a change of a connected agent's
/// model is announced: spawning and restoring an agent are silent, since
/// views show its model anyway.
pub(crate) fn connect(
    inserted: On<Insert<ModelChoice>>,
    agents: Query<(&ModelChoice, Option<&Connection>)>,
    models: Res<Models>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = inserted.entity;
    let Ok((choice, connected)) = agents.get(agent) else {
        return;
    };
    if connected.is_some_and(|connection| connection.spec.reference() == choice.0) {
        return;
    }
    match models.0.connect(&choice.0) {
        Ok((spec, handler)) => {
            if connected.is_some() {
                let model = format!("Model: {} ({}).", spec.display_name, choice.0);
                notices.write(Notice::info(agent, model));
            }
            let handler = Handler(handler);
            commands.entity(agent).insert(Connection { spec, handler });
        }
        Err(why) => {
            commands.entity(agent).remove::<Connection>();
            let why = format!("Cannot use {}: {why}.", choice.0);
            notices.write(Notice::error(agent, why));
        }
    }
}

/// Resets a reasoning setting the agent's model does not take, whichever
/// of the two was inserted last.
pub(crate) fn check_effort(
    inserted: On<Insert<(Connection, Effort)>>,
    agents: Query<(&Connection, &Effort)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = inserted.entity;
    let Ok((Connection { spec, .. }, effort)) = agents.get(agent) else {
        return;
    };
    if let Err(refusal) = spec.validate(&spec.default_options(effort.0)) {
        commands.entity(agent).insert(Effort(None));
        notices.write(Notice::info(
            agent,
            format!("Reasoning reset to default: {refusal}."),
        ));
    }
}

/// Sets the agent's reasoning setting after checking it against the model.
pub(crate) fn on_set_effort(
    set: On<SetEffort>,
    agents: Query<(Option<&Connection>, Has<ActiveTurn>), With<Agent>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((connection, busy)) = agents.get(set.entity) else {
        return;
    };
    if refused_mid_turn(set.entity, busy, &mut notices) {
        return;
    }
    let Some(connection) = connection else {
        notices.write(Notice::info(set.entity, NO_MODEL));
        return;
    };
    let spec = &connection.spec;
    match spec.validate(&spec.default_options(set.effort.0)) {
        Ok(()) => {
            commands.entity(set.entity).insert(set.effort);
            notices.write(Notice::info(
                set.entity,
                format!("Reasoning: {}.", set.effort),
            ));
        }
        Err(refusal) => {
            notices.write(Notice::error(set.entity, format!("{refusal}.")));
        }
    }
}

/// Refuses a model or reasoning change while the agent's turn runs: the
/// rest of the turn would go to a model, or use a setting, it did not start
/// with. Every sender of [`SetModel`] and [`SetEffort`] gets the same
/// refusal.
fn refused_mid_turn(agent: Entity, busy: bool, notices: &mut MessageWriter<Notice>) -> bool {
    if busy {
        notices.write(Notice::info(agent, "A turn is running; stop it first."));
    }
    busy
}
