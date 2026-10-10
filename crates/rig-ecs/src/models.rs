//! Catalog models the agent can use, the reasoning settings each takes,
//! and how an agent's choice of them is connected.

use std::collections::HashMap;
use std::sync::Arc;

use rig_core::catalog::{Catalog, ModelSpec};
use rig_core::completion::{Reasoning, UnsupportedOption};
use rig_core::operation::Completion;
use rig_core::providers::registry::{ConnectError, ConnectOptions, ProviderId};
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;

use bevy_ecs::prelude::*;

use super::agent::{
    ActiveTurn, Agent, Connection, Effort, ModelChoice, Notice, SetEffort, SetModel,
};
use super::effects::{Effects, Handler};
use super::turn::NO_MODEL;

/// The model and reasoning setting of an agent spawned by an agent with
/// `parent`'s: `model`, a reference in `connector`'s catalog of a model
/// that calls tools, when given, else the parent's; the parent's reasoning
/// only with the parent's model.
pub fn child_model(
    connector: &ModelConnector,
    parent: Option<&ModelChoice>,
    parent_effort: Effort,
    model: Option<&str>,
) -> Result<(Option<ModelChoice>, Effort), String> {
    let model = match model.map(str::trim) {
        Some(asked) if !asked.is_empty() => {
            let spec = connector
                .resolve(asked)
                .ok_or_else(|| format!("The catalog has no model `{asked}`; use vendor/model"))?;
            if !spec.tools {
                return Err(format!("{asked} cannot call tools"));
            }
            Some(ModelChoice(spec.reference()))
        }
        _ => parent.cloned(),
    };
    let effort = if model.is_some() && model.as_ref() == parent {
        parent_effort
    } else {
        Effort::default()
    };
    Ok((model, effort))
}

/// A way to reach catalog models the environment has no key for, such as
/// a subscription sign-in. A plugin sets it with
/// [`ModelConnector::set_sign_ins`].
pub trait SignIns: Send + Sync + 'static {
    /// Whether a sign-in serves `spec` now.
    fn serves(&self, spec: &ModelSpec) -> bool;
    /// The handler serving `spec` with the signed-in credential.
    fn handler(&self, spec: &ModelSpec) -> Result<ErasedHandler, ConnectError>;
    /// The plan a sign-in to which serves `spec`, such as `ChatGPT`, for
    /// pickers; signed in or not.
    fn plan(&self, spec: &ModelSpec) -> Option<&'static str>;
}

/// The models agents can use and how they are reached: a catalog, rig's
/// built-in one by default, whose models connect with the environment's
/// keys, else through the [`SignIns`] a plugin set.
#[derive(Resource, Clone)]
pub struct ModelConnector {
    catalog: Catalog,
    sign_ins: Option<Arc<dyn SignIns>>,
}

impl Default for ModelConnector {
    fn default() -> Self {
        Self::new(Catalog::builtin().clone())
    }
}

impl ModelConnector {
    /// A connector for the models of `catalog`, such as the built-in one
    /// with a user's overrides laid over it.
    pub fn new(catalog: Catalog) -> Self {
        Self {
            catalog,
            sign_ins: None,
        }
    }

    /// Falls back to `sign_ins` for a model the environment has no key for.
    pub fn set_sign_ins(&mut self, sign_ins: impl SignIns) {
        self.sign_ins = Some(Arc::new(sign_ins));
    }

    /// The catalog.
    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// The catalog spec a `vendor/model` reference names.
    pub fn resolve(&self, reference: &str) -> Option<Arc<ModelSpec>> {
        self.catalog
            .resolve(reference)
            .ok()
            .map(|resolved| Arc::new(resolved.spec.clone()))
    }

    /// Builds `spec`'s provider client from the environment, on rig-core's
    /// shared HTTP client (its `reqwest` feature, which the app enables),
    /// and wraps the model as an effect handler. Without a key in the environment, a
    /// model a sign-in serves signs each request with that credential
    /// instead. The [`Effects`](super::effects::Effects) keep one per
    /// model.
    pub fn handler(&self, spec: &ModelSpec) -> Result<ErasedHandler, ConnectError> {
        match self.catalog.connect_with(spec, ConnectOptions::new()) {
            Ok(model) => Ok(ErasedHandler::new(ModelAdapter::<Completion>::new(
                spec.reference(),
                model,
            ))),
            Err(error @ ConnectError::MissingKey { .. }) => match &self.sign_ins {
                Some(sign_ins) if sign_ins.serves(spec) => sign_ins.handler(spec),
                _ => Err(error),
            },
            Err(error) => Err(error),
        }
    }

    /// The plan a sign-in to which serves `spec`, if any.
    pub fn plan(&self, spec: &ModelSpec) -> Option<&'static str> {
        self.sign_ins
            .as_ref()
            .and_then(|sign_ins| sign_ins.plan(spec))
    }

    /// Catalog models that call tools and whose provider can be reached
    /// from the environment or a sign-in: first those of providers with a
    /// key set or signed in, then those of providers that need none (local
    /// servers such as Ollama), which a view marks "no key needed". The
    /// check builds the provider's client exactly as a request would, once
    /// per provider.
    pub fn available(&self) -> Vec<&ModelSpec> {
        let signed_in = |spec: &ModelSpec| {
            self.sign_ins
                .as_ref()
                .is_some_and(|sign_ins| sign_ins.serves(spec))
        };
        let mut usable: HashMap<ProviderId, bool> = HashMap::new();
        let mut models: Vec<&ModelSpec> = self
            .catalog
            .iter()
            .filter(|spec| spec.tools)
            .filter(|spec| {
                *usable.entry(spec.provider).or_insert_with(|| {
                    self.catalog
                        .connect_with(*spec, ConnectOptions::new())
                        .is_ok()
                        || signed_in(spec)
                })
            })
            .collect();
        // Stable: catalog order within each group.
        models.sort_by_key(|spec| !spec.provider.requires_credential());
        models
    }
}

/// The reasoning setting of `spec` named `name`, or what the model takes
/// instead.
pub fn effort_named(spec: &ModelSpec, name: &str) -> Result<Option<Reasoning>, String> {
    let choices = spec.reasoning.choices();
    if let Some(choice) = choices.iter().find(|choice| choice.name == name) {
        return Ok(choice.reasoning);
    }
    let names: Vec<&str> = choices.iter().map(|choice| choice.name).collect();
    Err(format!(
        "{} takes the reasoning settings {}, not `{name}`",
        spec.display_name,
        names.join(", ")
    ))
}

/// A short label for a reasoning setting.
pub fn effort_label(effort: Option<Reasoning>) -> String {
    match effort {
        None => "default".to_owned(),
        Some(Reasoning::Off) => "off".to_owned(),
        Some(Reasoning::Effort(level)) => level.as_str().to_owned(),
        Some(Reasoning::Budget { tokens }) => format!("{tokens} tokens"),
        Some(_) => "custom".to_owned(),
    }
}

/// Whether `spec` takes `effort`, or why not.
fn check_effort(spec: &ModelSpec, effort: Option<Reasoning>) -> Result<(), UnsupportedOption> {
    spec.validate(&spec.default_options(effort))
}

/// Chooses the agent's model: a known catalog model replaces the agent's
/// [`ModelChoice`], which [`connect`] connects.
pub(crate) fn on_set_model(
    set: On<SetModel>,
    agents: Query<Has<ActiveTurn>, With<Agent>>,
    connector: Res<ModelConnector>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(busy) = agents.get(set.entity) else {
        return;
    };
    if refused_mid_turn(set.entity, busy, &mut notices) {
        return;
    }
    match connector.resolve(&set.model) {
        Some(spec) => {
            commands
                .entity(set.entity)
                .insert(ModelChoice(spec.reference()));
        }
        None => {
            notices.write(Notice::error(
                set.entity,
                format!("No catalog model `{}`. Use vendor/model.", set.model),
            ));
        }
    }
}

/// Connects an agent whose [`ModelChoice`] or [`Effort`] was inserted, in
/// either order, by [`SetModel`], by restoring a session or by a plugin:
/// each choice is connected once, so requests never re-resolve the
/// provider, and a reasoning setting the model does not take is reset.
/// Only a change of a connected agent's model is announced: spawning and
/// restoring an agent are silent, since views show its model anyway.
pub(crate) fn connect(
    inserted: On<Insert<(ModelChoice, Effort)>>,
    agents: Query<(&ModelChoice, &Effort, Option<&Connection>)>,
    mut effects: ResMut<Effects>,
    connector: Res<ModelConnector>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = inserted.entity;
    let Ok((choice, effort, connected)) = agents.get(agent) else {
        return;
    };
    let spec = match connected.filter(|connection| connection.spec.reference() == choice.0) {
        Some(connection) => connection.spec.clone(),
        None => {
            let spec = connector.resolve(&choice.0);
            let spec = spec.ok_or_else(|| format!("the catalog has no model `{}`", choice.0));
            let handler = spec.and_then(|spec| match effects.model_handler(&spec, &connector) {
                Ok(handler) => Ok((spec, Handler(handler))),
                Err(error) => Err(error.to_string()),
            });
            let Ok((spec, handler)) = handler.inspect_err(|why| {
                commands.entity(agent).remove::<Connection>();
                let why = format!("Cannot use {}: {why}.", choice.0);
                notices.write(Notice::error(agent, why));
            }) else {
                return;
            };
            if connected.is_some() {
                let model = format!("Model: {} ({}).", spec.display_name, choice.0);
                notices.write(Notice::info(agent, model));
            }
            let connection = Connection { spec, handler };
            commands.entity(agent).insert(connection.clone());
            connection.spec
        }
    };
    if let Err(refusal) = check_effort(&spec, effort.0) {
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
    match check_effort(&connection.spec, set.effort.0) {
        Ok(()) => {
            commands.entity(set.entity).insert(set.effort);
            notices.write(Notice::info(
                set.entity,
                format!("Reasoning: {}.", effort_label(set.effort.0)),
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
