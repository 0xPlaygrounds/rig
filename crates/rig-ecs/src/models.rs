//! Catalog models the agent can use, and the reasoning settings each takes.

use std::collections::HashMap;
use std::sync::Arc;

use rig_core::catalog::{Catalog, ModelSpec};
use rig_core::completion::{Reasoning, UnsupportedOption};
use rig_core::operation::Completion;
use rig_core::providers::registry::{ConnectError, ConnectOptions, ProviderId};
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;

use bevy_ecs::prelude::*;

use super::agent::{Effort, ModelChoice};

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
pub(crate) fn check_effort(
    spec: &ModelSpec,
    effort: Option<Reasoning>,
) -> Result<(), UnsupportedOption> {
    spec.validate(&spec.default_options(effort))
}
