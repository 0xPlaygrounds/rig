//! How a catalog's models are reached: with the environment's keys, else
//! through a sign-in, such as a subscription plan's.

use std::collections::HashMap;
use std::sync::Arc;

use super::{Catalog, ModelSpec};
use crate::operation::Completion;
use crate::providers::registry::{ConnectError, ConnectOptions, ProviderId};
use crate::serve::ErasedHandler;
use crate::serve::adapters::ModelAdapter;

/// A way to reach catalog models the environment has no key for, such as
/// a subscription plan signed in to. A [`Connector`] asks it for a model
/// only when the environment has no key for the model's provider.
pub trait SignIn: Send + Sync + 'static {
    /// The handler serving `spec` with the signed-in credential, or `None`
    /// when this sign-in does not serve `spec` now.
    fn handler(&self, spec: &ModelSpec) -> Option<Result<ErasedHandler, ConnectError>>;
    /// The plan a sign-in to which serves `spec`, such as `ChatGPT`, for
    /// pickers; signed in or not.
    fn plan(&self, spec: &ModelSpec) -> Option<&'static str>;
}

/// The models an application can use and how each is reached: a catalog,
/// rig's built-in one by default, whose models connect with the
/// environment's keys, else through the first [`SignIn`] that serves them.
/// Cloning is cheap.
#[derive(Clone)]
pub struct Connector {
    catalog: Catalog,
    sign_ins: Vec<Arc<dyn SignIn>>,
}

impl Default for Connector {
    fn default() -> Self {
        Self::new(Catalog::builtin().clone())
    }
}

impl Connector {
    /// A connector for the models of `catalog`, such as the built-in one
    /// with a user's overrides laid over it.
    pub fn new(catalog: Catalog) -> Self {
        Self {
            catalog,
            sign_ins: Vec::new(),
        }
    }

    /// Falls back to `sign_in` for a model the environment has no key for
    /// and no sign-in added earlier serves.
    pub fn add_sign_in(&mut self, sign_in: impl SignIn) {
        self.sign_ins.push(Arc::new(sign_in));
    }

    /// The catalog.
    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// The handler of the first sign-in that serves `spec` now, if any.
    fn signed_in(&self, spec: &ModelSpec) -> Option<Result<ErasedHandler, ConnectError>> {
        self.sign_ins
            .iter()
            .find_map(|sign_in| sign_in.handler(spec))
    }

    /// The catalog model `reference` names, as [`Catalog::resolve`] finds
    /// it, and that model as an effect handler: its provider's client built
    /// from the environment, on the shared HTTP client of the `reqwest`
    /// feature, or, without a key in the environment, the handler of a
    /// sign-in that serves it.
    ///
    /// # Errors
    ///
    /// [`ConnectError::NotFound`] for a model the catalog does not list,
    /// else what [`Catalog::connect_with`] reports, when no sign-in serves
    /// a model the environment has no key for.
    pub fn connect(
        &self,
        reference: &str,
    ) -> Result<(Arc<ModelSpec>, ErasedHandler), ConnectError> {
        let spec = self.catalog.resolve(reference)?.shared();
        let handler = match self.catalog.connect_with(&*spec, ConnectOptions::new()) {
            Ok(model) => {
                ErasedHandler::new(ModelAdapter::<Completion>::new(spec.reference(), model))
            }
            Err(error @ ConnectError::MissingKey { .. }) => {
                self.signed_in(&spec).unwrap_or(Err(error))?
            }
            Err(error) => return Err(error),
        };
        Ok((spec, handler))
    }

    /// The plan a sign-in to which serves `spec`, if any.
    pub fn plan(&self, spec: &ModelSpec) -> Option<&'static str> {
        self.sign_ins.iter().find_map(|sign_in| sign_in.plan(spec))
    }

    /// Catalog models that call tools and whose provider can be reached
    /// from the environment or a sign-in: first those of providers with a
    /// key set or signed in, then those of providers that need none (local
    /// servers such as Ollama), which a picker can mark "no key needed".
    /// The check builds the provider's client exactly as a request would,
    /// once per provider.
    pub fn reachable(&self) -> Vec<&ModelSpec> {
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
                        || self.signed_in(spec).is_some_and(|handler| handler.is_ok())
                })
            })
            .collect();
        // Stable: catalog order within each group.
        models.sort_by_key(|spec| !spec.provider.requires_credential());
        models
    }
}
