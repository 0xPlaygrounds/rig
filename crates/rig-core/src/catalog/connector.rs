//! How a catalog's models are reached: through a provider registered for
//! their vendor, else with the environment's keys, else through a sign-in,
//! such as a subscription plan's.

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

/// How a [`Connector`] reaches every model of one vendor, in place of the
/// registry's preset and the environment: a vendor declared at run time
/// ([`ProviderId::declare`]) on a gateway of its own, a built-in vendor at
/// another address, or a scripted model in a test.
///
/// [`ConnectOptions`] is one, natively: the vendor's models connect with
/// those options (a base URL, a key, an HTTP client, a protocol family). A
/// closure from the spec and the catalog to a handler is another.
pub trait Provide: Send + Sync + 'static {
    /// The handler serving `spec`, a model of `catalog`.
    fn handler(&self, spec: &ModelSpec, catalog: &Catalog) -> Result<ErasedHandler, ConnectError>;
}

impl<F> Provide for F
where
    F: Fn(&ModelSpec, &Catalog) -> Result<ErasedHandler, ConnectError> + Send + Sync + 'static,
{
    fn handler(&self, spec: &ModelSpec, catalog: &Catalog) -> Result<ErasedHandler, ConnectError> {
        self(spec, catalog)
    }
}

/// The vendor's models connect as [`Catalog::connect_with`] connects them
/// with these options. Native only: on the web an HTTP client is not
/// `Send`, so a connector there takes a closure.
#[cfg(not(target_family = "wasm"))]
impl Provide for ConnectOptions {
    fn handler(&self, spec: &ModelSpec, catalog: &Catalog) -> Result<ErasedHandler, ConnectError> {
        catalog
            .connect_with(spec, self.clone())
            .map(|model| adapted(spec, model))
    }
}

/// `model`, which serves `spec`, as an effect handler labelled with the
/// spec's reference.
fn adapted(spec: &ModelSpec, model: crate::driver::DynModel<Completion>) -> ErasedHandler {
    ErasedHandler::new(ModelAdapter::<Completion>::new(spec.reference(), model))
}

/// The models an application can use and how each is reached: a catalog,
/// rig's built-in one by default, whose models connect through the
/// provider registered for their vendor ([`Self::provide`]), else with the
/// environment's keys, else through the first [`SignIn`] that serves them.
/// Cloning is cheap.
#[derive(Clone)]
pub struct Connector {
    catalog: Catalog,
    providers: HashMap<&'static str, Arc<dyn Provide>>,
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
            providers: HashMap::new(),
            sign_ins: Vec::new(),
        }
    }

    /// Falls back to `sign_in` for a model the environment has no key for
    /// and no sign-in added earlier serves.
    pub fn add_sign_in(&mut self, sign_in: impl SignIn) {
        self.sign_ins.push(Arc::new(sign_in));
    }

    /// Reaches every model of `vendor` (any of its protocol families)
    /// through `provider`, before the environment and the sign-ins. A later
    /// provider for the same vendor replaces this one.
    ///
    /// ```
    /// use rig_core::catalog::{Catalog, Connector, ModelSpec};
    /// use rig_core::providers::registry::{ConnectOptions, ProviderId};
    ///
    /// let corp = ProviderId::declare("corp")?;
    /// let mut connector = Connector::default();
    /// connector.catalog_mut().insert(ModelSpec::new(corp, "big").with_tools(true));
    /// // A built-in vendor at another address.
    /// let ollama = ProviderId::catalog("ollama").ok_or("known")?;
    /// connector.provide(ollama, ConnectOptions::new().base_url("http://gpu:11434/v1"));
    /// assert!(connector.is_provided(ollama) && !connector.is_provided(corp));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn provide(&mut self, vendor: ProviderId, provider: impl Provide) {
        self.providers.insert(vendor.vendor(), Arc::new(provider));
    }

    /// Whether a provider is registered for `vendor`'s models.
    pub fn is_provided(&self, vendor: ProviderId) -> bool {
        self.providers.contains_key(vendor.vendor())
    }

    /// The catalog.
    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// The catalog, to add or change models in place, keeping the
    /// providers and sign-ins.
    pub fn catalog_mut(&mut self) -> &mut Catalog {
        &mut self.catalog
    }

    /// The handler of the first sign-in that serves `spec` now, if any.
    fn signed_in(&self, spec: &ModelSpec) -> Option<Result<ErasedHandler, ConnectError>> {
        self.sign_ins
            .iter()
            .find_map(|sign_in| sign_in.handler(spec))
    }

    /// The provider registered for `spec`'s vendor, if any.
    fn provider(&self, spec: &ModelSpec) -> Option<&Arc<dyn Provide>> {
        self.providers.get(spec.provider.vendor())
    }

    /// The catalog model `reference` names, as [`Catalog::resolve`] finds
    /// it, and that model as an effect handler: from the provider
    /// registered for its vendor, else its provider's client built from the
    /// environment, on the shared HTTP client of the `reqwest` feature, or,
    /// without a key in the environment, the handler of a sign-in that
    /// serves it.
    ///
    /// # Errors
    ///
    /// [`ConnectError::NotFound`] for a model the catalog does not list,
    /// what the registered provider reports, else what
    /// [`Catalog::connect_with`] reports, when no sign-in serves a model
    /// the environment has no key for.
    pub fn connect(
        &self,
        reference: &str,
    ) -> Result<(Arc<ModelSpec>, ErasedHandler), ConnectError> {
        let spec = self.catalog.resolve(reference)?.shared();
        let handler = self.handler(&spec)?;
        Ok((spec, handler))
    }

    /// The handler serving `spec`, as [`Self::connect`] finds it.
    fn handler(&self, spec: &ModelSpec) -> Result<ErasedHandler, ConnectError> {
        if let Some(provider) = self.provider(spec) {
            return provider.handler(spec, &self.catalog);
        }
        match self.catalog.connect_with(spec, ConnectOptions::new()) {
            Ok(model) => Ok(adapted(spec, model)),
            Err(error @ ConnectError::MissingKey { .. }) => {
                self.signed_in(spec).unwrap_or(Err(error))
            }
            Err(error) => Err(error),
        }
    }

    /// The plan a sign-in to which serves `spec`, if any.
    pub fn plan(&self, spec: &ModelSpec) -> Option<&'static str> {
        self.sign_ins.iter().find_map(|sign_in| sign_in.plan(spec))
    }

    /// Catalog models that call tools and can be reached through a
    /// registered provider, the environment or a sign-in: first those of
    /// providers registered, with a key set or signed in, then those of
    /// providers that need none (local servers such as Ollama), which a
    /// picker can mark "no key needed". The check builds the handler
    /// exactly as a request would, once per provider.
    pub fn reachable(&self) -> Vec<&ModelSpec> {
        let mut usable: HashMap<ProviderId, bool> = HashMap::new();
        let mut models: Vec<&ModelSpec> = self
            .catalog
            .iter()
            .filter(|spec| spec.tools)
            .filter(|spec| {
                *usable
                    .entry(spec.provider)
                    .or_insert_with(|| match self.provider(spec) {
                        Some(provider) => provider.handler(spec, &self.catalog).is_ok(),
                        None => {
                            self.catalog
                                .connect_with(*spec, ConnectOptions::new())
                                .is_ok()
                                || self.signed_in(spec).is_some_and(|handler| handler.is_ok())
                        }
                    })
            })
            .collect();
        // Stable: catalog order within each group.
        models.sort_by_key(|spec| {
            self.provider(spec).is_none() && !spec.provider.requires_credential()
        });
        models
    }
}

#[cfg(test)]
mod tests;
