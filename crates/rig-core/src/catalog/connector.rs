//! How a catalog's models are reached: through a provider registered for
//! their vendor, else with the environment's keys, else through a sign-in,
//! such as a subscription plan's. Every HTTP client a model is reached
//! through can be wrapped, such as to log or rotate keys.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use super::{Catalog, ModelSpec};
use crate::http_client::DynHttpClient;
use crate::operation::Completion;
use crate::providers::registry::{ConnectError, ConnectOptions, ProviderId};
use crate::serve::ErasedHandler;
use crate::serve::adapters::ModelAdapter;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// A way to reach catalog models no key is set for, such as a
/// subscription plan signed in to. A [`Connector`] asks it for a model only
/// when the provider registered for the model's vendor, else the
/// environment, has no key for it.
pub trait SignIn: WasmCompatSend + WasmCompatSync + 'static {
    /// The handler serving `spec` with the signed-in credential, or `None`
    /// when this sign-in does not serve `spec` now. A handler that sends
    /// HTTP sends through `transport`.
    fn handler(
        &self,
        spec: &ModelSpec,
        transport: &Transport,
    ) -> Option<Result<ErasedHandler, ConnectError>>;
    /// The name of the subscription plan through which this sign-in would
    /// serve `spec`, such as `"ChatGPT"`, whether or not the user is signed
    /// in; for model pickers.
    fn plan(&self, spec: &ModelSpec) -> Option<&'static str>;

    /// Whether the signed-in account offers `spec`. Asked only for a model
    /// this sign-in serves. Must answer at once without blocking; return
    /// [`ModelStatus::Checking`] while finding out.
    fn offers(&self, _spec: &ModelSpec, _transport: &Transport) -> ModelStatus {
        ModelStatus::Usable
    }
}

/// How a [`Connector`] reaches every model of one vendor, in place of the
/// registry's preset and the environment: a vendor declared at run time
/// ([`ProviderId::declare`]) on a gateway of its own, a built-in vendor at
/// another address, or a scripted model in a test.
///
/// [`ConnectOptions`] is one: the vendor's models connect with those
/// options (a base URL, a key, an HTTP client, a protocol family). A
/// closure from the spec, the catalog and the transport to a handler is
/// another.
pub trait Provide: WasmCompatSend + WasmCompatSync + 'static {
    /// The handler serving `spec`, a model of `catalog`. A handler that
    /// sends HTTP sends through `transport`: its shared client, or a client
    /// of its own that it wraps.
    fn handler(
        &self,
        spec: &ModelSpec,
        catalog: &Catalog,
        transport: &Transport,
    ) -> Result<ErasedHandler, ConnectError>;
}

impl<F> Provide for F
where
    F: Fn(&ModelSpec, &Catalog, &Transport) -> Result<ErasedHandler, ConnectError>
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
{
    fn handler(
        &self,
        spec: &ModelSpec,
        catalog: &Catalog,
        transport: &Transport,
    ) -> Result<ErasedHandler, ConnectError> {
        self(spec, catalog, transport)
    }
}

/// The vendor's models connect as [`Catalog::connect_with`] connects them
/// with these options, on their HTTP client, else the shared one, wrapped
/// by the transport.
impl Provide for ConnectOptions {
    fn handler(
        &self,
        spec: &ModelSpec,
        catalog: &Catalog,
        transport: &Transport,
    ) -> Result<ErasedHandler, ConnectError> {
        let mut options = self.clone();
        options.http = match options.http {
            Some(own) => Some(transport.wrap(own)),
            None => transport.shared(),
        };
        catalog
            .connect_with(spec, options)
            .map(|model| adapted(spec, model))
    }
}

/// `model`, which serves `spec`, as an effect handler labelled with the
/// spec's reference.
fn adapted(spec: &ModelSpec, model: crate::driver::DynModel<Completion>) -> ErasedHandler {
    ErasedHandler::new(ModelAdapter::<Completion>::new(spec.reference(), model))
}

/// One wrapper of a [`Transport`], `Send` and `Sync` outside browser wasm.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
type Wrapper = Arc<dyn Fn(DynHttpClient) -> DynHttpClient + Send + Sync>;
/// One wrapper of a [`Transport`], which may hold browser values.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
type Wrapper = Arc<dyn Fn(DynHttpClient) -> DynHttpClient>;

/// The wrappers a [`Connector`] applies to every HTTP client its models are
/// reached through, in the order added: the first is closest to the
/// network. Cloning is cheap.
///
/// ```
/// use rig_core::catalog::Transport;
/// use rig_core::http_client::DynHttpClient;
///
/// fn reach(transport: &Transport, own: DynHttpClient) -> DynHttpClient {
///     // A provider with a client of its own wraps it; one without takes
///     // the shared client, when the build has one.
///     transport.wrap(own)
/// }
/// ```
#[derive(Clone, Default)]
pub struct Transport {
    wrappers: Vec<Wrapper>,
}

impl Transport {
    /// `base` with every wrapper applied.
    pub fn wrap(&self, base: DynHttpClient) -> DynHttpClient {
        self.wrappers.iter().fold(base, |client, wrap| wrap(client))
    }

    /// The shared client, wrapped, when the build has one (the `reqwest`
    /// feature); `None` otherwise, such as on the web.
    pub fn shared(&self) -> Option<DynHttpClient> {
        #[cfg(feature = "reqwest")]
        {
            Some(self.wrap(rig_reqwest::shared()))
        }
        #[cfg(not(feature = "reqwest"))]
        {
            None
        }
    }
}

/// Shows how many wrappers there are.
impl fmt::Debug for Transport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Transport")
            .field("wrappers", &self.wrappers.len())
            .finish()
    }
}

/// Whether a [`Connector`] can reach a catalog model for an agent, and why
/// not, for a picker's note.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelStatus {
    /// It calls tools, connects, and any sign-in serving it offers it.
    Usable,
    /// The sign-in serving it is still finding out whether its account offers it.
    Checking,
    /// The account served by the sign-in does not offer it, with a reason for logs.
    NotOffered(String),
    /// It calls tools, but no key is set: the variables read, in order.
    NoKey(Vec<Cow<'static, str>>),
    /// Its provider needs no key, such as a local server, and nothing
    /// configures it: no provider is registered for it and the environment
    /// sets neither its key nor its base URL. It connects when named, at
    /// its default address, but nothing says a server runs there.
    Unconfigured,
    /// It does not call tools, so an agent cannot use it.
    NoTools,
    /// It calls tools but does not connect, for this reason.
    Unavailable(String),
}

/// The models an application can use and how each is reached: a catalog,
/// rig's built-in one by default, whose models connect through the
/// provider registered for their vendor ([`Self::provide`]), else with the
/// environment's keys, else through the first [`SignIn`] that serves them,
/// each on an HTTP client wrapped by its [`Transport`]. Cloning is cheap.
/// It is `Send` and `Sync` natively; on browser wasm its providers,
/// sign-ins and wrappers may hold JavaScript values instead.
#[derive(Clone)]
pub struct Connector {
    catalog: Catalog,
    providers: HashMap<&'static str, Arc<dyn Provide>>,
    sign_ins: Vec<Arc<dyn SignIn>>,
    transport: Transport,
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
            transport: Transport::default(),
        }
    }

    /// Wraps the HTTP client every model of this connector is reached
    /// through, outside the wrappers added before. Connections made before
    /// the call are not rebuilt.
    ///
    /// ```
    /// use rig_core::catalog::Connector;
    /// use rig_core::http_client::DynHttpClient;
    ///
    /// let mut connector = Connector::default();
    /// // A wrapper sees every request and response, such as to log them.
    /// connector.wrap_http(|inner: DynHttpClient| inner);
    /// ```
    pub fn wrap_http(
        &mut self,
        wrap: impl Fn(DynHttpClient) -> DynHttpClient + WasmCompatSend + WasmCompatSync + 'static,
    ) {
        self.transport.wrappers.push(Arc::new(wrap));
    }

    /// The transport this connector's models are reached through, for a
    /// host that builds a model of its own beside them.
    pub fn transport(&self) -> &Transport {
        &self.transport
    }

    /// Falls back to `sign_in` for a model the environment has no key for
    /// and no sign-in added earlier serves.
    pub fn add_sign_in(&mut self, sign_in: impl SignIn) {
        self.sign_ins.push(Arc::new(sign_in));
    }

    /// Reaches every model of `vendor` (any of its protocol families)
    /// through `provider`, before the environment. A model the provider
    /// has no key for ([`ConnectError::MissingKey`]) falls back to the
    /// sign-ins, as one the environment has no key for does. A later
    /// provider for the same vendor replaces this one: to wrap it instead,
    /// take it first with [`Self::provider`].
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

    /// The provider registered for `vendor`'s models, if any: what a
    /// provider that wraps it, such as one that changes the key, delegates
    /// to.
    pub fn provider(&self, vendor: ProviderId) -> Option<Arc<dyn Provide>> {
        self.providers.get(vendor.vendor()).cloned()
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

    /// The catalog model `reference` names, as [`Catalog::resolve`] finds
    /// it, and that model as an effect handler: from the provider
    /// registered for its vendor, else its provider's client built from the
    /// environment, on the shared HTTP client of the `reqwest` feature, or,
    /// without a key in either, the handler of a sign-in that serves it. The
    /// HTTP client is wrapped by [`Self::wrap_http`]'s wrappers.
    ///
    /// # Errors
    ///
    /// [`ConnectError::NotFound`] for a model the catalog does not list,
    /// what the registered provider reports, else what
    /// [`Catalog::connect_with`] reports, when no sign-in serves a model
    /// the provider or the environment has no key for.
    pub fn connect(
        &self,
        reference: &str,
    ) -> Result<(Arc<ModelSpec>, ErasedHandler), ConnectError> {
        let spec = self.catalog.resolve(reference)?.shared();
        let (handler, _) = self.handler(&spec);
        Ok((spec, handler?))
    }

    /// The handler serving `spec`, as [`Self::connect`] finds it, and the
    /// sign-in serving it when reached through one.
    fn handler(
        &self,
        spec: &ModelSpec,
    ) -> (Result<ErasedHandler, ConnectError>, Option<&dyn SignIn>) {
        let connected = match self.providers.get(spec.provider.vendor()) {
            Some(provider) => provider.handler(spec, &self.catalog, &self.transport),
            None => {
                let options = match self.transport.shared() {
                    Some(http) => ConnectOptions::new().http(http),
                    None => ConnectOptions::new(),
                };
                self.catalog
                    .connect_with(spec, options)
                    .map(|model| adapted(spec, model))
            }
        };
        match connected {
            Err(error @ ConnectError::MissingKey { .. }) => self
                .sign_ins
                .iter()
                .find_map(|sign_in| {
                    sign_in
                        .handler(spec, &self.transport)
                        .map(|handler| (handler, Some(sign_in.as_ref())))
                })
                .unwrap_or((Err(error), None)),
            other => (other, None),
        }
    }

    /// The plan of the first sign-in that would serve `spec`
    /// ([`SignIn::plan`]).
    pub fn plan(&self, spec: &ModelSpec) -> Option<&'static str> {
        self.sign_ins.iter().find_map(|sign_in| sign_in.plan(spec))
    }

    /// Whether `spec` can serve an agent now, and why not: it must call
    /// tools, connect as [`Self::connect`] would connect it, be offered by
    /// the account when served through a sign-in, and, when its provider
    /// needs no key, be configured ([`ModelStatus::Unconfigured`]).
    ///
    /// ```
    /// use rig_core::catalog::{Connector, ModelSpec, ModelStatus};
    /// use rig_core::providers::registry::ProviderId;
    ///
    /// let corp = ProviderId::declare("status-doc-corp")?;
    /// let connector = Connector::default();
    /// let chat = ModelSpec::new(corp, "chat");
    /// assert_eq!(connector.status(&chat), ModelStatus::NoTools);
    /// let agent = chat.with_tools(true);
    /// assert!(matches!(connector.status(&agent), ModelStatus::Unavailable(_)));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn status(&self, spec: &ModelSpec) -> ModelStatus {
        if !spec.tools {
            return ModelStatus::NoTools;
        }
        let (handler, sign_in) = self.handler(spec);
        if let Some(sign_in) = sign_in {
            let offered = sign_in.offers(spec, &self.transport);
            if offered != ModelStatus::Usable {
                return offered;
            }
        }
        let provider = spec.provider;
        match handler {
            Ok(_)
                if !provider.requires_credential()
                    && !self.is_provided(provider)
                    && !provider.configured_in_env() =>
            {
                ModelStatus::Unconfigured
            }
            Ok(_) => ModelStatus::Usable,
            Err(ConnectError::MissingKey { tried, .. }) => ModelStatus::NoKey(tried),
            Err(error) => ModelStatus::Unavailable(error.to_string()),
        }
    }

    /// Catalog models that call tools and can be reached through a
    /// registered provider, the environment or a sign-in: first those of
    /// providers registered, with a key set or signed in, then those of
    /// configured providers that need none (local servers such as Ollama),
    /// which a picker can mark "no key needed". Only [`ModelStatus::Usable`]
    /// models are reachable. Sign-in plans are checked per model; other models
    /// share one check per provider.
    pub fn reachable(&self) -> Vec<&ModelSpec> {
        let statuses = self.statuses().into_iter();
        let usable = statuses.filter(|(_, status)| *status == ModelStatus::Usable);
        usable.map(|(spec, _)| spec).collect()
    }

    /// Every catalog model with its [`ModelStatus`]: first the
    /// [`reachable`](Self::reachable) ones in that order, then the others in
    /// catalog order. Only [`ModelStatus::Usable`] is reachable. Models
    /// claimed by a sign-in's plan are checked individually; other models
    /// share one check per provider, unless the provider refused it for a
    /// reason of its own ([`ModelStatus::Unavailable`]), which may be that
    /// model's alone.
    pub fn statuses(&self) -> Vec<(&ModelSpec, ModelStatus)> {
        let mut checked: HashMap<ProviderId, ModelStatus> = HashMap::new();
        let mut statuses: Vec<(&ModelSpec, ModelStatus)> = self
            .catalog
            .iter()
            .map(|spec| {
                let status = match spec.tools {
                    false => ModelStatus::NoTools,
                    true if self.plan(spec).is_some() => self.status(spec),
                    true => match checked.get(&spec.provider) {
                        Some(status) => status.clone(),
                        None => {
                            let status = self.status(spec);
                            if !matches!(status, ModelStatus::Unavailable(_)) {
                                checked.insert(spec.provider, status.clone());
                            }
                            status
                        }
                    },
                };
                (spec, status)
            })
            .collect();
        // Stable: catalog order within each group.
        statuses.sort_by_key(|(spec, status)| match status {
            ModelStatus::Usable => {
                u8::from(!self.is_provided(spec.provider) && !spec.provider.requires_credential())
            }
            _ => 2,
        });
        statuses
    }
}

// On browser wasm a provider, a sign-in and an HTTP wrapper may hold values
// that are neither `Send` nor `Sync`, such as JavaScript handles.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
const _: fn(&mut Connector, ProviderId) = |connector, vendor| {
    use std::rc::Rc;

    struct PageSignIn {
        _page: Rc<()>,
    }
    impl SignIn for PageSignIn {
        fn handler(
            &self,
            _: &ModelSpec,
            _: &Transport,
        ) -> Option<Result<ErasedHandler, ConnectError>> {
            None
        }
        fn plan(&self, _: &ModelSpec) -> Option<&'static str> {
            None
        }
    }

    let page = Rc::new(());
    let wrapped = page.clone();
    connector.wrap_http(move |client| {
        let _ = &wrapped;
        client
    });
    let provided = page.clone();
    connector.provide(vendor, move |_: &ModelSpec, _: &Catalog, _: &Transport| {
        let _ = &provided;
        Err(ConnectError::Provider("not on this page".into()))
    });
    connector.add_sign_in(PageSignIn { _page: page });
};

#[cfg(test)]
mod tests;
