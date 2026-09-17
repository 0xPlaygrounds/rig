//! Provider bindings as data: a [`ProviderBinding`] component says *which*
//! provider serves a key — a [`ProviderRef`], a label, a credential
//! *reference* — and nothing executable. A scene saves it beside the
//! handler's [`Bound`]; loading it spawns the same data and makes no network
//! call and no credential lookup. The executable half is built later, on the
//! host's word: [`materialize_bindings`] reads the host-installed
//! [`Materializer`] (a credential resolver and a transport factory — rig-ecs
//! reads no environment variable itself), builds the provider's completion
//! wire for every binding that nothing serves yet, binds it to the host's
//! transport, and registers a `CompletionAdapter` under the binding's key
//! through the same [`Handlers`] API a hand-registered adapter goes through —
//! so the bound descriptor, and with it the policy hash, is the one a
//! hand-registered adapter would produce.
//!
//! The provider vocabulary is rig-core's
//! ([`providers::registry`](rig_core::providers::registry)), not this
//! crate's: a reference either names a registered selection
//! (`deepseek/openai:deepseek-chat`, whose configuration is the registry's
//! preset) or carries an explicit
//! [`rig_core::providers::registry::ProviderConfig`] with its
//! own host, route and typed options. Everything a binding can express is
//! therefore something a provider configuration already expresses; there is
//! no ECS provider enum, no dialect string and no untyped option bag.
//!
//! Secrets never enter the world: the component holds a [`CredentialRef`]
//! (a name — an environment variable, a key id in the host's vault), the
//! resolver returns a [`Secret`] whose `Debug` and serialized form are
//! redacted, and the secret lives only inside the bound wire.

use bevy_ecs::prelude::*;
use bevy_reflect::Reflect;
use rig_core::{
    effect::{HandlerDescriptor, HandlerKey},
    http_client::BoxedHttpClient,
    providers::registry::{ProviderConfig, ProviderRef, RefError},
    serve::ErasedHandler,
};
use serde::{Deserialize, Serialize};

use super::handlers::{Bound, Handler, Handlers};

/// A resolved credential: what the host's resolver returns and the bound
/// wire holds. rig-core's own, because a wire's credential and a binding's
/// resolved credential are the same thing — it goes straight into the
/// provider config, and its `Debug` and `Serialize` are redacted there for
/// the same reason they are here.
pub use rig_core::wire::Secret;

/// A reference to a credential the host resolves — an environment variable
/// name, a vault key id, a label the host's resolver knows. Never the
/// secret: the reference is saved verbatim in scenes and printed verbatim
/// in diagnostics, so whatever it names must be safe to print.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Reflect)]
#[serde(transparent)]
pub struct CredentialRef(pub String);

impl CredentialRef {
    /// A reference by name.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// The name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for CredentialRef {
    fn from(name: &str) -> Self {
        Self(name.to_owned())
    }
}

impl std::fmt::Display for CredentialRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The data half of a provider-served handler: which provider serves the
/// key. Lives on the handler entity beside [`Bound`] once materialized; on
/// its own (a host spawned it, or a scene saved it before it was
/// materialized) until then. A scene saves it with the bound descriptor,
/// and a load spawns exactly that — the key resolves for the scene's links
/// as any bound key does, and nothing is served until the host
/// materializes.
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct ProviderBinding {
    /// The key the built handler serves (`Bound.key` once materialized).
    #[reflect(remote = crate::bus::reflect::HandlerKeyReflect)]
    pub key: HandlerKey,
    /// Which provider and model. A registered selection writes as its
    /// canonical `vendor/format:model` string; an explicit configuration
    /// writes as an object and keeps every host, route and option it names.
    #[reflect(remote = crate::bus::reflect::ProviderRefReflect)]
    pub provider: ProviderRef,
    /// The adapter's label: the `ModelRef` the bound descriptor advertises
    /// (`CompletionAdapter::new(label, model)`), which is what the log's
    /// header and the policy hash name. Defaults to the reference's model.
    pub label: String,
    /// Which credential the host's resolver hands the provider config. A
    /// name, never a secret.
    pub credential: CredentialRef,
}

impl ProviderBinding {
    /// A binding of `key` to `provider`, labelled by the reference's model.
    pub fn new(
        key: impl Into<HandlerKey>,
        provider: ProviderRef,
        credential: impl Into<CredentialRef>,
    ) -> Self {
        Self {
            key: key.into(),
            label: provider.model.clone(),
            provider,
            credential: credential.into(),
        }
    }

    /// [`Self::new`] from a reference to parse — `vendor[/format]:model`,
    /// shorthand accepted.
    pub fn parse(
        key: impl Into<HandlerKey>,
        reference: &str,
        credential: impl Into<CredentialRef>,
    ) -> Result<Self, RefError> {
        Ok(Self::new(key, ProviderRef::parse(reference)?, credential))
    }

    /// [`Self::new`] from an explicit configuration and a model: a host,
    /// a route or a typed option the registry's preset does not name.
    pub fn configured(
        key: impl Into<HandlerKey>,
        config: ProviderConfig,
        model: impl Into<String>,
        credential: impl Into<CredentialRef>,
    ) -> Self {
        Self::new(key, ProviderRef::configured(config, model), credential)
    }

    /// With the adapter's label.
    pub fn labelled(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// The provider's model identifier.
    pub fn model(&self) -> &str {
        &self.provider.model
    }
}

/// Why a materialization did not happen. Every variant is deterministic
/// for a given world and resolver; none carries a secret.
///
/// There is no variant for a malformed provider selection or a wrong typed
/// option: both are refused where the value is defined — by
/// [`ProviderRef::parse`] or by the deserializer — so nothing that reaches
/// here can name a provider this build does not have.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MaterializeError {
    /// The world has no [`Materializer`].
    #[error("the world has no `Materializer`: install one before materializing bindings")]
    NoMaterializer,
    /// The resolver had nothing for the reference.
    #[error("`{key}`: the credential `{credential}` did not resolve: {detail}")]
    MissingCredential {
        /// The binding's key.
        key: HandlerKey,
        /// The reference that did not resolve.
        credential: CredentialRef,
        /// The resolver's reason.
        detail: String,
    },
    /// Two binding entities name one key.
    #[error("`{key}` is bound by two `ProviderBinding` entities")]
    DuplicateKey {
        /// The key.
        key: HandlerKey,
    },
    /// The binding sits beside a `Bound` of another key.
    #[error("the binding for `{key}` sits on the entity bound to `{bound}`")]
    KeyMismatch {
        /// The binding's key.
        key: HandlerKey,
        /// The entity's `Bound.key`.
        bound: HandlerKey,
    },
    /// The client the binding builds describes itself differently from the
    /// descriptor the scene saved: the model's capabilities changed, or the
    /// label did.
    #[error(
        "`{key}`: the built adapter's descriptor is not the bound one (saved {saved}, built {built})"
    )]
    DescriptorDrift {
        /// The key.
        key: HandlerKey,
        /// The saved descriptor, as JSON.
        saved: String,
        /// The built descriptor, as JSON.
        built: String,
    },
    /// The bus refused the registration.
    #[error("`{key}`: the bus refused the handler: {detail}")]
    Register {
        /// The key.
        key: HandlerKey,
        /// The bus's reason.
        detail: String,
    },
}

/// What one [`materialize_bindings`] call did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MaterializeReport {
    /// Bindings now served by a freshly built client, by key.
    pub materialized: Vec<HandlerKey>,
    /// Bindings whose key something already served — a handler the host
    /// registered by hand, or an earlier materialization — left as they
    /// were: the existing handler wins.
    pub kept: Vec<HandlerKey>,
}

type Credentials = dyn Fn(&CredentialRef) -> Result<Secret, String> + Send + Sync;
type Transport = dyn Fn() -> BoxedHttpClient + Send + Sync;
type Serving = dyn Fn(ErasedHandler) -> ErasedHandler + Send + Sync;

/// How bindings become handlers: the host's credential resolver and
/// transport factory, installed as a resource. rig-ecs never reads an
/// environment variable or picks a transport itself; a host that wants
/// `std::env` writes a resolver that reads it.
#[derive(Resource)]
pub struct Materializer {
    credentials: Box<Credentials>,
    transport: Box<Transport>,
    serving: Box<Serving>,
}

impl Materializer {
    /// A materializer over `credentials` (a reference to its secret, or the
    /// reason it did not resolve — reported as
    /// [`MaterializeError::MissingCredential`] under the binding's key) and
    /// `transport` (one fresh transport handle per wire bound; a factory
    /// returning clones of one handle shares it).
    pub fn new(
        credentials: impl Fn(&CredentialRef) -> Result<Secret, String> + Send + Sync + 'static,
        transport: impl Fn() -> BoxedHttpClient + Send + Sync + 'static,
    ) -> Self {
        Self {
            credentials: Box::new(credentials),
            transport: Box::new(transport),
            serving: Box::new(|handler| handler),
        }
    }

    /// How a built adapter is served: `wrap` sees every adapter before it
    /// is registered — a host whose transport needs a runtime context on
    /// each poll wraps here. The wrapper must keep the descriptor.
    pub fn serving(
        mut self,
        wrap: impl Fn(ErasedHandler) -> ErasedHandler + Send + Sync + 'static,
    ) -> Self {
        self.serving = Box::new(wrap);
        self
    }

    /// Resolve a reference: the secret, or the resolver's reason.
    pub fn resolve(&self, credential: &CredentialRef) -> Result<Secret, String> {
        (self.credentials)(credential)
    }

    /// A transport handle.
    pub fn transport(&self) -> BoxedHttpClient {
        (self.transport)()
    }

    /// Build the adapter for `binding`: the reference's configuration with
    /// the resolved credential, its completion wire bound to a fresh
    /// transport, the `CompletionAdapter` under the binding's label, wrapped
    /// as the host serves it. What [`materialize_bindings`] registers; a host
    /// can also build one to register itself.
    ///
    /// Resolving the credential is the only step that can fail: a provider
    /// configuration is data, and binding it to a socket is a struct
    /// literal.
    pub fn build(&self, binding: &ProviderBinding) -> Result<ErasedHandler, MaterializeError> {
        let secret = self.resolve(&binding.credential).map_err(|detail| {
            MaterializeError::MissingCredential {
                key: binding.key.clone(),
                credential: binding.credential.clone(),
                detail,
            }
        })?;
        let handler = binding.provider.config(secret).completion_handler(
            &binding.label,
            binding.model(),
            self.transport(),
        );
        Ok((self.serving)(handler))
    }
}

/// The descriptor `handler` is bound with under `key`: what
/// `Handlers::register_erased` records.
fn bound_descriptor(key: &HandlerKey, handler: &ErasedHandler) -> HandlerDescriptor {
    let described = handler.descriptor();
    HandlerDescriptor {
        key: key.clone(),
        family: described.family,
        layers: described.layers,
    }
}

/// Every `(entity, key)` a `Bound` in the world holds.
///
/// Archetype-filtered rather than a walk over every entity: a rig-ecs world
/// holds an entity per message part, effect and run, and this runs on every
/// materialization pass. `None` means no entity has ever carried a `Bound`,
/// which is the empty answer.
pub(crate) fn bound_keys(world: &World) -> Vec<(Entity, HandlerKey)> {
    world
        .try_query::<(Entity, &Bound)>()
        .map(|mut bound| {
            bound
                .iter(world)
                .map(|(entity, bound)| (entity, bound.key.clone()))
                .collect()
        })
        .unwrap_or_default()
}

/// Whether this entity is itself serving: a handler the host registered on
/// it by hand, or an earlier materialization of the binding it carries.
///
/// One half of the test that decides `MaterializeReport::kept`; a binding's
/// own `Bound` is not it — only a live `Handler` is.
pub(crate) fn serves_itself(world: &World, entity: Entity) -> bool {
    world.get::<Handler>(entity).is_some()
}

/// Whether `key` is already served or held on an entity other than
/// `entity`'s own: the existing handler a materialization defers to.
///
/// The other half. A binding's own stale `Bound` — what a checkpoint load
/// leaves — is neither, which is the point of the entity comparison: the
/// binding is still the thing that will build the client.
pub(crate) fn served_elsewhere(
    bound: &[(Entity, HandlerKey)],
    entity: Entity,
    key: &HandlerKey,
) -> bool {
    bound
        .iter()
        .any(|(other, held)| *other != entity && held == key)
}

/// One binding the world holds, as [`materialize_bindings`] sees it.
struct Pending {
    entity: Entity,
    binding: ProviderBinding,
    bound: Option<Bound>,
    /// The key is served, or bound, on some entity: this one (a hand
    /// registration, an earlier materialization) or another.
    taken: bool,
}

/// Materialize every binding nothing serves yet: resolve its credential,
/// build its client, register a `CompletionAdapter` under its key through
/// [`Handlers`] — on the binding's own entity, so the binding and its
/// `Bound` share one entity. All or nothing: every credential is resolved,
/// every client built and every registration checked (the key free or the
/// binding's own, the descriptor the saved one) before the first
/// registration, so a refusal leaves the world's handlers as they were.
///
/// The host calls this after a scene load and before the first dispatch;
/// [`materialize`] is the same as a system.
///
/// | the world holds | what happens |
/// |---|---|
/// | a binding on an entity nothing serves, no `Bound` | built, `Bound` inserted, served: `materialized` |
/// | a binding beside a `Bound` (a scene load) nothing serves | built; the built descriptor must equal the saved one (else `DescriptorDrift`); served: `materialized` |
/// | a binding beside a `Bound` something serves | left alone: `kept` — the existing handler wins |
/// | a binding whose key another entity serves or holds in its `Bound` — beside a `Bound` of its own or not | left alone: `kept` — the existing handler wins, the binding's own `Bound` untouched |
/// | two binding entities with one key | `DuplicateKey` |
/// | a binding beside a `Bound` of another key | `KeyMismatch` |
/// | no `Materializer` | `NoMaterializer` |
/// | a reference the resolver refuses | `MissingCredential`, nothing registered |
pub fn materialize_bindings(world: &mut World) -> Result<MaterializeReport, MaterializeError> {
    if !world.contains_resource::<Materializer>() {
        return Err(MaterializeError::NoMaterializer);
    }
    let mut pending: Vec<Pending> = world
        .query::<(Entity, &ProviderBinding, Option<&Bound>)>()
        .iter(world)
        .map(|(entity, binding, bound)| Pending {
            entity,
            binding: binding.clone(),
            bound: bound.cloned(),
            taken: false,
        })
        .collect();
    let bound = bound_keys(world);
    for item in &mut pending {
        // Served on this entity (a hand registration, an earlier
        // materialization), or bound on another — served there (what
        // `Handlers::bind` would re-serve instead of this entity) or
        // not: the existing handler wins, whether or not this entity
        // carries a `Bound` of its own.
        item.taken = serves_itself(world, item.entity)
            || served_elsewhere(&bound, item.entity, &item.binding.key);
    }
    pending.sort_by(|a, b| a.binding.key.cmp(&b.binding.key));
    let mut report = MaterializeReport::default();
    let mut todo: Vec<&Pending> = Vec::new();
    for (index, item) in pending.iter().enumerate() {
        let key = &item.binding.key;
        if pending
            .get(index + 1)
            .is_some_and(|next| &next.binding.key == key)
        {
            return Err(MaterializeError::DuplicateKey { key: key.clone() });
        }
        if let Some(bound) = &item.bound
            && &bound.key != key
        {
            return Err(MaterializeError::KeyMismatch {
                key: key.clone(),
                bound: bound.key.clone(),
            });
        }
        if item.taken {
            report.kept.push(key.clone());
            continue;
        }
        todo.push(item);
    }
    // Build everything first: a refusal here registers nothing.
    let mut built: Vec<(Entity, HandlerKey, HandlerDescriptor, ErasedHandler)> = Vec::new();
    {
        let materializer = world.resource::<Materializer>();
        for item in &todo {
            let handler = materializer.build(&item.binding)?;
            let descriptor = bound_descriptor(&item.binding.key, &handler);
            if let Some(bound) = &item.bound
                && bound.descriptor != descriptor
            {
                return Err(MaterializeError::DescriptorDrift {
                    key: item.binding.key.clone(),
                    saved: serde_json::to_string(&bound.descriptor).unwrap_or_default(),
                    built: serde_json::to_string(&descriptor).unwrap_or_default(),
                });
            }
            built.push((item.entity, item.binding.key.clone(), descriptor, handler));
        }
    }
    // Every registration is now known to succeed: `Handlers::bind` refuses
    // only a key bound to another family, and every key here is either
    // free or bound on the binding's own entity with the descriptor about
    // to be registered. The binding's own entity becomes the handler
    // entity: `Bound` first, so the registry finds it and re-serves it
    // rather than spawning.
    let plan: Vec<(Entity, HandlerKey)> = built
        .iter()
        .map(|(entity, key, _, _)| (*entity, key.clone()))
        .collect();
    for (entity, key, descriptor, _) in &built {
        world.entity_mut(*entity).insert(Bound {
            key: key.clone(),
            descriptor: descriptor.clone(),
        });
    }
    let registered = Handlers::with(world, |handlers| {
        for (entity, key, _, handler) in built {
            let served = handlers
                .register_erased(key.clone(), handler)
                .map_err(|error| (key, error))?;
            debug_assert_eq!(served, entity, "the binding's entity is the handler's");
        }
        Ok::<(), (HandlerKey, rig_core::error::ErrorReport)>(())
    });
    let (key, error) = match registered {
        Ok(Ok(())) => {
            report
                .materialized
                .extend(plan.into_iter().map(|(_, key)| key));
            return Ok(report);
        }
        Ok(Err(refused)) => refused,
        Err(error) => (
            plan.first()
                .map(|(_, key)| key.clone())
                .unwrap_or_else(|| HandlerKey::from("")),
            error,
        ),
    };
    // Unreachable by the check above (`Handlers::with` cannot fail once the
    // table was read, and `bind` cannot refuse a free or same-entity key);
    // kept so a future `bind` rule cannot leave a half-registered world:
    // what registered before the refusal is unserved again, and every
    // `Bound` inserted here is taken out or put back as the scene saved it.
    for (entity, _) in &plan {
        world.entity_mut(*entity).remove::<Handler>();
        let saved = todo
            .iter()
            .find(|item| item.entity == *entity)
            .and_then(|item| item.bound.clone());
        let mut handler = world.entity_mut(*entity);
        match saved {
            Some(bound) => {
                handler.insert(bound);
            }
            None => {
                handler.remove::<Bound>();
            }
        }
    }
    world.flush();
    Err(MaterializeError::Register {
        key,
        detail: error.message,
    })
}

/// [`materialize_bindings`] as an exclusive system: what a host schedules
/// (once, or before every pass — a materialized world has nothing left to
/// do). A refusal is logged and left in [`MaterializeFailed`] for the host
/// to read; a success removes it.
pub fn materialize(world: &mut World) {
    match materialize_bindings(world) {
        Ok(_) => {
            world.remove_resource::<MaterializeFailed>();
        }
        Err(error) => {
            log::error!("provider bindings did not materialize: {error}");
            world.insert_resource(MaterializeFailed(error));
        }
    }
}

/// Why the last [`materialize`] pass refused.
#[derive(Resource, Debug, Clone, PartialEq, Eq)]
pub struct MaterializeFailed(pub MaterializeError);
