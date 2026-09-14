//! Provider bindings as data: a [`ProviderBinding`] component says *which*
//! provider client serves a key — kind, model, base URL, a credential
//! *reference* — and nothing executable. A scene saves it beside the
//! handler's [`Bound`]; loading it spawns the same data and makes no
//! network call and no credential lookup. The executable half is built
//! later, on the host's word: [`materialize_bindings`] reads the
//! host-installed [`Materializer`] (a credential resolver and a transport
//! factory — rig-ecs reads no environment variable itself), builds the
//! rig-core client for every binding that nothing serves yet, and registers
//! a `CompletionAdapter` under the binding's key through the same
//! [`Handlers`] API a hand-registered adapter goes through — so the bound
//! descriptor, and with it the policy hash, is the one a hand-registered
//! adapter would produce.
//!
//! Secrets never enter the world: the component holds a [`CredentialRef`]
//! (a name — an environment variable, a key id in the host's vault), the
//! resolver returns a [`Secret`] whose `Debug` is redacted, and the secret
//! lives only inside the built client.

use bevy_ecs::prelude::*;
use rig_core::{
    client::{CompletionClient, Provider},
    effect::{HandlerDescriptor, HandlerKey},
    http_client::BoxedHttpClient,
    markers::Missing,
    providers::{anthropic, deepseek, gemini, openai},
    serve::{ErasedHandler, adapters::CompletionAdapter},
};
use serde::{Deserialize, Serialize};

use super::handlers::{Bound, HandlerTable, Handlers};

/// Which rig-core provider client a binding builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    /// `rig_core::providers::anthropic` (the Messages API).
    Anthropic,
    /// `rig_core::providers::openai` over Chat Completions.
    OpenAiChat,
    /// `rig_core::providers::openai` over the Responses API.
    OpenAiResponses,
    /// `rig_core::providers::gemini` (GenerateContent).
    Gemini,
    /// `rig_core::providers::deepseek`.
    DeepSeek,
}

impl ProviderKind {
    /// The kind's serde spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAiChat => "openai_chat",
            Self::OpenAiResponses => "openai_responses",
            Self::Gemini => "gemini",
            Self::DeepSeek => "deepseek",
        }
    }
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A reference to a credential the host resolves — an environment variable
/// name, a vault key id, a label the host's resolver knows. Never the
/// secret: the reference is saved verbatim in scenes and printed verbatim
/// in diagnostics, so whatever it names must be safe to print.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
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

/// A resolved credential: what the host's resolver returns and the built
/// client consumes. `Debug` and `Display` are redacted; the only way out
/// is [`expose`](Self::expose), which the materializer calls once, inside
/// the client builder.
#[derive(Clone, PartialEq, Eq)]
pub struct Secret(String);

impl Secret {
    /// Wrap a resolved secret.
    pub fn new(secret: impl Into<String>) -> Self {
        Self(secret.into())
    }

    /// The secret, for the client builder and nothing else.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Secret(<redacted>)")
    }
}

/// The data half of a provider-served handler: which client serves the
/// key. Lives on the handler entity beside [`Bound`] once materialized; on
/// its own (a host spawned it, or a scene saved it before it was
/// materialized) until then. A scene saves it with the bound descriptor,
/// and a load spawns exactly that — the key resolves for the scene's links
/// as any bound key does, and nothing is served until the host
/// materializes.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct ProviderBinding {
    /// The key the built handler serves (`Bound.key` once materialized).
    #[cfg_attr(feature = "reflect", reflect(remote = crate::bus::reflect::HandlerKeyReflect))]
    pub key: HandlerKey,
    /// Which provider client.
    pub kind: ProviderKind,
    /// The provider's model id (`claude-haiku-4-5-20251001`, `gpt-4.1-mini`).
    pub model: String,
    /// The adapter's label: the `ModelRef` the bound descriptor advertises
    /// (`CompletionAdapter::new(label, model)`), which is what the log's
    /// header and the policy hash name. Defaults to `model`.
    pub label: String,
    /// The provider base URL; `None` is the provider's default.
    pub base_url: Option<String>,
    /// Which credential the host's resolver hands the client. A name, never
    /// a secret.
    pub credential: CredentialRef,
    /// Provider-specific client settings, by kind — see
    /// [`ProviderBinding::extra_params`](#extra-params). Unknown keys are
    /// refused at materialization.
    ///
    /// # Extra params
    ///
    /// | kind | keys |
    /// |---|---|
    /// | `anthropic` | `anthropic_version: string`, `anthropic_betas: [string]` |
    /// | `openai_responses` | `system_instructions_as_messages: bool` |
    /// | `openai_chat`, `gemini`, `deepseek` | none |
    #[cfg_attr(feature = "reflect", reflect(remote = crate::bus::reflect::ExtraParamsReflect))]
    pub extra_params: Option<serde_json::Value>,
}

impl ProviderBinding {
    /// A binding of `key` to `model` on `kind`, labelled by the model id,
    /// on the provider's default base URL, with no extra params.
    pub fn new(
        key: impl Into<HandlerKey>,
        kind: ProviderKind,
        model: impl Into<String>,
        credential: impl Into<CredentialRef>,
    ) -> Self {
        let model = model.into();
        Self {
            key: key.into(),
            kind,
            label: model.clone(),
            model,
            base_url: None,
            credential: credential.into(),
            extra_params: None,
        }
    }

    /// With the adapter's label.
    pub fn labelled(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// With a base URL.
    pub fn at(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// With provider-specific client settings.
    pub fn with_extra_params(mut self, params: serde_json::Value) -> Self {
        self.extra_params = Some(params);
        self
    }
}

/// Why a materialization did not happen. Every variant is deterministic
/// for a given world and resolver; none carries a secret.
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
    /// `extra_params` holds something the kind does not take.
    #[error("`{key}`: extra params for {kind}: {detail}")]
    ExtraParams {
        /// The key.
        key: HandlerKey,
        /// The kind.
        kind: ProviderKind,
        /// What was wrong.
        detail: String,
    },
    /// The rig-core client builder refused.
    #[error("`{key}`: the {kind} client did not build: {detail}")]
    Client {
        /// The key.
        key: HandlerKey,
        /// The kind.
        kind: ProviderKind,
        /// The builder's reason.
        detail: String,
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

/// How bindings become clients: the host's credential resolver and
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
    /// `transport` (one fresh transport handle per client built; a factory
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

    /// Build the adapter for `binding`: the client, the model, the
    /// `CompletionAdapter` under the binding's label, wrapped as the host
    /// serves it. What [`materialize_bindings`] registers; a host can also
    /// build one to register itself.
    pub fn build(&self, binding: &ProviderBinding) -> Result<ErasedHandler, MaterializeError> {
        validate_extra_params(binding)?;
        let secret = self.resolve(&binding.credential).map_err(|detail| {
            MaterializeError::MissingCredential {
                key: binding.key.clone(),
                credential: binding.credential.clone(),
                detail,
            }
        })?;
        let handler = build_adapter(binding, &secret, self.transport())?;
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

fn extra_object<'a>(
    binding: &'a ProviderBinding,
    allowed: &[&str],
) -> Result<Option<&'a serde_json::Map<String, serde_json::Value>>, MaterializeError> {
    let Some(params) = &binding.extra_params else {
        return Ok(None);
    };
    let Some(object) = params.as_object() else {
        return Err(MaterializeError::ExtraParams {
            key: binding.key.clone(),
            kind: binding.kind,
            detail: "not an object".to_owned(),
        });
    };
    if let Some(unknown) = object.keys().find(|k| !allowed.contains(&k.as_str())) {
        return Err(MaterializeError::ExtraParams {
            key: binding.key.clone(),
            kind: binding.kind,
            detail: format!("unknown key `{unknown}` (takes {allowed:?})"),
        });
    }
    Ok(Some(object))
}

/// The keys a kind's `extra_params` may hold.
fn allowed_extra_params(kind: ProviderKind) -> &'static [&'static str] {
    match kind {
        ProviderKind::Anthropic => &["anthropic_version", "anthropic_betas"],
        ProviderKind::OpenAiResponses => &["system_instructions_as_messages"],
        ProviderKind::OpenAiChat | ProviderKind::Gemini | ProviderKind::DeepSeek => &[],
    }
}

/// Refuse `extra_params` the kind does not take, before any credential is
/// resolved or transport built.
fn validate_extra_params(binding: &ProviderBinding) -> Result<(), MaterializeError> {
    extra_object(binding, allowed_extra_params(binding.kind)).map(|_| ())
}

fn client_error(binding: &ProviderBinding, error: impl std::fmt::Display) -> MaterializeError {
    MaterializeError::Client {
        key: binding.key.clone(),
        kind: binding.kind,
        detail: error.to_string(),
    }
}

fn builder<P>(
    binding: &ProviderBinding,
    secret: &Secret,
    transport: BoxedHttpClient,
) -> rig_core::client::ClientBuilder<P, BoxedHttpClient>
where
    P: Provider,
    P::ApiKey: From<String>,
{
    let mut builder = rig_core::client::Client::<P, Missing>::builder()
        .api_key(secret.expose().to_owned())
        .http_client(transport);
    if let Some(base_url) = &binding.base_url {
        builder = builder.base_url(base_url);
    }
    builder
}

fn build_adapter(
    binding: &ProviderBinding,
    secret: &Secret,
    transport: BoxedHttpClient,
) -> Result<ErasedHandler, MaterializeError> {
    let label = binding.label.as_str();
    let model = binding.model.as_str();
    Ok(match binding.kind {
        ProviderKind::Anthropic => {
            let params = extra_object(binding, allowed_extra_params(binding.kind))?;
            let mut builder = builder::<anthropic::client::Anthropic>(binding, secret, transport);
            if let Some(params) = params {
                if let Some(version) = params.get("anthropic_version") {
                    let version =
                        version
                            .as_str()
                            .ok_or_else(|| MaterializeError::ExtraParams {
                                key: binding.key.clone(),
                                kind: binding.kind,
                                detail: "`anthropic_version` is not a string".to_owned(),
                            })?;
                    builder = builder.anthropic_version(version);
                }
                if let Some(betas) = params.get("anthropic_betas") {
                    let betas: Vec<&str> = betas
                        .as_array()
                        .and_then(|items| items.iter().map(|v| v.as_str()).collect())
                        .ok_or_else(|| MaterializeError::ExtraParams {
                            key: binding.key.clone(),
                            kind: binding.kind,
                            detail: "`anthropic_betas` is not an array of strings".to_owned(),
                        })?;
                    builder = builder.anthropic_betas(&betas);
                }
            }
            let client = builder.build().map_err(|e| client_error(binding, e))?;
            ErasedHandler::new(CompletionAdapter::new(
                label,
                client.completion_model(model),
            ))
        }
        ProviderKind::OpenAiChat => {
            extra_object(binding, allowed_extra_params(binding.kind))?;
            let client = builder::<openai::client::OpenAICompletions>(binding, secret, transport)
                .build()
                .map_err(|e| client_error(binding, e))?;
            ErasedHandler::new(CompletionAdapter::new(
                label,
                client.completion_model(model),
            ))
        }
        ProviderKind::OpenAiResponses => {
            let params = extra_object(binding, allowed_extra_params(binding.kind))?;
            let mut client = builder::<openai::client::OpenAIResponses>(binding, secret, transport)
                .build()
                .map_err(|e| client_error(binding, e))?;
            if let Some(params) = params
                && let Some(flag) = params.get("system_instructions_as_messages")
            {
                match flag.as_bool() {
                    Some(true) => client = client.with_system_instructions_as_messages(),
                    Some(false) => {}
                    None => {
                        return Err(MaterializeError::ExtraParams {
                            key: binding.key.clone(),
                            kind: binding.kind,
                            detail: "`system_instructions_as_messages` is not a bool".to_owned(),
                        });
                    }
                }
            }
            ErasedHandler::new(CompletionAdapter::new(
                label,
                client.completion_model(model),
            ))
        }
        ProviderKind::Gemini => {
            extra_object(binding, allowed_extra_params(binding.kind))?;
            let client = builder::<gemini::client::Gemini>(binding, secret, transport)
                .build()
                .map_err(|e| client_error(binding, e))?;
            ErasedHandler::new(CompletionAdapter::new(
                label,
                client.completion_model(model),
            ))
        }
        ProviderKind::DeepSeek => {
            extra_object(binding, allowed_extra_params(binding.kind))?;
            let client = builder::<deepseek::DeepSeek>(binding, secret, transport)
                .build()
                .map_err(|e| client_error(binding, e))?;
            ErasedHandler::new(CompletionAdapter::new(
                label,
                client.completion_model(model),
            ))
        }
    })
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
/// | a reference the resolver refuses, a kind's params it does not take, a builder that refuses | that error, nothing registered |
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
    let bound: Vec<(Entity, HandlerKey)> = world
        .query::<(Entity, &Bound)>()
        .iter(world)
        .map(|(entity, bound)| (entity, bound.key.clone()))
        .collect();
    {
        let table = world.non_send::<HandlerTable>();
        for item in &mut pending {
            // Served on this entity (a hand registration, an earlier
            // materialization), or bound on another — served there (what
            // `Handlers::bind` would re-serve instead of this entity) or
            // not: the existing handler wins, whether or not this entity
            // carries a `Bound` of its own.
            item.taken = table.served(item.entity).is_some()
                || bound
                    .iter()
                    .any(|(entity, key)| *entity != item.entity && key == &item.binding.key);
        }
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
                .map_err(|error| Box::new((key, error)))?;
            debug_assert_eq!(served, entity, "the binding's entity is the handler's");
        }
        Ok::<(), Box<(HandlerKey, rig_core::error::ErrorReport)>>(())
    });
    let (key, error) = match registered {
        Ok(Ok(())) => {
            report
                .materialized
                .extend(plan.into_iter().map(|(_, key)| key));
            return Ok(report);
        }
        Ok(Err(refused)) => *refused,
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
        world.non_send_mut::<HandlerTable>().remove(*entity);
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
