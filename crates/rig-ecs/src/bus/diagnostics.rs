//! What the world's provider bindings look like, without touching them.
//!
//! [`provider_diagnostics`] takes `&World`: it spawns nothing, resolves no
//! credential, builds no client and cannot materialize. It answers the two
//! questions a host asks when a dispatch reports `HandlerUnavailable` — *what
//! could this build serve*, and *what does this world actually hold*.
//!
//! It answers the second through
//! [`materialize_bindings`](super::materialize_bindings)' own predicates
//! rather than an interpretation of them: [`BindingReport::served`] and
//! [`BindingReport::served_elsewhere`] are the two halves of the test that
//! decides [`MaterializeReport::kept`](super::MaterializeReport::kept), and
//! [`BindingReport::kept`] composes them the same way, so a report and the
//! next materialization cannot read one world differently. In
//! particular a binding beside its own stale `Bound` — what a checkpoint
//! load leaves — is neither: it is the bound-but-unserved state a host is
//! looking at when a dispatch refuses.
//!
//! It deliberately reports less than it knows. A binding's
//! [`ProviderConfig`](rig_core::providers::registry::ProviderConfig) holds a
//! [`Secret`](rig_core::wire::Secret) and a host URL the host chose; the
//! report carries the canonical selection label, the model and the credential
//! *reference*, never the configuration and never the secret. The credential
//! guidance is the provider's documented environment variable, borrowed from
//! the dialect table rather than copied.

use bevy_ecs::prelude::*;
use rig_core::effect::HandlerKey;
use rig_core::providers::registry::ProviderId;

use super::binding::{
    CredentialRef, MaterializeFailed, ProviderBinding, bound_keys, served_elsewhere, serves_itself,
};

/// A registered provider selection this build can materialize.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredProvider {
    /// The selection, canonically spelled `vendor/format`.
    pub selection: String,
    /// What the provider documents about its credential.
    pub credential: CredentialGuidance,
}

/// Where a provider's credential comes from, and whether it must.
///
/// A hint is not a requirement: a local `llama-server` reads
/// `LLAMACPP_API_KEY` when it is set and serves fine when it is not, and
/// saying it "requires" one would send a host looking for a problem it does
/// not have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CredentialGuidance {
    /// The environment variable the provider documents. Borrowed: the
    /// dialect tables already own it.
    pub env: &'static str,
    /// Whether a request without a credential is refused by the provider.
    pub required: bool,
}

impl CredentialGuidance {
    /// What `id` documents.
    fn of(id: &ProviderId) -> Self {
        Self {
            env: id.api_key_env(),
            required: id.requires_credential(),
        }
    }
}

/// One binding the world holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingReport {
    /// The entity carrying the binding — its identity is preserved, so a
    /// caller can go back to the world and look at it.
    pub entity: Entity,
    /// The key the built handler would serve.
    pub key: HandlerKey,
    /// The provider's model identifier.
    pub model: String,
    /// The provider's canonical selection label, `vendor/format` — the
    /// registry spelling, for a registered reference and an explicit
    /// configuration alike. Never the configuration itself.
    pub selection: String,
    /// The label the bound descriptor advertises.
    pub label: String,
    /// The credential reference the host's resolver is asked for.
    pub credential: CredentialRef,
    /// What the provider documents about that credential.
    pub guidance: CredentialGuidance,
    /// Whether this binding's own entity already serves the key: a
    /// hand-registered handler that the binding later landed on, or an
    /// earlier materialization of this very binding.
    pub served: bool,
    /// Whether *another* entity holds this key — the existing handler a
    /// materialization would defer to, reporting the binding `kept`. A
    /// binding's own stale `Bound`, which is what a checkpoint load leaves,
    /// is not that.
    pub served_elsewhere: bool,
}

impl BindingReport {
    /// Whether the next materialization would leave this binding alone and
    /// report it `kept` — the same `served || served_elsewhere` test
    /// materialization makes.
    pub fn kept(&self) -> bool {
        self.served || self.served_elsewhere
    }
}

/// Every provider fact a world holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderDiagnostics {
    /// The selections this build registers, in registration order. This
    /// build's completion registry — not every provider in the workspace and
    /// not every operation a provider speaks.
    pub registered: Vec<RegisteredProvider>,
    /// The world's bindings, by key.
    pub bindings: Vec<BindingReport>,
    /// Why the last [`materialize`](super::materialize) pass refused, when
    /// the world tracks one ([`MaterializeFailed`]). Rendered, because the
    /// host's resolver wrote part of it: the text is as safe to print as
    /// whatever that resolver returned, which is the host's contract to keep,
    /// not one this report can promise.
    pub refusal: Option<String>,
}

/// Read the world's provider state.
///
/// Read-only by signature: `&World` cannot spawn, insert or materialize.
pub fn provider_diagnostics(world: &World) -> ProviderDiagnostics {
    let bound = bound_keys(world);
    let Some(mut query) = world.try_query::<(Entity, &ProviderBinding)>() else {
        // No entity has ever carried a binding.
        return ProviderDiagnostics {
            registered: registered(),
            bindings: Vec::new(),
            refusal: refusal(world),
        };
    };
    let mut bindings: Vec<BindingReport> = query
        .iter(world)
        .map(|(entity, binding)| {
            let id = binding.provider.id();
            BindingReport {
                entity,
                key: binding.key.clone(),
                model: binding.model().to_owned(),
                selection: id.to_string(),
                label: binding.label.clone(),
                credential: binding.credential.clone(),
                guidance: CredentialGuidance::of(&id),
                served: serves_itself(world, entity),
                served_elsewhere: served_elsewhere(&bound, entity, &binding.key),
            }
        })
        .collect();
    bindings.sort_by(|a, b| a.key.cmp(&b.key).then(a.entity.cmp(&b.entity)));
    ProviderDiagnostics {
        registered: registered(),
        bindings,
        refusal: refusal(world),
    }
}

/// Every selection this build registers, with its credential guidance.
fn registered() -> Vec<RegisteredProvider> {
    ProviderId::all()
        .map(|id| RegisteredProvider {
            selection: id.to_string(),
            credential: CredentialGuidance::of(&id),
        })
        .collect()
}

/// The last materialization refusal the world tracks, rendered.
fn refusal(world: &World) -> Option<String> {
    world
        .get_resource::<MaterializeFailed>()
        .map(|failed| failed.0.to_string())
}
