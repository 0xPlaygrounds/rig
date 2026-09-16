//! Classic runtime construction extensions for portable completion clients and models.

use schemars::JsonSchema;
use serde::Serialize;

use crate::{agent::AgentBuilder, extractor::ExtractorBuilder};
use rig_core::driver::{Bound, Socket};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};
use rig_core::wire::HasCompletion;

/// Classic-runtime construction sugar layered on any portable completion client.
///
/// Builds on `completion_model` / `CompletionModel` from its supertrait bound
/// [`rig_core::client::completion::CompletionClient`] and adds the classic
/// runtime's `agent` and `extractor` builders. The supertrait bound is what lets
/// the default bodies call `self.completion_model(..)`, so nothing needs
/// re-forwarding if the portable trait grows a method.
///
/// Provider authors implement the portable
/// [`rig_core::client::completion::CompletionClient`]; this extension trait is
/// blanket-implemented for every type that does. Callers need *both* traits in
/// scope to use the full surface — importing `AgentClientExt` alone does not
/// bring `completion_model` into method-resolution scope, since that method
/// belongs to the supertrait. `use rig::prelude::*;` brings both in at once for
/// the full `completion_model` + `agent` + `extractor` surface.
pub trait AgentClientExt: rig_core::client::completion::CompletionClient {
    /// Construct a classic agent builder for `model`.
    fn agent(&self, model: impl Into<String>) -> AgentBuilder
    where
        Self::CompletionModel: 'static,
    {
        AgentBuilder::new(self.completion_model(model))
    }

    /// Construct a classic typed extractor builder for `model`.
    fn extractor<T>(&self, model: impl Into<String>) -> ExtractorBuilder<T>
    where
        T: JsonSchema
            + serde::de::DeserializeOwned
            + Serialize
            + WasmCompatSend
            + WasmCompatSync
            + 'static,
        Self::CompletionModel: 'static,
    {
        ExtractorBuilder::new(self.completion_model(model))
    }
}

impl<C: rig_core::client::completion::CompletionClient> AgentClientExt for C {}

/// Classic-runtime construction sugar on a bound provider configuration.
///
/// A provider config that names its completion wire
/// ([`rig_core::wire::HasCompletion`]), bundled with the socket it speaks
/// over ([`Bound`]), is already everything an agent needs: one impl serves
/// every provider, and no provider contributes an `agent` forwarder.
///
/// This is a trait of its own rather than a second impl of
/// [`AgentClientExt`] because the two receivers overlap as far as the
/// compiler can tell: one blanket impl over
/// [`CompletionClient`](rig_core::client::completion::CompletionClient) and
/// one over `Bound<P, H>` on the same trait would need `Bound<P, H>:
/// !CompletionClient` to be provable, and negative bounds are not.
///
/// ```ignore
/// use rig_agent::prelude::*;
/// use rig_core::providers::anthropic::wire::Anthropic;
///
/// let provider = Anthropic::from_env()?.bound()?;
/// let agent = provider.agent("claude-sonnet-4-5").preamble("Be brief.").build();
/// ```
pub trait AgentProviderExt {
    /// Construct a classic agent builder for `model`.
    fn agent(&self, model: impl Into<String>) -> AgentBuilder;

    /// Construct a classic typed extractor builder for `model`.
    fn extractor<T>(&self, model: impl Into<String>) -> ExtractorBuilder<T>
    where
        T: JsonSchema
            + serde::de::DeserializeOwned
            + Serialize
            + WasmCompatSend
            + WasmCompatSync
            + 'static;
}

impl<P, H> AgentProviderExt for Bound<P, H>
where
    P: HasCompletion,
    H: Socket,
{
    fn agent(&self, model: impl Into<String>) -> AgentBuilder {
        AgentBuilder::new(self.completion(model))
    }

    fn extractor<T>(&self, model: impl Into<String>) -> ExtractorBuilder<T>
    where
        T: JsonSchema
            + serde::de::DeserializeOwned
            + Serialize
            + WasmCompatSend
            + WasmCompatSync
            + 'static,
    {
        ExtractorBuilder::new(self.completion(model))
    }
}

/// Adds classic agent construction to every portable completion model.
pub trait AgentModelExt: rig_core::completion::CompletionModel + Sized {
    /// Convert this model into a classic agent builder.
    fn into_agent_builder(self) -> AgentBuilder
    where
        Self: 'static,
    {
        AgentBuilder::new(self)
    }
}

impl<M> AgentModelExt for M where M: rig_core::completion::CompletionModel {}
