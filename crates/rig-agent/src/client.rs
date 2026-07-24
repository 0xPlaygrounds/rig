//! Classic runtime construction extensions for portable completion clients and models.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{agent::AgentBuilder, extractor::ExtractorBuilder};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

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
    fn agent(&self, model: impl Into<String>) -> AgentBuilder<Self::CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    /// Construct a classic typed extractor builder for `model`.
    fn extractor<T>(&self, model: impl Into<String>) -> ExtractorBuilder<Self::CompletionModel, T>
    where
        T: JsonSchema
            + for<'de> Deserialize<'de>
            + Serialize
            + WasmCompatSend
            + WasmCompatSync
            + 'static,
    {
        ExtractorBuilder::new(self.completion_model(model))
    }
}

impl<C: rig_core::client::completion::CompletionClient> AgentClientExt for C {}

/// Adds classic agent construction to every portable completion model.
pub trait AgentModelExt: rig_core::completion::CompletionModel + Sized {
    /// Convert this model into a classic agent builder.
    fn into_agent_builder(self) -> AgentBuilder<Self> {
        AgentBuilder::new(self)
    }
}

impl<M> AgentModelExt for M where M: rig_core::completion::CompletionModel {}
