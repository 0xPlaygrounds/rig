//! Classic runtime construction extensions for portable completion clients and models.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{agent::AgentBuilder, extractor::ExtractorBuilder};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// The classic runtime's completion client surface.
///
/// This mirrors the portable provider trait
/// [`rig_core::client::completion::CompletionClient`] — forwarding
/// `completion_model` — and adds the classic runtime's `agent` and `extractor`
/// builders. A single `use rig::client::CompletionClient` therefore exposes the
/// same surface it did before the runtime split, so `client.completion_model(m)`
/// and `client.agent(m)` both keep working without a second import.
///
/// Provider authors still implement the portable
/// [`rig_core::client::completion::CompletionClient`]; this trait is blanket-
/// implemented for every type that does, and forwards to it.
pub trait CompletionClient {
    /// The completion model produced by this client (the provider's model type).
    type CompletionModel: rig_core::completion::CompletionModel<Client = Self>;

    /// Create a completion model for `model`.
    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel;

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

// This forward mirrors the portable `rig_core` `CompletionClient` surface (its
// associated type + `completion_model`). If the portable trait grows a method,
// add a matching forward here so a single `rig::client::CompletionClient` import
// keeps exposing the full portable surface.
impl<C> CompletionClient for C
where
    C: rig_core::client::completion::CompletionClient,
{
    type CompletionModel = <C as rig_core::client::completion::CompletionClient>::CompletionModel;

    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel {
        <C as rig_core::client::completion::CompletionClient>::completion_model(self, model)
    }
}

/// Adds classic agent construction to every portable completion model.
pub trait AgentModelExt: rig_core::completion::CompletionModel + Sized {
    /// Convert this model into a classic agent builder.
    fn into_agent_builder(self) -> AgentBuilder<Self> {
        AgentBuilder::new(self)
    }
}

impl<M> AgentModelExt for M where M: rig_core::completion::CompletionModel {}
