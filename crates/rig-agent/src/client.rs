//! Classic runtime construction extensions for portable completion clients and models.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{agent::AgentBuilder, extractor::ExtractorBuilder};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// Classic-runtime construction sugar layered on any portable completion client.
///
/// Inherits `completion_model` / `CompletionModel` from its supertrait
/// [`rig_core::client::completion::CompletionClient`] and adds the classic
/// runtime's `agent` and `extractor` builders. Because the supertrait provides
/// `completion_model`, there is nothing to keep in sync if the portable trait
/// grows a method.
///
/// Provider authors implement the portable
/// [`rig_core::client::completion::CompletionClient`]; this extension trait is
/// blanket-implemented for every type that does. Bring both traits into scope
/// (e.g. via `use rig::prelude::*;`) for the full `completion_model` + `agent` +
/// `extractor` surface.
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
