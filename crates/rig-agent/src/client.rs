//! Classic runtime construction extensions for portable completion clients and models.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    agent::AgentBuilder,
    extractor::ExtractorBuilder,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

pub use rig_core::client::completion::CompletionClient;

/// Adds classic agent and extractor construction to every completion client.
pub trait AgentClientExt: CompletionClient {
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

impl<C> AgentClientExt for C where C: CompletionClient {}

/// Adds classic agent construction to every portable completion model.
pub trait AgentModelExt: rig_core::completion::CompletionModel + Sized {
    /// Convert this model into a classic agent builder.
    fn into_agent_builder(self) -> AgentBuilder<Self> {
        AgentBuilder::new(self)
    }
}

impl<M> AgentModelExt for M where M: rig_core::completion::CompletionModel {}
