//! Classic-runtime construction extensions.

use crate::{agent::AgentBuilder, extractor::ExtractorBuilder};
use rig_core::{
    client::completion::CompletionClient,
    completion::CompletionModel,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Adds classic agent construction to every completion-capable client.
pub trait AgentClientExt: CompletionClient {
    /// Create a classic agent builder for a model name.
    fn agent(&self, model: impl Into<String>) -> AgentBuilder<Self::CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    /// Create a structured-data extractor builder for a model name.
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

impl<C> AgentClientExt for C where C: CompletionClient + ?Sized {}

/// Adds classic agent construction to every completion model.
pub trait AgentModelExt: CompletionModel + Sized {
    /// Convert this model into a classic agent builder.
    fn into_agent_builder(self) -> AgentBuilder<Self> {
        AgentBuilder::new(self)
    }
}

impl<M> AgentModelExt for M where M: CompletionModel {}
