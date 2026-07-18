//! Runtime construction extensions for portable completion clients and models.

pub use rig_core::client::*;

use rig_core::{
    client::completion::CompletionClient as CoreCompletionClient,
    completion::CompletionModel,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{agent::AgentBuilder, extractor::ExtractorBuilder};

/// Adds classic-runtime constructors to every portable completion client.
pub trait AgentClientExt: CoreCompletionClient {
    /// Create a classic agent builder for `model`.
    fn agent(&self, model: impl Into<String>) -> AgentBuilder<Self::CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    /// Create a structured extractor builder for `model`.
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

impl<C> AgentClientExt for C where C: CoreCompletionClient {}

/// Adds classic-runtime construction to an already-created completion model.
pub trait AgentModelExt: CompletionModel + Sized {
    /// Consume this model and create a classic agent builder.
    fn into_agent_builder(self) -> AgentBuilder<Self> {
        AgentBuilder::new(self)
    }
}

impl<M> AgentModelExt for M where M: CompletionModel {}
