//! Classic runtime construction extensions for portable completion providers and models.
//!
//! ```
//! use rig_agent::{Agent, client::AgentModelExt};
//! use rig_core::completion::CompletionModel;
//! fn assistant(model: impl CompletionModel + 'static) -> Agent {
//!     model.into_agent_builder().preamble("Be concise.").build()
//! }
//! ```

use schemars::JsonSchema;
use serde::Serialize;

use crate::{agent::AgentBuilder, extractor::ExtractorBuilder};
use rig_core::driver::CompletionProvider;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// Construct classic agents and typed extractors from any completion provider.
/// The provider must produce a model with a `'static` lifetime.
pub trait AgentProviderExt: CompletionProvider {
    /// Construct a classic agent builder for `model`.
    fn agent(&self, model: impl Into<String>) -> AgentBuilder
    where
        Self::Model: 'static,
    {
        AgentBuilder::new(self.completion(model))
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
        Self::Model: 'static,
    {
        ExtractorBuilder::new(self.completion(model))
    }
}

impl<P: CompletionProvider> AgentProviderExt for P {}

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
