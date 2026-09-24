//! Classic runtime construction extensions for portable completion models.
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
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// Adds classic agent and extractor construction to every portable
/// completion model.
pub trait AgentModelExt: rig_core::completion::CompletionModel + Sized {
    /// Convert this model into a classic agent builder.
    fn into_agent_builder(self) -> AgentBuilder
    where
        Self: 'static,
    {
        AgentBuilder::new(self)
    }

    /// Convert this model into a classic typed extractor builder.
    fn into_extractor_builder<T>(self) -> ExtractorBuilder<T>
    where
        T: JsonSchema
            + serde::de::DeserializeOwned
            + Serialize
            + WasmCompatSend
            + WasmCompatSync
            + 'static,
        Self: 'static,
    {
        ExtractorBuilder::new(self)
    }
}

impl<M> AgentModelExt for M where M: rig_core::completion::CompletionModel {}
