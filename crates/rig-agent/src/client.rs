//! Classic runtime construction extensions for portable completion providers and models.

use schemars::JsonSchema;
use serde::Serialize;

use crate::{agent::AgentBuilder, extractor::ExtractorBuilder};
use rig_core::driver::CompletionProvider;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// Classic-runtime construction sugar on anything that builds a completion
/// model: [`rig_core::driver::CompletionProvider`].
///
/// One blanket impl covers both kinds of provider. A wire-backed provider
/// arrives as a `Bound<P, H>` — a provider config that names its completion
/// wire, bundled with the socket it speaks over — and a typed-transport
/// provider (Bedrock, Vertex AI, gemini-grpc, in-process inference)
/// implements `CompletionProvider` directly. No provider contributes an
/// `agent` forwarder either way.
///
/// This is a trait of its own rather than a second blanket impl behind
/// [`AgentModelExt`]: a bound provider is both a provider and a model, so a
/// single trait blanket-implemented over [`CompletionProvider`] and over
/// [`CompletionModel`](rig_core::completion::CompletionModel) would need
/// those two impls to be provably disjoint, and they are not. The two share
/// no method name, so a type that implements both still resolves.
///
/// ```ignore
/// use rig_agent::prelude::*;
/// use rig_core::providers::anthropic::wire::Anthropic;
///
/// let provider = Anthropic::from_env()?.bound()?;
/// let agent = provider.agent("claude-sonnet-4-5").preamble("Be brief.").build();
/// ```
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
