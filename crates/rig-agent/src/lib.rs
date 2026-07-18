#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]
//! Rig's classic, provider-agnostic agent runtime.
//!
//! This crate owns agent orchestration, hooks, retries, prompt conveniences,
//! structured extraction, runtime tool dispatch, and interactive integrations.
//! Portable model, message, provider, memory, vector-store, and tool contracts
//! remain in [`rig_core`].

extern crate self as rig;
extern crate self as rig_agent;

pub mod agent;
pub mod client;
pub mod completion;
pub mod embeddings;
pub mod extractor;
pub mod integrations;
pub mod prelude;
pub mod streaming;
pub mod tool;

// The extracted runtime deliberately uses the portable modules through this
// crate root. Keeping these names local makes ownership visible to users while
// allowing the classic implementation to stay semantically unchanged.
#[doc(hidden)]
pub use rig_core::json_utils;
pub use rig_core::{
    Embed, OneOrMany, id, markers, memory, message, providers, telemetry, vector_store, wasm_compat,
};

#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub mod test_utils;

pub use extractor::{ExtractionError, ExtractionResponse, Extractor, ExtractorBuilder};

#[cfg(test)]
mod conformance;
