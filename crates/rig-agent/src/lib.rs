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
//! Rig's classic agent runtime.
//!
//! This crate owns the mature builder, run state machine, typed hook system,
//! contextual tool registry, memory orchestration, extraction, and shared
//! blocking/streaming driver. Portable provider, message, tool, and storage
//! contracts remain in [`rig_core`].

extern crate self as rig;

pub use rig_core::*;

pub mod agent;
pub mod client;
pub mod completion;
pub mod extractor;
pub mod integrations;
pub(crate) mod json_utils;
pub mod prelude;
pub mod streaming;
#[cfg(any(test, feature = "test-utils"))]
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub mod test_utils;
pub mod tool;

pub use agent::{Agent, AgentBuilder, AgentRun, AgentRunner};
pub use extractor::ExtractionResponse;

#[cfg(feature = "derive")]
#[cfg_attr(docsrs, doc(cfg(feature = "derive")))]
pub use rig_derive::rig_tool;
#[cfg(feature = "derive")]
#[cfg_attr(docsrs, doc(cfg(feature = "derive")))]
pub use rig_derive::rig_tool as tool_macro;
