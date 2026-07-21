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
//! contracts remain in [`rig_core`] and are reachable here through the
//! explicit [`core`] namespace; this crate's root deliberately exports only
//! runtime-owned items. The comprehensive end-user facade is the root `rig`
//! crate.

extern crate self as rig;

/// Direct access to portable provider, data, memory, and tool contracts.
///
/// This explicit namespace is also the stable expansion root for portable
/// `#[rig_tool]` functions in crates that depend on `rig-agent` without a
/// separate direct `rig-core` dependency.
///
/// Portable `rig-core` root items are reachable here, but deliberately *not*
/// at the `rig_agent` crate root — adding a root export to `rig-core` must not
/// silently add one to `rig-agent`. A stable `rig-core` root export
/// ([`rig_core::OneOrMany`]) demonstrates both halves of that invariant (the
/// two doctests below enforce it):
///
/// ```
/// // Reachable through the explicit `core` namespace.
/// use rig_agent::core::OneOrMany;
/// let _reachable: Option<OneOrMany<u8>> = None;
/// ```
///
/// ```compile_fail
/// // NOT reachable at the `rig_agent` crate root.
/// use rig_agent::OneOrMany as _;
/// ```
pub mod core {
    pub use rig_core::*;
}

pub mod agent;
pub mod client;
pub mod completion;
pub mod extractor;
pub mod integrations;
// Shared JSON helpers live in rig-core; re-export so call sites stay
// `json_utils::merge` / `json_utils::serialize_json_value`.
pub(crate) use rig_core::json_utils;
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
