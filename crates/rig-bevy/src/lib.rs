#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used
    )
)]
//! Experimental ECS-native runtime for Rig.
//!
//! Runtime state is authoritative Bevy ECS data. Async work crosses the world
//! boundary only as owned effect requests and correlated completions.

#[cfg(target_family = "wasm")]
compile_error!("rig-bevy is a native-only experimental runtime; use rig-agent on WebAssembly");

pub mod adapters;
pub mod components;
pub mod debug;
pub mod effects;
pub mod persistence;
pub mod policy;
pub mod prelude;
pub mod runtime;
pub mod schedule;
pub mod topology;

pub use runtime::{AgentSpec, BevyCompletionClientExt, BevyRuntime, PendingRun, RunHandle};

#[cfg(test)]
mod conformance;
