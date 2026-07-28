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
//! Agent-to-Agent (A2A) protocol client integration for the Rig agent
//! framework.
//!
//! See the [crate README](https://crates.io/crates/rig-a2a) for an overview.
//!
//! [`A2AClient`] fetches a remote `AgentCard` and [`A2ATool`] wraps each
//! declared skill as a Rig `Tool`, while [`A2AAgentBuilderExt::a2a_tools`]
//! binds them onto a Rig agent at build time.
//!
//! This crate currently supports native targets. Its upstream A2A client uses
//! Tokio networking, which is not available on `wasm32-unknown-unknown`.

pub mod error;
pub use error::A2AError;

pub(crate) mod parts;

pub mod builder_ext;
pub mod client;
pub mod tool;

pub use builder_ext::A2AAgentBuilderExt;
pub use client::{
    A2AClient, A2AClientBuilder, A2ARequest, DEFAULT_HTTP_TIMEOUT, SendMessageResponse,
};
pub use tool::A2ATool;

/// Default well-known path where an A2A `AgentCard` is published.
pub const WELL_KNOWN_AGENT_CARD_PATH: &str = "/.well-known/agent-card.json";
