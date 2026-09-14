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
//! Google Cloud Vertex AI provider integration for Rig.
//!
//! This crate exposes Vertex AI hosted model completions through Rig's
//! completion traits. Configure Google Cloud Application Default Credentials or
//! provide credentials through Google Cloud's standard environment before
//! constructing a client.
//!
//! The root `rig` facade re-exports this crate as `rig::vertexai` when the
//! `vertexai` feature is enabled.

pub mod client;
pub mod completion;
pub(crate) mod types;

pub use client::{Client, ClientBuilder};
