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
//! `vertexai` feature is enabled. The raw-response escape hatch returns
//! [`completion::VertexGenerateContentOutput`], which can be named in downstream
//! APIs and recovered from a normalized response's `raw` JSON.
//!
//! SDK initialization belongs to the host, before borrowing an ECS world:
//!
//! ```no_run
//! # async fn prepare() -> Result<(), rig_vertexai::client::VertexAiClientError> {
//! let client = rig_vertexai::Client::from_env()?;
//! client.inner().await?;
//! # Ok(())
//! # }
//! ```
//!
//! The `ecs_host_model` Cargo example then wraps the completion model in
//! `CompletionAdapter` and registers it without a core provider reference.
//! A dispatching host supplies Tokio polling context; this integration is unary-only.

pub mod client;
pub mod completion;
pub(crate) mod types;

pub use client::{Client, ClientBuilder};
