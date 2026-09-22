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
//! A host that already owns a
//! [`google_cloud_aiplatform_v1::client::PredictionService`]. with its own
//! endpoint, credentials, transport and retry policy. can pass it to
//! [`ClientBuilder::with_prediction_service`] instead. That path resolves no
//! credentials of its own; the credential-resolving paths
//! ([`Client::from_env`], [`Client::new`], and a [`ClientBuilder`] without a
//! supplied service or explicit credentials) spawn a token-refresh task during construction and so
//! require a Tokio runtime that outlives the client, as documented on
//! [`Client::from_env`].
//!
//! The root `rig` facade re-exports this crate as `rig::vertexai` when the
//! `vertexai` feature is enabled. The raw-response escape hatch returns
//! [`completion::VertexGenerateContentOutput`], which can be named in downstream
//! APIs and recovered from a normalized response's `raw` JSON.

pub mod client;
pub mod completion;
pub(crate) mod types;

pub use client::{Client, ClientBuilder};
