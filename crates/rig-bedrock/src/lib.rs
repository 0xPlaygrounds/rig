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
//! AWS Bedrock provider integration for Rig.
//!
//! This crate exposes Bedrock completion, streaming, embedding, and image
//! generation models through Rig's provider traits. It requires AWS credentials
//! configured for the AWS SDK and a region with access to the selected Bedrock
//! model.
//!
//! The root `rig` facade re-exports this crate as `rig::bedrock` when the
//! `bedrock` feature is enabled.

pub mod client;
pub mod completion;
pub mod embedding;
pub mod image;
pub mod streaming;
pub mod types;
