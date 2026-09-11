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
//! [`anthropic`] is the exception to the sentence above: it is not the Bedrock
//! Runtime API but an AWS-fronted endpoint that speaks Anthropic's own
//! `/v1/messages` dialect and authenticates with SigV4. It lives here because
//! the signing needs this crate's AWS dependencies, which have no place in
//! rig-core.
//!
//! The root `rig` facade re-exports this crate as `rig::bedrock` when the
//! `bedrock` feature is enabled.

pub mod anthropic;
pub mod client;
pub mod completion;
pub mod embedding;
pub mod image;
pub mod streaming;
pub mod types;
