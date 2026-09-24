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
//! ```no_run
//! use rig_bedrock::{client::BedrockRuntime, completion::{AMAZON_NOVA_LITE, Converse}};
//! use rig_core::Model;
//!
//! let model = Model::new(Converse::new(AMAZON_NOVA_LITE), BedrockRuntime::from_env()?);
//! # let _ = model;
//! # Ok::<(), rig_core::client::ProviderClientError>(())
//! ```

pub mod client;
pub mod completion;
pub mod embedding;
pub mod image;
pub mod streaming;
pub mod types;
