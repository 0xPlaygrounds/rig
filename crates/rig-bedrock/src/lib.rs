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
//! The data-oriented face is [`functions`]: a serde [`functions::Config`]
//! (plus [`functions::EmbeddingConfig`] and [`functions::ImageConfig`])
//! describing how to build an `aws_sdk_bedrockruntime::Client`, and free
//! functions taking that client explicitly —
//! [`functions::complete`], [`functions::open_stream`],
//! [`functions::embed`], [`functions::embed_batches`], and
//! [`functions::generate_image`]. [`Client`] is the concrete, monomorphic
//! ergonomic connection handle; it materializes those same configs and can
//! retain a caller-built AWS SDK client for reuse. No model traits are needed.
//!
//! ```no_run
//! use rig_bedrock::{Client, completion::AMAZON_NOVA_LITE, functions};
//! use rig_core::completion::CompletionRequest;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = Client::from_env();
//! let cfg = client.config(AMAZON_NOVA_LITE);
//! let aws = client.get_inner().await;
//! let response = functions::complete(
//!     &aws,
//!     &cfg.model,
//!     CompletionRequest::from_prompt("Describe the solar system"),
//! )
//! .await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```
//!
//! With the root `rig` crate's `bedrock` feature and `rig::prelude::*`, the
//! same client supports `client.agent(model)` and
//! `client.completion_model(model).completion_request(prompt)`.
//!
//! The sibling modules hold the model-id constants ([`completion`],
//! [`embedding`], [`image`]) and the AWS wire-type conversions.
//!
//! Requires AWS credentials configured for the AWS SDK and a region with
//! access to the selected Bedrock model.
//!
//! The root `rig` facade re-exports this crate as `rig::bedrock` when the
//! `bedrock` feature is enabled.

/// The AWS SDK's Bedrock runtime client, re-exported: the free functions in
/// [`functions`] take one explicitly, so callers need to name its type.
pub use aws_sdk_bedrockruntime;

pub mod client;
pub mod completion;
pub mod embedding;
pub mod functions;
pub mod image;
pub mod streaming;
pub mod types;

pub use client::{Client, ClientBuilder};
