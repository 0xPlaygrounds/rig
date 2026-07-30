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
//! The crate's only face is [`functions`]: a serde [`functions::Config`]
//! (plus [`functions::EmbeddingConfig`] and [`functions::ImageConfig`])
//! describing how to build an `aws_sdk_bedrockruntime::Client`, and free
//! functions taking that client explicitly —
//! [`functions::complete`], [`functions::open_stream`],
//! [`functions::embed`], [`functions::embed_batches`], and
//! [`functions::generate_image`]. There is no Bedrock client type and no
//! model traits.
//!
//! ```no_run
//! use rig_bedrock::{completion::AMAZON_NOVA_LITE, functions};
//! use rig_core::completion::CompletionRequest;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = functions::Config::new(AMAZON_NOVA_LITE);
//! let client = functions::client_from_config(&cfg).await;
//! let response = functions::complete(
//!     &client,
//!     &cfg.model,
//!     CompletionRequest::from_prompt("Describe the solar system"),
//! )
//! .await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```
//!
//! Agents drive Bedrock through
//! `rig_agent::provider::ProviderConfig::Bedrock(cfg)` (feature `bedrock`).
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

pub mod completion;
pub mod embedding;
pub mod functions;
pub mod image;
pub mod streaming;
pub mod types;
