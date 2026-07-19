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
//! This crate exposes two independent paths to Bedrock models:
//!
//! - **Converse** ([`client`], [`completion`], [`streaming`], …) — the classic
//!   Bedrock Runtime Converse / ConverseStream APIs. Requires AWS credentials
//!   configured for the AWS SDK and a region with access to the selected model.
//! - **Mantle** ([`mantle`]) — OpenAI-compatible HTTP API for models such as
//!   OpenAI GPT-OSS. Auth is a short-term IAM bearer token (or
//!   `AWS_BEARER_TOKEN_BEDROCK`). Reuses Rig's OpenAI Responses / Completions
//!   clients. Converse behavior is unchanged.
//!
//! The root `rig` facade re-exports this crate as `rig::bedrock` when the
//! `bedrock` feature is enabled.

pub mod client;
pub mod completion;
pub mod embedding;
pub mod image;
pub mod mantle;
pub mod streaming;
pub mod types;
