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
//! # Entry point: [`functions`]
//!
//! The crate's face is data-oriented: a serde [`functions::Config`]
//! (project / location / credential *source* — never key material) plus the
//! free functions [`functions::complete`] and [`functions::open_stream`].
//!
//! Vertex AI authenticates through Google's Application Default Credentials
//! (ADC) chain, so the authenticated handle cannot honestly be serde data.
//! [`functions::client_from_config`] turns a [`functions::Config`] into the
//! live [`Client`], which every free function takes by reference. Configure
//! ADC (`gcloud auth application-default login`) and set
//! `GOOGLE_CLOUD_PROJECT` before building one.
//!
//! ```no_run
//! use rig_core::completion::CompletionRequest;
//! use rig_vertexai::{completion::GEMINI_2_5_FLASH_LITE, functions};
//!
//! # async fn example() -> Result<(), anyhow::Error> {
//! let cfg = functions::Config::new(GEMINI_2_5_FLASH_LITE);
//! let client = functions::client_from_config(&cfg)?;
//!
//! let response = functions::complete(
//!     &client,
//!     &cfg.model,
//!     CompletionRequest::from_prompt("What is the capital of France?"),
//! )
//! .await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```
//!
//! Because of that OAuth handle, Vertex AI has **no** `rig-agent`
//! `ProviderConfig` arm. To run an agent loop, drive the public
//! `AgentRun` + `prepare_request` protocol and call
//! [`functions::complete`] yourself — see `examples/tool_vertexai.rs`.
//!
//! Streaming is not supported by this integration:
//! [`functions::open_stream`] always errors.
//!
//! The root `rig` facade re-exports this crate as `rig::vertexai` when the
//! `vertexai` feature is enabled.

pub mod client;
pub mod completion;
pub mod functions;
pub(crate) mod types;

pub use client::{Client, ClientBuilder};
