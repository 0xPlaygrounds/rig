//! Doubleword inference API client and Rig integration.
//!
//! [Doubleword](https://docs.doubleword.ai) is an OpenAI-compatible inference
//! provider. This integration covers the **realtime** tier: synchronous chat
//! completions and streaming via [`CompletionModel`], plus embeddings
//! ([`EmbeddingModel`]) on the same endpoint. Doubleword's cheaper **async**
//! and **batch** tiers run through the OpenAI-compatible Batch API
//! (`/v1/batches`); Rig support for them is not yet included.
//!
//! Set `DOUBLEWORD_API_KEY` (and optionally `DOUBLEWORD_BASE_URL`) to use
//! [`Client::from_env`].
//!
//! # Example
//! ```no_run
//! use rig_core::{
//!     client::{CompletionClient, ProviderClient},
//!     providers::doubleword,
//! };
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = doubleword::Client::from_env()?;
//! let agent = client.agent(doubleword::QWEN3_5_9B).build();
//! # let _ = agent;
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod completion;
pub mod embedding;

pub use client::Client;
pub use completion::*;
pub use embedding::*;
