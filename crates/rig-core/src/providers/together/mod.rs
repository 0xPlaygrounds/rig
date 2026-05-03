//! Together AI API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig_core::{client::EmbeddingsClient, providers::together};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = together::Client::new("YOUR_API_KEY")?;
//!
//! let together_embedding_model = client.embedding_model(together::BGE_BASE_EN_V1_5);
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod completion;
pub mod embedding;
pub mod streaming;

pub use client::Client;
pub use completion::*;
pub use embedding::*;
