//! OpenRouter Inference API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig_core::{client::CompletionClient, providers::openrouter};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = openrouter::Client::new("YOUR_API_KEY")?;
//!
//! let sonar = client.completion_model(openrouter::PERPLEXITY_SONAR_PRO);
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod completion;
pub mod embedding;
pub mod model_listing;
pub mod streaming;

pub use client::*;
pub use completion::*;
pub use embedding::*;
pub use model_listing::OpenRouterModelLister;
