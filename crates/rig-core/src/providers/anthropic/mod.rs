//! Anthropic API client and Rig integration
//!
//! # Example
//! ```ignore
//! use rig_core::{client::CompletionClient, providers::anthropic};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = anthropic::Client::new("YOUR_API_KEY")?;
//!
//! let sonnet = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod completion;
pub mod model_listing;
pub mod streaming;

pub use client::{Client, ClientBuilder};
// Hoisted to the provider root, where the old `rig-reqwest` alias tree put
// them and where callers name them.
pub use completion::CompletionModel;
pub use model_listing::AnthropicModelLister;
