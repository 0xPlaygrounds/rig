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
#[cfg(feature = "anthropic")]
#[cfg_attr(docsrs, doc(cfg(feature = "anthropic")))]
pub mod model_listing;
pub mod streaming;

#[cfg(feature = "anthropic")]
pub use client::{Client, ClientBuilder};
