//! Anthropic API client and Rig integration
//!
//! # Example
//! ```no_run
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
#[cfg(feature = "sigv4")]
pub(crate) mod sigv4;
pub mod streaming;

pub use client::{AnthropicKey, Client, ClientBuilder};
