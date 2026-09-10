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
// Native-only: the AWS credential chain has no socket to use on wasm, and the
// crate's `sigv4` dependencies are declared in a `cfg(not(target_arch =
// "wasm32"))` table, so this module has nothing to build against there.
#[cfg(all(feature = "sigv4", not(target_arch = "wasm32")))]
pub(crate) mod sigv4;
pub mod streaming;

pub use client::{AnthropicKey, Client, ClientBuilder};
pub use completion::CompletionModel;
pub use model_listing::AnthropicModelLister;
