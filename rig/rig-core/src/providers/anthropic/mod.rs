//! Anthropic API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::anthropic;
//!
//! let client = anthropic::Client::new("YOUR_API_KEY");
//!
//! let sonnet = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
//! ```

pub mod client;
pub mod completion;
pub mod decoders;
pub mod model_listing;
pub mod streaming;

pub use client::{Client, ClientBuilder};
