//! Anthropic API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::anthropic;
//!
//! let client = anthropic::Anthropic::new("YOUR_API_KEY");
//!
//! let sonnet = client.completion_model(anthropic::CLAUDE_SONNET_4_5);
//! ```

pub mod client;
pub mod completion;
pub mod decoders;
pub mod streaming;

pub use client::{Client, ClientBuilder};
