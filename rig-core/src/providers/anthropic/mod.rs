//! Anthropic API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::anthropic;
//!
//! let client = anthropic::Anthropic::new("YOUR_API_KEY");
//!
//! let sonnet = client.completion_model(anthropic::CLAUDE_3_5_SONNET);
//! ```

pub mod client;
pub mod completion;
pub mod decoders;
pub mod streaming;

pub use client::{Client, ClientBuilder};
