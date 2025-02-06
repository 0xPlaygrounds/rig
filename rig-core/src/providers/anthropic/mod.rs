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
pub mod streaming;

pub use client::{Client, ClientBuilder};
pub use completion::{
    ANTHROPIC_VERSION_2023_01_01, ANTHROPIC_VERSION_2023_06_01, ANTHROPIC_VERSION_LATEST,
    CLAUDE_3_5_SONNET, CLAUDE_3_HAIKU, CLAUDE_3_OPUS, CLAUDE_3_SONNET,
};
