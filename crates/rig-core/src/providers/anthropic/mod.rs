//! Anthropic provider configuration and Messages-format endpoint wires.
//!
//! ```no_run
//! use rig_core::providers::anthropic;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = anthropic::Anthropic::from_env()?;
//!
//! let sonnet = provider.completion(anthropic::completion::CLAUDE_SONNET_4_6);
//! # Ok(())
//! # }
//! ```
//!
//! Pair a wire with a transport in a [`Model`](crate::Model) to send it.

pub mod completion;
pub mod streaming;
pub mod wire;

pub use crate::client::anthropic::Anthropic;
pub use completion::{
    CLAUDE_FABLE_5, CLAUDE_FABLE_5_1, CLAUDE_HAIKU_4_5, CLAUDE_OPUS_4_6, CLAUDE_OPUS_4_7,
    CLAUDE_OPUS_4_8, CLAUDE_OPUS_5, CLAUDE_SONNET_4_6, CLAUDE_SONNET_5,
};
pub use wire::{
    ANTHROPIC, AnthropicConfig, Dialect, MaxTokens, Messages, Models, Quirks, Verify, compatible,
};
