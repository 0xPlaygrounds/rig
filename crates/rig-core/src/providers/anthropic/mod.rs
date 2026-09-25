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

pub use wire::{
    ANTHROPIC, Anthropic, Dialect, MaxTokens, Messages, Models, Quirks, Verify, compatible,
};
