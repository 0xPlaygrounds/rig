//! Anthropic as data: one config, its wires, and every Messages-format dialect.
//!
//! # Example
//! ```no_run
//! use rig_core::providers::anthropic;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = anthropic::Anthropic::from_env()?;
//!
//! let sonnet = provider.messages(anthropic::completion::CLAUDE_SONNET_4_6);
//! # Ok(())
//! # }
//! ```
//!
//! A wire says what to send and how to read the reply; `.bind(transport)`
//! joins it to a socket and yields the [`Bound`](crate::driver::Bound) that
//! implements the consumer-facing model traits.

pub mod completion;
pub mod streaming;
pub mod wire;

pub use wire::{
    ANTHROPIC, Anthropic, Dialect, MaxTokens, Messages, Models, Quirks, Verify, compatible,
};
