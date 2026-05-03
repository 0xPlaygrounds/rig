//! Provider-agnostic completion and chat abstractions.
//!
//! This module contains the low-level request and response types used by provider
//! implementations, plus the high-level traits most callers use through
//! [`Agent`](crate::agent::Agent):
//!
//! - [`Prompt`] sends one user prompt and returns assistant text.
//! - [`Chat`] sends a prompt with existing history and returns assistant text.
//! - [`TypedPrompt`] requests structured output and deserializes it into a Rust type.
//! - [`Completion`] exposes a request builder for call-site overrides.
//! - [`CompletionModel`] is the provider-facing trait implemented by completion models.
//!
//! `CompletionRequest` is Rig's canonical request representation. Provider modules
//! translate it into provider-specific request bodies and convert responses back into
//! [`CompletionResponse`].
//!
//! # Example
//!
//! ```no_run
//! use rig_core::{
//!     client::{CompletionClient, ProviderClient},
//!     completion::Prompt,
//!     providers::openai,
//! };
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = openai::Client::from_env()?;
//! let agent = client
//!     .agent(openai::GPT_5_2)
//!     .preamble("Answer concisely.")
//!     .build();
//!
//! let answer = agent.prompt("What is Rig?").await?;
//! println!("{answer}");
//! # Ok(())
//! # }
//! ```

pub mod message;
pub mod request;

pub use message::{AssistantContent, Message, MessageError};
pub use request::*;
