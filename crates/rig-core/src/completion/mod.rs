//! Provider-agnostic completion and chat abstractions.
//!
//! This module contains the low-level request and response types used by provider
//! implementations. [`CompletionModel`] is the portable provider-facing trait;
//! runtime conveniences such as `Prompt`, `Chat`, and agent runners live in
//! `rig-agent` or another runtime crate.
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
//!     completion::{AssistantContent, CompletionModel},
//!     providers::openai,
//! };
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = openai::Client::from_env()?;
//! let model = client
//!     .completion_model(openai::GPT_5_2);
//! let request = model
//!     .completion_request("What is Rig?")
//!     .preamble("Answer concisely.".to_string())
//!     .build();
//!
//! let response = model.completion(request).await?;
//! if let AssistantContent::Text(text) = response.choice.first() {
//!     println!("{}", text.text);
//! }
//! # Ok(())
//! # }
//! ```

pub mod message;
pub mod request;

pub use message::{AssistantContent, Message, MessageError};
pub use request::*;
