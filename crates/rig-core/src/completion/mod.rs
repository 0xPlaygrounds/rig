//! Provider-agnostic completion and chat abstractions.
//!
//! This module contains the low-level request and response types used by provider
//! implementations. [`CompletionModel`] is the provider-facing trait implemented
//! by completion models; runtimes build orchestration on top of this boundary.
//!
//! `CompletionRequest` is Rig's canonical request representation. Provider modules
//! translate it into provider-specific request bodies and convert responses back into
//! [`CompletionResponse`].
//!
//! # Example
//!
//! ```ignore
//! use rig_core::{
//!     client::CompletionClient,
//!     completion::CompletionModel,
//!     providers::openai,
//! };
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = openai::Client::from_env()?;
//! let model = client.completion_model(openai::GPT_5_2);
//! let request = model.completion_request("What is Rig?").build();
//! let response = model.completion(request).await?;
//! println!("{:?}", response.choice);
//! # Ok(())
//! # }
//! ```

pub mod handle;
pub mod message;
pub mod request;

pub use handle::{ModelHandle, ModelRef};
pub use message::{AssistantContent, Message, MessageError};
pub use request::*;
