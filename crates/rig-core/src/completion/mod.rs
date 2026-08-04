//! Provider-agnostic completion and chat abstractions.
//!
//! This module contains the low-level request and response types used by provider
//! implementations. Each provider's `functions` module exposes free functions
//! over these types; runtimes build orchestration on top of that boundary.
//!
//! `CompletionRequest` is Rig's canonical request representation. Provider modules
//! translate it into provider-specific request bodies and convert responses back into
//! [`CompletionResponse`].
//!
//! # Example
//!
//! ```no_run
//! use rig_core::{http_runtime::HttpRuntime, providers::openai};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = openai::functions::Config::from_env(openai::GPT_5_2)?;
//! let rt = HttpRuntime::new();
//! let request = rig_core::completion::CompletionRequest::from_prompt("What is Rig?");
//! let response = openai::functions::complete(&cfg, &rt, request).await?;
//! println!("{:?}", response.choice);
//! # Ok(())
//! # }
//! ```

pub mod message;
pub mod request;

pub use message::{AssistantContent, Message, MessageError};
pub use request::*;
