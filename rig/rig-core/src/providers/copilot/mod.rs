//! GitHub Copilot API client and Rig integration
//!
//! The Copilot API exposes an OpenAI-compatible `/chat/completions` endpoint
//! but omits several response fields that the standard OpenAI specification
//! considers required (`object`, `created`, `finish_reason`). This provider
//! derives from the OpenAI module, re-using its request types and message
//! format while relaxing the response contract.
//!
//! # Example
//! ```no_run
//! use rig::providers::copilot;
//!
//! let client = copilot::Client::builder()
//!     .api_key("ghu_xxxx")
//!     .base_url("https://api.githubcopilot.com")
//!     .build()
//!     .expect("failed to build Copilot client");
//!
//! let model = client.completion_model("gpt-4o");
//! ```

pub mod client;
pub mod completion;

pub use client::*;
pub use completion::*;
