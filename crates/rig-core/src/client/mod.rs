//! Provider clients and environment configuration helpers. A provider's
//! client (such as [`crate::providers::openai::OpenAI`]) puts its
//! configuration on a transport and builds the models it serves; each is
//! re-exported from its provider module.
//!
//! ```no_run
//! use rig_core::client::env;
//!
//! let credential = env::required("OPENAI_API_KEY")?;
//! # let _ = credential;
//! # Ok::<(), env::EnvError>(())
//! ```

pub(crate) mod anthropic;
mod cached_content;
pub(crate) mod chatgpt;
pub(crate) mod cohere;
pub(crate) mod copilot;
pub mod env;
pub(crate) mod gemini;
pub(crate) mod gemini_caching;
pub(crate) mod macros;
pub(crate) mod ollama;
pub(crate) mod openai;
pub(crate) mod voyageai;

pub use env::EnvError;

#[cfg(test)]
mod tests;
