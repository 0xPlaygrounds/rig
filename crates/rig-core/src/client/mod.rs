//! Environment configuration helpers and the declarations provider clients
//! share. A provider's client (such as [`crate::providers::openai::OpenAI`])
//! puts its configuration on a transport and builds the models it serves; each
//! is defined in its provider module.
//!
//! ```no_run
//! use rig_core::client::env;
//!
//! let credential = env::required("OPENAI_API_KEY")?;
//! # let _ = credential;
//! # Ok::<(), env::EnvError>(())
//! ```

pub mod env;
pub(crate) mod macros;

pub use env::EnvError;

#[cfg(test)]
mod tests;
