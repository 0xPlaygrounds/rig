//! Environment configuration helpers and provider construction errors.
//! A provider configuration builds wires; a [`Model`](crate::driver::Model)
//! binds one to a transport.
//!
//! ```no_run
//! use rig_core::client::env;
//!
//! let credential = env::required("OPENAI_API_KEY")?;
//! # let _ = credential;
//! # Ok::<(), env::EnvError>(())
//! ```

mod cached_content;
pub mod env;
pub(crate) mod verify;

use thiserror::Error;

pub use env::EnvError;

use crate::http_client;

/// A provider's transport could not be built before any request was sent.
#[derive(Debug, Error)]
pub enum ProviderClientError {
    /// The HTTP transport could not be constructed for this provider.
    #[error(transparent)]
    Http(#[from] http_client::Error),
}

/// Result type returned by provider configuration helpers.
pub type ProviderClientResult<T> = std::result::Result<T, ProviderClientError>;
