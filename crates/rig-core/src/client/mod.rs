//! Environment configuration helpers and provider construction errors.
//! Bind provider configuration to a transport through [`Bound`](crate::driver::Bound).
//!
//! ```no_run
//! use rig_core::client::required_env_var;
//!
//! let credential = required_env_var("OPENAI_API_KEY")?;
//! # let _ = credential;
//! # Ok::<(), rig_core::client::ProviderClientError>(())
//! ```

pub mod env;
pub(crate) mod verify;

use std::env::VarError;
use thiserror::Error;

pub use env::EnvError;

use crate::http_client;

/// Errors returned while reading a provider's configuration from the
/// environment, or while binding that configuration to a transport.
///
/// These are the problems detectable before any request is sent: a missing or
/// invalid environment value, and a transport that cannot be built.
#[derive(Debug, Error)]
pub enum ProviderClientError {
    /// A required or optional environment variable could not be read as valid Unicode.
    ///
    /// For required variables, this variant is also returned when the variable is not present.
    #[error("environment variable `{name}` is not set or is invalid")]
    EnvironmentVariable {
        /// The environment variable name.
        name: &'static str,
        /// The underlying environment lookup error.
        #[source]
        source: VarError,
    },
    /// The HTTP transport could not be constructed for this provider.
    #[error(transparent)]
    Http(#[from] http_client::Error),
}

/// Result type returned by provider configuration helpers.
pub type ProviderClientResult<T> = std::result::Result<T, ProviderClientError>;

/// Read a required environment variable for provider construction.
///
/// Returns [`ProviderClientError::EnvironmentVariable`] when the variable is missing or contains
/// invalid Unicode.
pub fn required_env_var(name: &'static str) -> ProviderClientResult<String> {
    std::env::var(name).map_err(|source| ProviderClientError::EnvironmentVariable { name, source })
}

/// Read an optional environment variable for provider construction.
///
/// Missing variables return `Ok(None)`. Variables containing invalid Unicode return
/// [`ProviderClientError::EnvironmentVariable`].
pub fn optional_env_var(name: &'static str) -> ProviderClientResult<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(VarError::NotPresent) => Ok(None),
        Err(source) => Err(ProviderClientError::EnvironmentVariable { name, source }),
    }
}
