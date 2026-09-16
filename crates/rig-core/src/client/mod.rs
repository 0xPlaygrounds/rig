//! What is left of provider construction now that a provider is data.
//!
//! A provider is plain configuration (`openai::wire::OpenAI`,
//! `anthropic::wire::Anthropic`, `cohere::Cohere`, …) plus one
//! [`Wire`](crate::wire::Wire) per API endpoint; binding that configuration to
//! a transport with [`Bound`](crate::driver::Bound) is what produces a model,
//! and [`Bound`](crate::driver::Bound) is where `completion(model)`,
//! `embedding(model, ndims)`, `verify()` and their siblings live. Nothing
//! generic sits between a provider and its transport any more.
//!
//! Three things outlive that move, and this module is exactly those three:
//!
//! - [`mod@env`]: reading a provider's configuration out of the process
//!   environment, with [`EnvError`] naming a variable that is absent or
//!   unusable.
//! - [`ProviderClientError`], with [`required_env_var`] and
//!   [`optional_env_var`]: the faults of *binding* — a credential that cannot
//!   be read, and a transport that cannot be constructed (no CA store, an
//!   unusable proxy). Both happen before any wire is driven, so they belong to
//!   no operation.
//! - [`VerifyError`]: the `Verify` operation's error. Verification is the one
//!   operation whose reply *status* is the entire answer, so the 401/403
//!   reading lives in its error type instead of being restated by every wire.

pub mod env;
pub mod verify;

use std::env::VarError;
use thiserror::Error;

pub use env::EnvError;
pub use verify::VerifyError;

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
