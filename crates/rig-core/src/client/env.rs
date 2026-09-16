//! Reading a provider's configuration from the environment.
//!
//! This is all that survives of the client layer: a wire is built from data,
//! and the only construction step that can fail before a request is sent is
//! reading a credential out of the environment.

use std::env::VarError;

/// A provider's configuration could not be read from the environment.
#[derive(Debug, thiserror::Error)]
pub enum EnvError {
    /// A required variable is absent, or its value is not valid Unicode.
    #[error("environment variable `{name}` is not set or is invalid")]
    Variable {
        /// The variable's name.
        name: &'static str,
        /// The underlying lookup failure.
        #[source]
        source: VarError,
    },
    /// A variable's value is present but unusable (an unparsable URL, an
    /// empty credential).
    #[error("environment variable `{name}` is invalid: {detail}")]
    Invalid {
        /// The variable's name.
        name: &'static str,
        /// What is wrong with the value.
        detail: String,
    },
}

/// Read a required environment variable.
pub fn required(name: &'static str) -> Result<String, EnvError> {
    std::env::var(name).map_err(|source| EnvError::Variable { name, source })
}

/// Read an optional environment variable. An absent variable is `Ok(None)`;
/// a present one that is not valid Unicode is an error, because silently
/// ignoring it would send a request with the wrong configuration.
pub fn optional(name: &'static str) -> Result<Option<String>, EnvError> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(VarError::NotPresent) => Ok(None),
        Err(source) => Err(EnvError::Variable { name, source }),
    }
}
