//! Reads Unicode provider configuration values from the process environment.
//!
//! ```no_run
//! use rig_core::client::env;
//!
//! let credential = env::required("OPENAI_API_KEY")?;
//! # let _ = credential;
//! # Ok::<(), env::EnvError>(())
//! ```

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

/// Reads a required variable, returning an error if absent or non-Unicode.
/// Empty strings are accepted; callers validate the value.
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
