//! Bind provider configuration to the shared bundled transport.
//!
//! ```no_run
//! use rig_core::client::DefaultTransport;
//! use rig_core::providers::openai::OpenAI;
//!
//! let provider = OpenAI::from_env()?.bound()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use super::ProviderClientError;
use crate::driver::Bound;
use crate::http_client::BoxedHttpClient;

/// Bind provider configuration to the process-wide HTTP transport.
pub trait DefaultTransport: Sized {
    /// Share the default connection pool. A failed transport initialization
    /// is retained by the handle and reported by each call.
    fn bound(self) -> Result<Bound<Self, BoxedHttpClient>, ProviderClientError> {
        Ok(Bound::new(self, rig_reqwest::client::shared()))
    }
}

impl<W: Sized> DefaultTransport for W {}
