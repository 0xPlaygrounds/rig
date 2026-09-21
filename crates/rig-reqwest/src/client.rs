//! Fallible construction of the bundled, type-erased HTTP transport.
//!
//! ```no_run
//! let transport = rig_reqwest::client::bundled()?;
//! # Ok::<(), rig_core::client::ProviderClientError>(())
//! ```

use rig_core::client::ProviderClientError;
use rig_core::driver::Bound;
use rig_core::http_client::{self, BoxedHttpClient};

/// Build a fresh bundled transport, returning a client-construction error if
/// reqwest initialization fails.
pub fn bundled() -> Result<BoxedHttpClient, ProviderClientError> {
    let client = reqwest::Client::builder()
        .build()
        .map_err(|error| http_client::Error::Instance(Box::new(TransportBuildError(error))))?;
    Ok(BoxedHttpClient::from(crate::ReqwestClient::new(client)))
}

/// A transport build failure that displays the source chain and retains the
/// original reqwest error as its source.
#[derive(Debug)]
struct TransportBuildError(reqwest::Error);

impl std::fmt::Display for TransportBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "could not build the bundled reqwest transport: {}",
            self.0
        )?;
        let mut source = std::error::Error::source(&self.0);
        while let Some(cause) = source {
            write!(f, ": {cause}")?;
            source = cause.source();
        }
        Ok(())
    }
}

impl std::error::Error for TransportBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

/// Bind a wire or provider configuration to the bundled, type-erased transport.
pub trait DefaultTransport: Sized {
    /// Bind `self` to a fresh bundled transport, returning an error if transport
    /// initialization fails.
    fn bound(self) -> Result<Bound<Self, BoxedHttpClient>, ProviderClientError> {
        Ok(Bound::new(self, bundled()?))
    }
}

impl<W: Sized> DefaultTransport for W {}
