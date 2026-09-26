//! The process-wide bundled HTTP transport and connection pool.
//!
//! ```no_run
//! let transport = rig_reqwest::client::shared();
//! ```

use rig_http::http_client::{self, BoxedHttpClient};
use std::sync::Arc;

/// Share the default transport and its connection pool.
/// Construction never panics on a client-build failure: every send reports
/// the saved error, including its source chain. Explicit clients are independent.
/// Environment-derived proxy and certificate settings are read once on first
/// construction. Native shared connections use a process-wide reactor, not a
/// caller's potentially short-lived runtime.
pub fn shared() -> BoxedHttpClient {
    #[cfg(not(target_family = "wasm"))]
    {
        static DEFAULT: std::sync::LazyLock<BoxedHttpClient> =
            std::sync::LazyLock::new(|| crate::ReqwestClient::default().boxed());
        DEFAULT.clone()
    }
    #[cfg(target_family = "wasm")]
    {
        thread_local! {
            static DEFAULT: BoxedHttpClient = crate::ReqwestClient::default().boxed();
        }
        DEFAULT.with(Clone::clone)
    }
}

pub(super) fn default_client() -> crate::ReqwestClient {
    fn build() -> crate::ReqwestClient {
        crate::ReqwestClient(
            Arc::new(reqwest::Client::builder().build().map_err(Arc::new)),
            crate::RuntimePolicy::Shared,
        )
    }
    #[cfg(not(target_family = "wasm"))]
    {
        static DEFAULT: std::sync::LazyLock<crate::ReqwestClient> = std::sync::LazyLock::new(build);
        DEFAULT.clone()
    }
    #[cfg(target_family = "wasm")]
    {
        thread_local! {
            static DEFAULT: crate::ReqwestClient = build();
        }
        DEFAULT.with(Clone::clone)
    }
}

pub(super) fn build_error(error: &Arc<reqwest::Error>) -> http_client::Error {
    http_client::Error::instance(TransportBuildError(Arc::clone(error)))
}

/// Displays the full construction failure while retaining its original source.
#[derive(Debug)]
struct TransportBuildError(Arc<reqwest::Error>);

impl std::fmt::Display for TransportBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "could not build the bundled reqwest transport: {}",
            self.0
        )?;
        let mut source = std::error::Error::source(self.0.as_ref());
        while let Some(cause) = source {
            write!(f, ": {cause}")?;
            source = cause.source();
        }
        Ok(())
    }
}

impl std::error::Error for TransportBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

#[cfg(test)]
mod tests;
