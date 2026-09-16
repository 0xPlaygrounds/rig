//! Construction of a provider client with the bundled [`crate::ReqwestClient`]
//! behind the erased default transport.
//!
//! rig-core's provider types default to
//! [`BoxedHttpClient`], the erased
//! transport, so `openai::Client` names a concrete type in every
//! configuration; but rig-core deliberately depends on no transport, so it
//! cannot build one. These two traits are that value: implemented exactly once,
//! for the erased client, they construct it over a fresh `ReqwestClient`. That
//! single applicable impl is what lets `openai::Client::new(key)` infer the
//! transport in expression position, where a type alias default does not
//! apply. An inherent method could do the same, but only from inside rig-core,
//! which would have to know reqwest; a trait in this crate is the
//! orphan-rule-legal seam.
//!
//! Bring it into scope with `use rig::prelude::*` or
//! `use rig_reqwest::prelude::*`. To keep the concrete transport in the type
//! instead, use rig-core's `Bind::bind(ReqwestClient::default())`.

use rig_core::client::ProviderClientError;
use rig_core::driver::Bound;
use rig_core::http_client::{self, BoxedHttpClient};

/// The bundled transport, built fallibly. `reqwest::Client::new()` (and so
/// `ReqwestClient::default()`) panics when the client cannot be built — on a
/// host with no CA store, for one — while every constructor below promises
/// a `Result`. Build through the builder and hand the failure back as the
/// `Http` variant the rest of the client-construction path already uses.
pub fn bundled() -> Result<BoxedHttpClient, ProviderClientError> {
    let client = reqwest::Client::builder()
        .build()
        .map_err(|error| http_client::Error::Instance(Box::new(TransportBuildError(error))))?;
    Ok(BoxedHttpClient::from(crate::ReqwestClient::new(client)))
}

/// The bundled reqwest transport could not be built. `reqwest::Error`
/// displays as just "builder error" and keeps the reason (no CA store, a
/// bad proxy, ..) in its source chain, so this flattens the chain into the
/// message a caller prints, while `source()` still exposes the original.
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

/// One-argument binding of a wire or provider config to the bundled
/// transport: `anthropic::Anthropic::from_env()?.bound()?`.
///
/// rig-core depends on no transport, so it cannot build one; this crate is
/// the orphan-rule-legal seam that supplies the default. To keep the
/// concrete transport in the type instead, use rig-core's
/// `Bind::bind(ReqwestClient::default())`.
pub trait DefaultTransport: Sized {
    /// Bind `self` to a fresh bundled transport.
    fn bound(self) -> Result<Bound<Self, BoxedHttpClient>, ProviderClientError> {
        Ok(Bound::new(self, bundled()?))
    }
}

impl<W: Sized> DefaultTransport for W {}
