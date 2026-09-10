use crate::{http_client, provider_response, wasm_compat::WasmCompatSend};
use thiserror::Error;

/// Errors from provider client verification.
///
/// Inspect provider failures with [`Self::provider_response_body`],
/// [`Self::provider_response_json`], and [`Self::provider_response_status`].
///
/// Note: no provider path currently constructs [`Self::ProviderResponse`] for
/// verification; real verify failures surface as [`Self::HttpError`], which
/// the helpers read. The variant is kept for symmetry with the other capability
/// errors and for future provider paths that preserve a 2xx error envelope.
#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("invalid authentication")]
    InvalidAuthentication,
    #[error("provider error: {0}")]
    ProviderError(String),
    /// Raw error response preserved from the provider
    #[error("provider response error: {0}")]
    ProviderResponse(provider_response::ProviderResponseError),
    #[error("http error: {0}")]
    HttpError(
        #[from]
        #[source]
        http_client::Error,
    ),
}

crate::provider_response::impl_provider_response_helpers!(VerifyError);

/// A provider client that can verify the configuration.
/// Clone is required for conversions between client types.
pub trait VerifyClient {
    /// Verify the configuration.
    fn verify(&self) -> impl Future<Output = Result<(), VerifyError>> + WasmCompatSend;
}

#[cfg(test)]
mod provider_response_tests;
