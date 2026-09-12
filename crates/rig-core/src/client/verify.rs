use crate::{http_client, provider_response, wasm_compat::WasmCompatSend};
use thiserror::Error;

/// Errors from provider client verification.
///
/// Inspect provider failures with [`Self::provider_response_body`],
/// [`Self::provider_response_json`], and [`Self::provider_response_status`].
///
#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("invalid authentication")]
    InvalidAuthentication,
    #[error("provider error: {0}")]
    ProviderError(String),
    /// Raw error response preserved from the provider
    #[error("provider response error: {0}")]
    ProviderResponse(provider_response::ProviderResponseError),
    /// A transport failure that produced no provider reply; a rejected
    /// verification with a body is [`Self::ProviderResponse`].
    #[error("http error: {0}")]
    HttpError(#[source] http_client::Error),
}

crate::provider_response::impl_provider_response_helpers!(VerifyError);

impl From<http_client::Error> for VerifyError {
    fn from(error: http_client::Error) -> Self {
        Self::from_transport_error(error)
    }
}

/// A provider client that can verify the configuration.
/// Clone is required for conversions between client types.
pub trait VerifyClient {
    /// Verify the configuration.
    fn verify(&self) -> impl Future<Output = Result<(), VerifyError>> + WasmCompatSend;
}

#[cfg(test)]
mod provider_response_tests;
