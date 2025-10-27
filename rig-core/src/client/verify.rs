use crate::{
    http_client,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend},
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("invalid authentication")]
    InvalidAuthentication,
    #[error("provider error: {0}")]
    ProviderError(String),
    #[error("http error: {0}")]
    HttpError(
        #[from]
        #[source]
        http_client::Error,
    ),
}

/// A provider client that can verify the configuration.
/// Clone is required for conversions between client types.
pub trait VerifyClient {
    /// Verify the configuration.
    fn verify(&self) -> impl Future<Output = Result<(), VerifyError>> + WasmCompatSend;
}

pub trait VerifyClientDyn {
    /// Verify the configuration.
    fn verify(&self) -> WasmBoxedFuture<'_, Result<(), VerifyError>>;
}

impl<T> VerifyClientDyn for T
where
    T: VerifyClient,
{
    fn verify(&self) -> WasmBoxedFuture<'_, Result<(), VerifyError>> {
        Box::pin(self.verify())
    }
}
