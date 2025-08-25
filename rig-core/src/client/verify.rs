use crate::client::{AsVerify, ProviderClient};
use futures::future::BoxFuture;
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
        reqwest::Error,
    ),
}

/// A provider client that can verify the configuration.
/// Clone is required for conversions between client types.
pub trait VerifyClient: ProviderClient + Clone {
    /// Verify the configuration.
    fn verify(&self) -> impl Future<Output = Result<(), VerifyError>> + Send;
}

pub trait VerifyClientDyn: ProviderClient {
    /// Verify the configuration.
    fn verify(&self) -> BoxFuture<'_, Result<(), VerifyError>>;
}

impl<T: VerifyClient> VerifyClientDyn for T {
    fn verify(&self) -> BoxFuture<'_, Result<(), VerifyError>> {
        Box::pin(self.verify())
    }
}

impl<T: VerifyClientDyn + Clone + 'static> AsVerify for T {
    fn as_verify(&self) -> Option<Box<dyn VerifyClientDyn>> {
        Some(Box::new(self.clone()))
    }
}
