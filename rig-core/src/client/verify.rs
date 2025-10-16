use crate::client::{AsVerify, ProviderClient};
use futures::future::BoxFuture;
use thiserror::Error;

/// Errors that can occur during client verification.
///
/// This enum represents the possible failure modes when verifying
/// a provider client's configuration and authentication.
#[derive(Debug, Error)]
pub enum VerifyError {
    /// Authentication credentials are invalid or have been revoked.
    ///
    /// This typically indicates that the API key or other authentication
    /// method is incorrect or no longer valid.
    #[error("invalid authentication")]
    InvalidAuthentication,

    /// The provider returned an error during verification.
    ///
    /// This wraps provider-specific error messages that occur during
    /// the verification process.
    #[error("provider error: {0}")]
    ProviderError(String),

    /// An HTTP error occurred while communicating with the provider.
    ///
    /// This typically indicates network issues, timeouts, or server errors.
    #[error("http error: {0}")]
    HttpError(
        #[from]
        #[source]
        reqwest::Error,
    ),
}

/// A provider client that can verify its configuration and authentication.
///
/// This trait extends [`ProviderClient`] to provide configuration verification functionality.
/// Providers that implement this trait can validate that their API keys and settings
/// are correct before making actual API calls.
///
/// # When to Implement
///
/// Implement this trait for provider clients that support:
/// - API key validation
/// - Configuration testing
/// - Authentication verification
/// - Connectivity checks
///
/// # Examples
///
/// ```no_run
/// use rig::client::{ProviderClient, VerifyClient};
/// use rig::providers::openai::Client;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = Client::new("api-key");
///
/// // Verify the client configuration
/// match client.verify().await {
///     Ok(()) => println!("Client configuration is valid"),
///     Err(e) => eprintln!("Verification failed: {}", e),
/// }
/// # Ok(())
/// # }
/// ```
///
/// # See Also
///
/// - [`VerifyError`] - Errors that can occur during verification
/// - [`VerifyClientDyn`] - Dynamic dispatch version for runtime polymorphism
pub trait VerifyClient: ProviderClient + Clone {
    /// Verifies the client configuration and authentication.
    ///
    /// This method tests whether the client is properly configured and can
    /// successfully authenticate with the provider. It typically makes a
    /// minimal API call to validate credentials.
    ///
    /// # Errors
    ///
    /// Returns [`VerifyError::InvalidAuthentication`] if the credentials are invalid.
    /// Returns [`VerifyError::HttpError`] if there are network connectivity issues.
    /// Returns [`VerifyError::ProviderError`] for provider-specific errors.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rig::client::VerifyClient;
    /// use rig::providers::openai::Client;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::new("api-key");
    /// client.verify().await?;
    /// println!("Configuration verified successfully");
    /// # Ok(())
    /// # }
    /// ```
    fn verify(&self) -> impl Future<Output = Result<(), VerifyError>> + Send;
}

/// Dynamic dispatch version of [`VerifyClient`].
///
/// This trait provides the same functionality as [`VerifyClient`] but uses
/// boxed futures for trait object compatibility, enabling runtime polymorphism.
/// It is automatically implemented for all types that implement [`VerifyClient`].
///
/// # When to Use
///
/// Use this trait when you need to work with verify clients of different types
/// at runtime, such as in the [`DynClientBuilder`](crate::client::builder::DynClientBuilder).
pub trait VerifyClientDyn: ProviderClient {
    /// Verifies the client configuration and authentication.
    ///
    /// Returns a boxed future for trait object compatibility.
    fn verify(&self) -> BoxFuture<'_, Result<(), VerifyError>>;
}

impl<T> VerifyClientDyn for T
where
    T: VerifyClient,
{
    fn verify(&self) -> BoxFuture<'_, Result<(), VerifyError>> {
        Box::pin(self.verify())
    }
}

impl<T> AsVerify for T
where
    T: VerifyClientDyn + Clone + 'static,
{
    fn as_verify(&self) -> Option<Box<dyn VerifyClientDyn>> {
        Some(Box::new(self.clone()))
    }
}
