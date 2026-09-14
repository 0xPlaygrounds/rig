use super::{Client, ModelTransport, Provider};
use crate::model::{ModelList, ModelListingError};
use crate::wasm_compat::WasmCompatSend;
use crate::wasm_compat::WasmCompatSync;
use std::future::Future;

/// A provider client with model listing capabilities.
///
/// This trait provides methods to discover and list available models from LLM providers.
/// All models are returned in a single list.
///
/// # Type Parameters
///
/// - `ModelLister`: The type that implements the actual model listing logic
///
/// # Example
///
/// ```rust,ignore
/// use rig_core::client::ModelListingClient;
/// use rig_core::providers::openai::Client;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Initialize the OpenAI client
///     let openai = Client::new("your-open-ai-api-key");
///
///     // List all available models
///     let models = openai.list_models().await?;
///
///     println!("Available models:");
///     for model in models.iter() {
///         println!("- {} ({})", model.display_name(), model.id);
///     }
///
///     Ok(())
/// }
/// ```
pub trait ModelListingClient {
    /// List all available models from the provider.
    ///
    /// This method retrieves all available models. Providers that support pagination
    /// internally handle fetching all pages and return complete results.
    ///
    /// # Returns
    ///
    /// A `ModelList` containing all available models from the provider.
    ///
    /// # Errors
    ///
    /// Returns a `ModelListingError` if:
    /// - The request to the provider fails
    /// - Authentication fails
    /// - The provider returns an error response
    /// - The response cannot be parsed
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use rig_core::client::ModelListingClient;
    /// use rig_core::prelude::*;
    /// use rig_core::providers::openai::Client;
    ///
    /// let openai = Client::from_env()?;
    /// let models = openai.list_models().await?;
    ///
    /// println!("Found {} models", models.len());
    /// for model in models.iter() {
    ///     println!("- {} ({})", model.display_name(), model.id);
    /// }
    /// ```
    fn list_models(
        &self,
    ) -> impl Future<Output = Result<ModelList, ModelListingError>> + WasmCompatSend;
}

/// A trait for implementing model listing logic for a specific provider.
///
/// This trait should be implemented by provider-specific types that handle the
/// details of making HTTP requests to list models and converting provider-specific
/// responses into the generic `Model` format. Providers with pagination
/// support should internally fetch all pages before returning results.
///
/// # Type Parameters
///
/// - `H`: The HTTP backend, any [`crate::http_client::HttpClientExt`] implementation
///
/// # Example Implementation
///
/// ```rust,ignore
/// use crate::client::{Client, HasModelListing, ModelLister, ModelTransport};
/// use crate::model::{Model, ModelList, ModelListingError};
///
/// struct MyProviderModelLister<H> {
///     client: Client<MyProvider, H>,
/// }
///
/// impl<H> ModelLister<H> for MyProviderModelLister<H>
/// where
///     H: HttpClientExt + WasmCompatSend + WasmCompatSync,
/// {
///     async fn list_all(&self) -> Result<ModelList, ModelListingError> {
///         // Fetch all models (handle pagination internally if needed)
///         todo!()
///     }
/// }
///
/// impl HasModelListing for MyProvider {
///     type Lister<H> = MyProviderModelLister<H> where H: ModelTransport;
///
///     fn model_lister<H: ModelTransport>(client: &Client<Self, H>) -> Self::Lister<H> {
///         MyProviderModelLister { client: client.clone() }
///     }
/// }
/// ```
///
/// `H` stays a parameter of this trait: it is the transport, not a provider
/// leak, and the listing request is written against it.
pub trait ModelLister<H>: WasmCompatSend + WasmCompatSync {
    /// List all available models from the provider.
    ///
    /// This implementation should handle fetching all pages if the provider
    /// supports pagination, returning complete results in a single call.
    ///
    /// # Returns
    ///
    /// A `ModelList` containing all available models.
    fn list_all(
        &self,
    ) -> impl std::future::Future<Output = Result<ModelList, ModelListingError>> + WasmCompatSend;
}

/// A [`Provider`] that can list its models. Implementing this is what makes
/// [`ModelListingClient`] available on `Client<Self, H>`.
pub trait HasModelListing: Provider {
    /// The concrete lister built over transport `H`.
    type Lister<H>: ModelLister<H>
    where
        H: ModelTransport;

    /// Build the lister from `client`.
    fn model_lister<H>(client: &Client<Self, H>) -> Self::Lister<H>
    where
        H: ModelTransport;
}

impl<P, H> ModelListingClient for Client<P, H>
where
    P: HasModelListing,
    H: ModelTransport,
{
    fn list_models(
        &self,
    ) -> impl Future<Output = Result<ModelList, ModelListingError>> + WasmCompatSend {
        let lister = P::model_lister(self);
        async move { lister.list_all().await }
    }
}

#[cfg(test)]
mod tests;
