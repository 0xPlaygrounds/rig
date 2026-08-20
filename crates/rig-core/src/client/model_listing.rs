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
    /// use rig_core::client::{ModelListingClient, ProviderClient};
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
/// - `H`: The HTTP client type (typically `reqwest::Client`)
///
/// # Example Implementation
///
/// ```rust,ignore
/// use crate::client::ModelLister;
/// use crate::model::{Model, ModelList, ModelListingError};
///
/// struct MyProviderModelLister<H> {
///     client: Client<MyProviderExt, H>,
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
/// // Construction is separate: the public hook the blanket
/// // `ModelListingClient` impl over `Client<Ext, H>` calls.
/// impl<H> ConstructModelLister<Client<MyProviderExt, H>> for MyProviderModelLister<H>
/// where
///     H: Clone,
/// {
///     fn construct(client: &Client<MyProviderExt, H>) -> Self {
///         Self { client: client.clone() }
///     }
/// }
/// ```
///
/// `H` stays a parameter of this trait: it is the transport, not a provider
/// leak, and the listing request is written against it.
pub trait ModelLister<H = reqwest::Client>: WasmCompatSend + WasmCompatSync {
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

/// Construction hook for the blanket [`ModelListingClient`] implementation
/// over [`crate::client::Client`] — the model-listing twin of
/// [`crate::client::ConstructCompletionModel`], and the last construction
/// associated type to leave a trait in this crate.
///
/// Public for the same reason as the others: an out-of-tree provider extension
/// built on the generic `Client<Ext, H>` cannot implement
/// [`ModelListingClient`] for that foreign type (orphan rule), so it
/// implements this trait on its own lister type and the blanket
/// implementation supplies `list_models`. Takes `&C`, like every other
/// `Construct*` hook; every implementation clones the client anyway.
pub trait ConstructModelLister<C>: Sized {
    /// Build this lister from its provider client.
    fn construct(client: &C) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;
    use crate::test_utils::MockModelLister;

    #[tokio::test]
    async fn test_model_lister_list_all() {
        let models = vec![
            Model::new("gpt-4", "GPT-4"),
            Model::new("gpt-3.5-turbo", "GPT-3.5 Turbo"),
        ];
        let lister = MockModelLister::new(models);

        let result = lister.list_all().await.unwrap();
        assert_eq!(result.len(), 2);
    }
}
