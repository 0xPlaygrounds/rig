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
/// use rig::client::ModelListingClient;
/// use rig::providers::openai::Client;
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
    /// use rig::client::ModelListingClient;
    /// use rig::providers::openai::Client;
    ///
    /// let openai = Client::from_env();
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
///     H: HttpClientExt + Send + Sync,
/// {
///     type Client = Client<MyProviderExt, H>;
///
///     fn new(client: Self::Client) -> Self {
///         Self { client }
///     }
///
///     async fn list_all(&self) -> Result<ModelList, ModelListingError> {
///         // Fetch all models (handle pagination internally if needed)
///         todo!()
///     }
/// }
/// ```
pub trait ModelLister<H = reqwest::Client>: WasmCompatSend + WasmCompatSync {
    /// The client type associated with this lister
    type Client;

    /// Create a new instance of the lister with the given client
    fn new(client: Self::Client) -> Self;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;

    // Mock implementation for testing
    struct MockModelLister {
        models: Vec<Model>,
    }

    impl MockModelLister {
        fn new(models: Vec<Model>) -> Self {
            Self { models }
        }
    }

    impl ModelLister for MockModelLister {
        type Client = Vec<Model>;

        fn new(client: Self::Client) -> Self {
            Self { models: client }
        }

        fn list_all(
            &self,
        ) -> impl std::future::Future<Output = Result<ModelList, ModelListingError>> + WasmCompatSend
        {
            let models = self.models.clone();
            async move { Ok(ModelList::new(models)) }
        }
    }

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
