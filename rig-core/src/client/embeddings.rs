use crate::Embed;
use crate::client::{AsEmbeddings, ProviderClient};
use crate::embeddings::embedding::EmbeddingModelDyn;
use crate::embeddings::{EmbeddingModel, EmbeddingsBuilder};

/// A provider client with vector embedding capabilities.
///
/// This trait extends [`ProviderClient`] to provide text-to-vector embedding functionality.
/// Providers that implement this trait can create embedding models for semantic search,
/// similarity comparison, and other vector-based operations.
///
/// # When to Implement
///
/// Implement this trait for provider clients that support:
/// - Text to vector embeddings
/// - Document embeddings for search
/// - Semantic similarity calculations
/// - Vector database integration
///
/// # Examples
///
/// ```no_run
/// use rig::prelude::*;
/// use rig::providers::openai::{Client, self};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = Client::new("api-key");
///
/// // Create an embedding model
/// let model = client.embedding_model(openai::TEXT_EMBEDDING_3_LARGE);
///
/// // Embed a single text
/// let embedding = model.embed_text("Hello, world!").await?;
/// println!("Vector dimension: {}", embedding.vec.len());
///
/// // Or build embeddings for multiple documents
/// let embeddings = client.embeddings(openai::TEXT_EMBEDDING_3_LARGE)
///     .documents(vec![
///         "First document".to_string(),
///         "Second document".to_string(),
///     ])?
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
///
/// # See Also
///
/// - [`crate::embeddings::EmbeddingModel`] - The model trait for creating embeddings
/// - [`crate::embeddings::EmbeddingsBuilder`] - Builder for batch embedding operations
/// - [`EmbeddingsClientDyn`] - Dynamic dispatch version for runtime polymorphism
pub trait EmbeddingsClient: ProviderClient + Clone {
    /// The type of EmbeddingModel used by the Client
    type EmbeddingModel: EmbeddingModel;

    /// Creates an embedding model with the specified model identifier.
    ///
    /// This method constructs an embedding model that can convert text into vector embeddings.
    /// If the model is not recognized, a default dimension of 0 is used. For unknown models,
    /// prefer using [`embedding_model_with_ndims`](Self::embedding_model_with_ndims) to specify the dimension explicitly.
    ///
    /// # Arguments
    ///
    /// * `model` - The model identifier (e.g., "text-embedding-3-large", "embed-english-v2.0")
    ///
    /// # Returns
    ///
    /// An embedding model that can be used to generate vector embeddings.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rig::prelude::*;
    /// use rig::providers::openai::{Client, self};
    /// use rig::embeddings::EmbeddingModel;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::new("your-api-key");
    /// let model = client.embedding_model(openai::TEXT_EMBEDDING_3_LARGE);
    ///
    /// // Use the model to generate embeddings
    /// let embedding = model.embed_text("Hello, world!").await?;
    /// println!("Embedding dimension: {}", embedding.vec.len());
    /// # Ok(())
    /// # }
    /// ```
    fn embedding_model(&self, model: &str) -> Self::EmbeddingModel;

    /// Creates an embedding model with explicit dimension specification.
    ///
    /// Use this method when working with models that are not pre-configured in Rig
    /// or when you need to explicitly control the embedding dimension.
    ///
    /// # Arguments
    ///
    /// * `model` - The model identifier (e.g., "custom-model", "text-embedding-3-large")
    /// * `ndims` - The number of dimensions in the generated embeddings
    ///
    /// # Returns
    ///
    /// An embedding model configured with the specified dimension.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rig::prelude::*;
    /// use rig::providers::openai::{Client, self};
    /// use rig::embeddings::EmbeddingModel;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::new("your-api-key");
    /// let model = client.embedding_model_with_ndims("custom-model", 1536);
    ///
    /// let embedding = model.embed_text("Test text").await?;
    /// assert_eq!(embedding.vec.len(), 1536);
    /// # Ok(())
    /// # }
    /// ```
    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> Self::EmbeddingModel;

    /// Creates an embeddings builder for batch embedding operations.
    ///
    /// The embeddings builder allows you to embed multiple documents at once,
    /// which is more efficient than embedding them individually.
    ///
    /// # Type Parameters
    ///
    /// * `D` - The document type that implements [`Embed`]
    ///
    /// # Arguments
    ///
    /// * `model` - The model identifier (e.g., "text-embedding-3-large")
    ///
    /// # Returns
    ///
    /// An [`EmbeddingsBuilder`] that can be used to add documents and build embeddings.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rig::prelude::*;
    /// use rig::providers::openai::{Client, self};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::new("your-api-key");
    ///
    /// let embeddings = client.embeddings(openai::TEXT_EMBEDDING_3_LARGE)
    ///     .documents(vec![
    ///         "Hello, world!".to_string(),
    ///         "Goodbye, world!".to_string(),
    ///     ])?
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    fn embeddings<D: Embed>(&self, model: &str) -> EmbeddingsBuilder<Self::EmbeddingModel, D> {
        EmbeddingsBuilder::new(self.embedding_model(model))
    }

    /// Creates an embeddings builder with explicit dimension specification.
    ///
    /// This is equivalent to [`embeddings`](Self::embeddings) but allows you to specify
    /// the embedding dimension explicitly for unknown models.
    ///
    /// # Type Parameters
    ///
    /// * `D` - The document type that implements [`Embed`]
    ///
    /// # Arguments
    ///
    /// * `model` - The model identifier
    /// * `ndims` - The number of dimensions in the generated embeddings
    ///
    /// # Returns
    ///
    /// An [`EmbeddingsBuilder`] configured with the specified dimension.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rig::prelude::*;
    /// use rig::providers::openai::{Client, self};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::new("your-api-key");
    ///
    /// let embeddings = client.embeddings_with_ndims(openai::TEXT_EMBEDDING_3_LARGE, 3072)
    ///     .documents(vec![
    ///         "Hello, world!".to_string(),
    ///         "Goodbye, world!".to_string(),
    ///     ])?
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    fn embeddings_with_ndims<D: Embed>(
        &self,
        model: &str,
        ndims: usize,
    ) -> EmbeddingsBuilder<Self::EmbeddingModel, D> {
        EmbeddingsBuilder::new(self.embedding_model_with_ndims(model, ndims))
    }
}

/// Dynamic dispatch version of [`EmbeddingsClient`].
///
/// This trait provides the same functionality as [`EmbeddingsClient`] but returns
/// trait objects instead of associated types, enabling runtime polymorphism.
/// It is automatically implemented for all types that implement [`EmbeddingsClient`].
///
/// # When to Use
///
/// Use this trait when you need to work with embedding clients of different types
/// at runtime, such as in the [`DynClientBuilder`](crate::client::builder::DynClientBuilder).
pub trait EmbeddingsClientDyn: ProviderClient {
    /// Creates a boxed embedding model with the specified model identifier.
    ///
    /// Note: A default embedding dimension of 0 is used if the model is not recognized.
    /// For unknown models, prefer using [`embedding_model_with_ndims`](Self::embedding_model_with_ndims).
    ///
    /// Returns a trait object that can be used for dynamic dispatch.
    fn embedding_model<'a>(&self, model: &str) -> Box<dyn EmbeddingModelDyn + 'a>;

    /// Creates a boxed embedding model with explicit dimension specification.
    ///
    /// Returns a trait object configured with the specified dimension.
    fn embedding_model_with_ndims<'a>(
        &self,
        model: &str,
        ndims: usize,
    ) -> Box<dyn EmbeddingModelDyn + 'a>;
}

impl<M, T> EmbeddingsClientDyn for T
where
    T: EmbeddingsClient<EmbeddingModel = M>,
    M: EmbeddingModel + 'static,
{
    fn embedding_model<'a>(&self, model: &str) -> Box<dyn EmbeddingModelDyn + 'a> {
        Box::new(self.embedding_model(model))
    }

    fn embedding_model_with_ndims<'a>(
        &self,
        model: &str,
        ndims: usize,
    ) -> Box<dyn EmbeddingModelDyn + 'a> {
        Box::new(self.embedding_model_with_ndims(model, ndims))
    }
}

impl<T> AsEmbeddings for T
where
    T: EmbeddingsClientDyn + Clone + 'static,
{
    fn as_embeddings(&self) -> Option<Box<dyn EmbeddingsClientDyn>> {
        Some(Box::new(self.clone()))
    }
}
