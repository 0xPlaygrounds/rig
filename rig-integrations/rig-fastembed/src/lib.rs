use std::sync::Arc;
use std::{error::Error as StdError, fmt};

pub use fastembed::EmbeddingModel as FastembedModel;
use fastembed::{InitOptionsUserDefined, ModelInfo, TextEmbedding, UserDefinedEmbeddingModel};
use rig::embeddings::{self, EmbeddingError};

#[cfg(feature = "hf-hub")]
use fastembed::InitOptions;
#[cfg(feature = "hf-hub")]
use rig::{Embed, embeddings::EmbeddingsBuilder};

/// The `rig-fastembed` client.
///
/// Use this as your main entrypoint for any `rig-fastembed` functionality.
#[derive(Clone)]
pub struct Client;

#[derive(Debug, Clone)]
pub enum FastembedError {
    UnknownModel(FastembedModel),
    Initialization(String),
    UnsupportedMake,
}

impl fmt::Display for FastembedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FastembedError::UnknownModel(model) => {
                write!(
                    f,
                    "Failed to resolve FastEmbed model metadata for {model:?}"
                )
            }
            FastembedError::Initialization(message) => {
                write!(f, "Failed to initialize FastEmbed model: {message}")
            }
            FastembedError::UnsupportedMake => write!(
                f,
                "`EmbeddingModel::make` is not supported for rig-fastembed; construct models via `Client::embedding_model` or `EmbeddingModel::new_from_user_defined`"
            ),
        }
    }
}

impl StdError for FastembedError {}

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}

impl Client {
    /// Create a new `rig-fastembed` client.
    pub fn new() -> Self {
        Self
    }

    /// Create an embedding model with the given name.
    /// Note: default embedding dimension of 0 will be used if model is not known.
    /// If this is the case, it's better to use function `embedding_model_with_ndims`
    ///
    /// # Example
    /// ```
    /// use rig_fastembed::{Client, FastembedModel};
    ///
    /// // Initialize the `rig-fastembed` client
    /// let fastembed_client = rig_fastembed::Client::new();
    ///
    /// let embedding_model = fastembed_client.embedding_model(&FastembedModel::AllMiniLML6V2Q);
    /// ```
    #[cfg(feature = "hf-hub")]
    pub fn embedding_model(
        &self,
        model: &FastembedModel,
    ) -> Result<EmbeddingModel, FastembedError> {
        let ndims = TextEmbedding::get_model_info(model)
            .map(|info| info.dim)
            .map_err(|_| FastembedError::UnknownModel(model.clone()))?;

        EmbeddingModel::new(model, ndims)
    }

    /// Create an embedding builder with the given embedding model.
    ///
    /// # Example
    /// ```
    /// use rig_fastembed::{Client, FastembedModel};
    ///
    /// // Initialize the Fastembed client
    /// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let fastembed_client = Client::new();
    ///
    /// let embeddings = fastembed_client
    ///     .embeddings(&FastembedModel::AllMiniLML6V2Q)?
    ///     .documents(vec![
    ///         "Hello, world!".to_string(),
    ///         "Goodbye, world!".to_string(),
    ///     ])?
    ///     .build()
    ///     .await?;
    /// # let _ = embeddings;
    /// # Ok(())
    /// # }
    /// # let _ = run();
    /// ```
    #[cfg(feature = "hf-hub")]
    pub fn embeddings<D: Embed>(
        &self,
        model: &fastembed::EmbeddingModel,
    ) -> Result<EmbeddingsBuilder<EmbeddingModel, D>, FastembedError> {
        Ok(EmbeddingsBuilder::new(self.embedding_model(model)?))
    }
}

#[derive(Clone)]
pub struct EmbeddingModel {
    embedder: Option<Arc<TextEmbedding>>,
    init_error: Option<FastembedError>,
    pub model: FastembedModel,
    ndims: usize,
}

impl EmbeddingModel {
    #[cfg(feature = "hf-hub")]
    pub fn new(model: &fastembed::EmbeddingModel, ndims: usize) -> Result<Self, FastembedError> {
        let embedder = Arc::new(
            TextEmbedding::try_new(
                InitOptions::new(model.to_owned()).with_show_download_progress(true),
            )
            .map_err(|err| FastembedError::Initialization(err.to_string()))?,
        );

        Ok(Self {
            embedder: Some(embedder),
            init_error: None,
            model: model.to_owned(),
            ndims,
        })
    }

    pub fn new_from_user_defined(
        user_defined_model: UserDefinedEmbeddingModel,
        ndims: usize,
        model_info: &ModelInfo<FastembedModel>,
    ) -> Result<Self, FastembedError> {
        let fastembed_embedding_model = TextEmbedding::try_new_from_user_defined(
            user_defined_model,
            InitOptionsUserDefined::default(),
        )
        .map_err(|err| FastembedError::Initialization(err.to_string()))?;

        let embedder = Arc::new(fastembed_embedding_model);

        Ok(Self {
            embedder: Some(embedder),
            init_error: None,
            model: model_info.model.to_owned(),
            ndims,
        })
    }
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;

    type Client = Client;

    fn make(_: &Self::Client, _: impl Into<String>, _: Option<usize>) -> Self {
        Self {
            embedder: None,
            init_error: Some(FastembedError::UnsupportedMake),
            model: FastembedModel::AllMiniLML6V2Q,
            ndims: 0,
        }
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let Some(embedder) = &self.embedder else {
            let message = self
                .init_error
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_else(|| "FastEmbed model initialization failed".to_string());
            return Err(EmbeddingError::ProviderError(message));
        };

        let documents_as_strings: Vec<String> = documents.into_iter().collect();

        let documents_as_vec = embedder
            .embed(documents_as_strings.clone(), None)
            .map_err(|err| EmbeddingError::ProviderError(err.to_string()))?;

        let docs = documents_as_strings
            .into_iter()
            .zip(documents_as_vec)
            .map(|(document, embedding)| embeddings::Embedding {
                document,
                vec: embedding.into_iter().map(|f| f as f64).collect(),
            })
            .collect::<Vec<embeddings::Embedding>>();

        Ok(docs)
    }
}
