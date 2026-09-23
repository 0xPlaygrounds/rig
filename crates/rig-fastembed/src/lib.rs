//! Local embedding model integration backed by `fastembed`.
//!
//! This crate adapts `fastembed` text embedding models to Rig's
//! [`rig_core::embeddings::EmbeddingModel`] trait. The default feature set
//! enables Hugging Face model downloads and ONNX Runtime binary downloads.
//!
//! `rig-fastembed` is native-only and does not target `wasm32-unknown-unknown`.
//! The root `rig` facade re-exports this crate as `rig::fastembed` when one of
//! its Fastembed features is enabled.

use std::sync::Arc;
use std::{error::Error as StdError, fmt};

pub use fastembed::EmbeddingModel as FastembedModel;
use fastembed::{InitOptionsUserDefined, ModelInfo, TextEmbedding, UserDefinedEmbeddingModel};
use rig_core::embeddings;
use rig_core::error::ProviderError;

#[cfg(feature = "hf-hub")]
use fastembed::InitOptions;
#[cfg(feature = "hf-hub")]
use rig_core::{Embed, embeddings::EmbeddingsBuilder};

/// Entry point for constructing local Fastembed embedding models.
#[derive(Clone)]
pub struct Client;

/// Errors raised while resolving or initializing a Fastembed model.
#[derive(Debug, Clone)]
pub enum FastembedError {
    /// `fastembed` has no metadata for the requested model.
    UnknownModel(FastembedModel),
    /// The model failed to load, download, or initialize.
    Initialization(String),
    /// Construction through the generic model factory is unavailable.
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
                "`EmbeddingModel::make` is not supported for rig-fastembed; construct models via `Client::embedding` or `EmbeddingModel::new_from_user_defined`"
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
    pub fn new() -> Self {
        Self
    }

    /// Loads `model`, downloading it when necessary, and returns an embedding
    /// model of width `ndims`. `None` takes the width from the model metadata,
    /// which errors for models `fastembed` does not know.
    ///
    /// # Example
    /// ```
    /// use rig_fastembed::{Client, FastembedModel};
    ///
    /// let fastembed = Client::new();
    ///
    /// let model = fastembed.embedding(&FastembedModel::AllMiniLML6V2Q, None);
    /// ```
    #[cfg(feature = "hf-hub")]
    pub fn embedding(
        &self,
        model: &FastembedModel,
        ndims: Option<usize>,
    ) -> Result<EmbeddingModel, FastembedError> {
        let ndims = match ndims {
            Some(ndims) => ndims,
            None => TextEmbedding::get_model_info(model)
                .map(|info| info.dim)
                .map_err(|_| FastembedError::UnknownModel(model.clone()))?,
        };

        EmbeddingModel::new(model, ndims)
    }

    /// Loads `model` with its documented width and returns a builder over it.
    ///
    /// # Example
    /// ```
    /// use rig_fastembed::{Client, FastembedModel};
    ///
    /// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let fastembed = Client::new();
    ///
    /// let embeddings = fastembed
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
        model: &FastembedModel,
    ) -> Result<EmbeddingsBuilder<EmbeddingModel, D>, FastembedError> {
        Ok(EmbeddingsBuilder::new(self.embedding(model, None)?))
    }
}

/// Local embedding model executing in the calling process.
#[derive(Clone)]
pub struct EmbeddingModel {
    embedder: Option<Arc<TextEmbedding>>,
    init_error: Option<FastembedError>,
    pub model: FastembedModel,
    ndims: usize,
}

impl EmbeddingModel {
    /// Loads `model`, reporting download progress on standard output.
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

    /// Loads a caller-supplied ONNX model, taking only its name from `model_info`.
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
            model: model_info.model.clone(),
            ndims,
        })
    }
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    fn max_documents(&self) -> usize {
        1024
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts_response(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<embeddings::EmbeddingResponse, ProviderError> {
        rig_core::telemetry::instrument_modality::<rig_core::operation::Embedding, _>(
            "fastembed",
            &format!("{:?}", self.model),
            async {
                let Some(embedder) = &self.embedder else {
                    let message = self.init_error.as_ref().map_or_else(
                        || "FastEmbed model initialization failed".to_string(),
                        ToString::to_string,
                    );
                    return Err(ProviderError::Provider(message));
                };

                let documents_as_strings: Vec<String> = documents.into_iter().collect();

                let documents_as_vec = embedder
                    .embed(
                        documents_as_strings.iter().map(String::as_str).collect(),
                        None,
                    )
                    .map_err(|err| ProviderError::Provider(err.to_string()))?;

                let docs = documents_as_strings
                    .into_iter()
                    .zip(documents_as_vec)
                    .map(|(document, embedding)| embeddings::Embedding {
                        document,
                        vec: embedding.into_iter().map(|f| f as f64).collect(),
                    })
                    .collect::<Vec<embeddings::Embedding>>();

                // In-process execution reports no raw payload, usage, or request id.
                Ok(embeddings::EmbeddingResponse::new(docs, "fastembed"))
            },
        )
        .await
    }
}
