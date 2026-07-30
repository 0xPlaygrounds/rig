//! Local embedding model integration backed by `fastembed`.
//!
//! The crate's face is [`functions`]: an [`EmbeddingConfig`] record that says
//! *which* model to load, a loaded [`EmbeddingModel`] handle, and the free
//! function [`functions::embed`]. There is no client type and no model trait —
//! FastEmbed runs an in-process ONNX session, so the only meaningful runtime
//! value is the loaded handle itself.
//!
//! ```no_run
//! use rig_fastembed::{EmbeddingConfig, FastembedModel, functions};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = EmbeddingConfig::new(FastembedModel::AllMiniLML6V2Q).load()?;
//! let response = functions::embed(&model, vec!["Hello, world!".to_string()])?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```
//!
//! For whole documents, pair [`functions::embed`] with
//! [`rig_core::embeddings::embed_documents`], which chunks and re-associates
//! embeddings the way the removed `EmbeddingsBuilder` used to.
//!
//! `rig-fastembed` is native-only and does not target `wasm32-unknown-unknown`.
//! The root `rig` facade re-exports this crate as `rig::fastembed` when one of
//! its Fastembed features is enabled.

use std::sync::Arc;
use std::{error::Error as StdError, fmt};

pub use fastembed::EmbeddingModel as FastembedModel;
use fastembed::{InitOptionsUserDefined, ModelInfo, TextEmbedding, UserDefinedEmbeddingModel};
use rig_core::embeddings::{self, EmbeddingError};

#[cfg(feature = "hf-hub")]
use fastembed::InitOptions;

pub use functions::{DESCRIPTOR, MAX_DOCUMENTS};

/// Failures raised while resolving or loading a FastEmbed model.
#[derive(Debug, Clone)]
pub enum FastembedError {
    /// `fastembed` has no metadata (and therefore no dimensionality) for the model.
    UnknownModel(FastembedModel),
    /// The ONNX session could not be constructed.
    Initialization(String),
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
        }
    }
}

impl StdError for FastembedError {}

impl From<FastembedError> for EmbeddingError {
    fn from(error: FastembedError) -> Self {
        EmbeddingError::ProviderError(error.to_string())
    }
}

/// Plain-data description of which FastEmbed model to load.
///
/// This is the config half of the crate's data-oriented face. It is not
/// `serde`-derived because [`FastembedModel`] is a `fastembed` enum rather
/// than a wire value; construct it in code and call [`Self::load`] to get the
/// [`EmbeddingModel`] handle the free functions take.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct EmbeddingConfig {
    /// Hugging Face model to download and run.
    pub model: FastembedModel,
    /// Embedding dimensionality; `None` resolves it from model metadata.
    pub ndims: Option<usize>,
    /// Whether the Hugging Face download prints progress.
    pub show_download_progress: bool,
}

impl EmbeddingConfig {
    /// Config for `model`, resolving its dimensionality from metadata at load time.
    pub fn new(model: FastembedModel) -> Self {
        Self {
            model,
            ndims: None,
            show_download_progress: true,
        }
    }

    /// Pin the reported embedding dimensionality instead of resolving it.
    pub fn with_ndims(mut self, ndims: usize) -> Self {
        self.ndims = Some(ndims);
        self
    }

    /// Silence (or re-enable) the Hugging Face download progress bar.
    pub fn with_download_progress(mut self, show: bool) -> Self {
        self.show_download_progress = show;
        self
    }

    /// Download (if needed) and load the model described by this config.
    ///
    /// # Errors
    /// [`FastembedError::UnknownModel`] when `ndims` is unset and metadata is
    /// unavailable; [`FastembedError::Initialization`] when the ONNX session
    /// cannot be created.
    #[cfg(feature = "hf-hub")]
    pub fn load(&self) -> Result<EmbeddingModel, FastembedError> {
        let ndims = match self.ndims {
            Some(ndims) => ndims,
            None => TextEmbedding::get_model_info(&self.model)
                .map(|info| info.dim)
                .map_err(|_| FastembedError::UnknownModel(self.model.clone()))?,
        };

        let embedder = Arc::new(
            TextEmbedding::try_new(
                InitOptions::new(self.model.clone())
                    .with_show_download_progress(self.show_download_progress),
            )
            .map_err(|err| FastembedError::Initialization(err.to_string()))?,
        );

        Ok(EmbeddingModel {
            embedder,
            model: self.model.clone(),
            ndims,
        })
    }
}

/// A loaded FastEmbed model: the runtime handle [`functions::embed`] takes.
///
/// Cheap to clone (the ONNX session is shared).
#[derive(Clone)]
pub struct EmbeddingModel {
    embedder: Arc<TextEmbedding>,
    /// Which FastEmbed model these weights are.
    pub model: FastembedModel,
    ndims: usize,
}

impl EmbeddingModel {
    /// Load `model` from Hugging Face with an explicit dimensionality.
    ///
    /// Shorthand for [`EmbeddingConfig::new`] + [`EmbeddingConfig::with_ndims`]
    /// + [`EmbeddingConfig::load`].
    #[cfg(feature = "hf-hub")]
    pub fn new(model: &FastembedModel, ndims: usize) -> Result<Self, FastembedError> {
        EmbeddingConfig::new(model.clone()).with_ndims(ndims).load()
    }

    /// Load a model from caller-supplied ONNX and tokenizer bytes.
    pub fn new_from_user_defined(
        user_defined_model: UserDefinedEmbeddingModel,
        ndims: usize,
        model_info: &ModelInfo<FastembedModel>,
    ) -> Result<Self, FastembedError> {
        let embedder = TextEmbedding::try_new_from_user_defined(
            user_defined_model,
            InitOptionsUserDefined::default(),
        )
        .map_err(|err| FastembedError::Initialization(err.to_string()))?;

        Ok(Self {
            embedder: Arc::new(embedder),
            model: model_info.model.clone(),
            ndims,
        })
    }

    /// Dimensionality of the vectors this model produces.
    pub fn ndims(&self) -> usize {
        self.ndims
    }
}

/// FastEmbed as free functions (data-oriented face).
///
/// FastEmbed runs local model weights (an in-process ONNX session), so
/// unlike the HTTP providers there is no serde config that can honestly
/// describe a connection: the "handle" is the loaded [`EmbeddingModel`]
/// itself. The functions face therefore takes that handle directly, built
/// from an [`EmbeddingConfig`] (or
/// [`EmbeddingModel::new_from_user_defined`] for local weights).
pub mod functions {
    use super::{EmbeddingModel, embeddings};
    use rig_core::embeddings::EmbeddingError;
    use rig_core::providers::descriptor::ProviderDescriptor;

    /// Largest batch handed to a single FastEmbed inference call.
    pub const MAX_DOCUMENTS: usize = 1024;

    /// FastEmbed's capability sheet: embeddings only, no chat.
    pub const DESCRIPTOR: ProviderDescriptor =
        ProviderDescriptor::named("fastembed").with_max_embedding_documents(MAX_DOCUMENTS);

    /// Embed `texts` through the loaded FastEmbed model, in input order.
    ///
    /// Local inference reports no token usage. Synchronous: FastEmbed
    /// computes on the calling thread.
    pub fn embed(
        model: &EmbeddingModel,
        texts: Vec<String>,
    ) -> Result<rig_core::embeddings::EmbeddingResponse, EmbeddingError> {
        let documents_as_vec = model
            .embedder
            .embed(texts.clone(), None)
            .map_err(|err| EmbeddingError::ProviderError(err.to_string()))?;

        let embeddings = texts
            .into_iter()
            .zip(documents_as_vec)
            .map(|(document, embedding)| embeddings::Embedding {
                document,
                vec: embedding.into_iter().map(|f| f as f64).collect(),
            })
            .collect::<Vec<embeddings::Embedding>>();

        Ok(rig_core::embeddings::EmbeddingResponse {
            embeddings,
            usage: rig_core::completion::Usage::new(),
        })
    }

    /// Embed one text, returning its [`embeddings::Embedding`].
    ///
    /// Convenience for the common "embed the query" call site.
    pub fn embed_text(
        model: &EmbeddingModel,
        text: impl Into<String>,
    ) -> Result<embeddings::Embedding, EmbeddingError> {
        let mut response = embed(model, vec![text.into()])?;
        if response.embeddings.len() == 1 {
            Ok(response.embeddings.remove(0))
        } else {
            Err(EmbeddingError::ResponseError(format!(
                "FastEmbed returned {} embeddings for a single text",
                response.embeddings.len()
            )))
        }
    }

    /// Embed caller-defined batches, returning one order-aligned
    /// [`rig_core::OneOrMany`] group per input batch plus (zero) usage.
    pub fn embed_batches(
        model: &EmbeddingModel,
        texts: Vec<Vec<String>>,
    ) -> Result<
        (
            Vec<rig_core::OneOrMany<embeddings::Embedding>>,
            rig_core::completion::Usage,
        ),
        EmbeddingError,
    > {
        let counts: Vec<usize> = texts.iter().map(Vec::len).collect();
        let flat: Vec<String> = texts.into_iter().flatten().collect();
        let response = embed(model, flat)?;
        let groups = rig_core::embeddings::batching::group_batches(&counts, response.embeddings)?;
        Ok((groups, response.usage))
    }
}
