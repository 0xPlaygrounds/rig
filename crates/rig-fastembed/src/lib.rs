//! Local embedding model integration backed by `fastembed`.
//!
//! A loaded `fastembed` model is the [`Fastembed`] transport, the runtime
//! behind a local embedding wire ([`text_embeddings`]) that embeds in the
//! calling process. The default feature set enables Hugging Face model
//! downloads and ONNX Runtime binary downloads.
//!
//! ```no_run
//! use rig_core::Model;
//! use rig_fastembed::{Fastembed, FastembedModel, text_embeddings};
//!
//! # fn run() -> Result<(), rig_fastembed::FastembedError> {
//! let model = Model::new(
//!     text_embeddings(&FastembedModel::AllMiniLML6V2Q, None)?,
//!     Fastembed::load(&FastembedModel::AllMiniLML6V2Q)?,
//! );
//! # let _ = model;
//! # Ok(())
//! # }
//! ```
//!
//! `rig-fastembed` is native-only and does not target `wasm32-unknown-unknown`.
//! The root `rig` facade re-exports this crate as `rig::fastembed` when one of
//! its Fastembed features is enabled.

use std::sync::Arc;
use std::{error::Error as StdError, fmt};

pub use fastembed::EmbeddingModel as FastembedModel;
#[cfg(feature = "hf-hub")]
use fastembed::InitOptions;
use fastembed::{InitOptionsUserDefined, TextEmbedding, UserDefinedEmbeddingModel};
use rig_core::driver::{Local, Observation, Opened, Transport};
use rig_core::embeddings;
use rig_core::error::ProviderError;
use rig_core::operation::{Embedding, EmbeddingCapabilities};
use rig_core::wire::Mode;

/// Errors raised while resolving or initializing a Fastembed model.
#[derive(Debug, Clone)]
pub enum FastembedError {
    /// `fastembed` has no metadata for the requested model.
    UnknownModel(FastembedModel),
    /// The model failed to load, download, or initialize.
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

/// The local embedding wire of `model` at `ndims` dimensions, named
/// `fastembed` and addressing the model by its `fastembed` name. `None`
/// takes the width from the model metadata, which errors for models
/// `fastembed` does not know.
pub fn text_embeddings(
    model: &FastembedModel,
    ndims: Option<usize>,
) -> Result<Local<Embedding>, FastembedError> {
    let ndims = match ndims {
        Some(ndims) => ndims,
        None => TextEmbedding::get_model_info(model)
            .map(|info| info.dim)
            .map_err(|_| FastembedError::UnknownModel(model.clone()))?,
    };
    Ok(Local::new("fastembed")
        .with_id(format!("{model:?}"))
        .with_capabilities(EmbeddingCapabilities::new(1024, ndims)))
}

/// A loaded Fastembed model: the transport that embeds in the calling
/// process. Clones share the loaded model. Pair it with the
/// [`text_embeddings`] wire of the model it loaded: the wire names the model
/// and width that spans and capabilities report, and the transport embeds
/// with whatever it loaded. In-process execution reports no raw payload,
/// usage, or request id.
#[derive(Clone)]
pub struct Fastembed {
    embedder: Arc<TextEmbedding>,
}

impl Fastembed {
    /// Loads `model`, downloading it when necessary and reporting download
    /// progress on standard output.
    #[cfg(feature = "hf-hub")]
    pub fn load(model: &FastembedModel) -> Result<Self, FastembedError> {
        let embedder = TextEmbedding::try_new(
            InitOptions::new(model.to_owned()).with_show_download_progress(true),
        )
        .map_err(|err| FastembedError::Initialization(err.to_string()))?;
        Ok(Self {
            embedder: Arc::new(embedder),
        })
    }

    /// Loads a caller-supplied ONNX model.
    pub fn from_user_defined(
        user_defined_model: UserDefinedEmbeddingModel,
    ) -> Result<Self, FastembedError> {
        let embedder = TextEmbedding::try_new_from_user_defined(
            user_defined_model,
            InitOptionsUserDefined::default(),
        )
        .map_err(|err| FastembedError::Initialization(err.to_string()))?;
        Ok(Self {
            embedder: Arc::new(embedder),
        })
    }
}

impl Transport<Local<Embedding>> for Fastembed {
    fn send(
        &self,
        texts: Vec<String>,
        _mode: Mode,
        _observation: Option<Observation>,
    ) -> Result<
        impl Future<Output = Opened<Vec<String>, Result<embeddings::EmbeddingResponse, ProviderError>>>
        + Send
        + 'static
        + use<>,
        ProviderError,
    > {
        let embedder = Arc::clone(&self.embedder);
        Ok(async move {
            let embedded = embedder
                .embed(texts.iter().map(String::as_str).collect(), None)
                .map(|vectors| {
                    let embeddings = texts
                        .into_iter()
                        .zip(vectors)
                        .map(|(document, vector)| embeddings::Embedding {
                            document,
                            vec: vector.into_iter().map(f64::from).collect(),
                        })
                        .collect();
                    embeddings::EmbeddingResponse::new(embeddings, "fastembed")
                })
                .map_err(|err| ProviderError::Provider(err.to_string()));
            // A failed embed fails the reply, as a transport failure does.
            Opened::new(futures::stream::iter([embedded.map(Ok)]))
        })
    }
}
