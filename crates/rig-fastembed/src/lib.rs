//! Local embedding model integration backed by `fastembed`.
//!
//! A loaded `fastembed` model is the [`Fastembed`] transport, the runtime
//! behind a local embedding wire ([`text_embeddings`]) that embeds in the
//! calling process. The default feature set enables Hugging Face model
//! downloads and ONNX Runtime binary downloads.
//!
//! ```no_run
//! use rig_fastembed::{Fastembed, FastembedModel};
//!
//! # fn run() -> Result<(), rig_fastembed::FastembedError> {
//! let model = Fastembed::load(&FastembedModel::AllMiniLML6V2Q)?.embedding(&FastembedModel::AllMiniLML6V2Q);
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
use rig_core::driver::{Exchange, Local, Model, Opened, Opening, Step, Transport};
use rig_core::embeddings;
use rig_core::error::ProviderError;
use rig_core::operation::Embedding;
use rig_core::wire::Capabilities;

/// Errors raised while initializing a Fastembed model.
#[derive(Debug, Clone)]
pub enum FastembedError {
    /// The model failed to load, download, or initialize.
    Initialization(String),
}

impl fmt::Display for FastembedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FastembedError::Initialization(message) => {
                write!(f, "Failed to initialize FastEmbed model: {message}")
            }
        }
    }
}

impl StdError for FastembedError {}

/// The local embedding wire of `model`, named `fastembed` and addressing the
/// model by its `fastembed` name, at the width `fastembed`'s metadata gives
/// it. Every [`FastembedModel`] has that metadata, so the width is always
/// known. A model loaded with [`Fastembed::from_user_defined`] is not one of
/// them: declare its width with
/// [`EmbeddingWidth::with_ndims`](rig_core::embeddings::EmbeddingWidth::with_ndims),
/// and a reply of another width fails instead of misdescribing the index.
pub fn text_embeddings(model: &FastembedModel) -> Local<Embedding> {
    // `fastembed` lists every variant (its own `Display` expects to find
    // it); zero, the unknown width, only guards a future release that
    // drops one.
    let ndims = TextEmbedding::get_model_info(model)
        .map(|info| info.dim)
        .unwrap_or_default();
    Local::new("fastembed")
        .with_id(format!("{model:?}"))
        .with_capabilities(Capabilities::embedding(1024, ndims))
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

    /// The embedding model of `model` on this runtime: the
    /// [`text_embeddings`] wire. `model` names what spans and capabilities
    /// report; this runtime embeds with whatever it loaded, so a
    /// user-defined model declares its width with
    /// [`EmbeddingWidth`](rig_core::embeddings::EmbeddingWidth).
    pub fn embedding(&self, model: &FastembedModel) -> Model<Local<Embedding>, Self> {
        Model::new(text_embeddings(model), self.clone())
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
    fn send(&self, texts: Vec<String>, _exchange: Exchange) -> Opening<Step<Embedding>> {
        let embedder = Arc::clone(&self.embedder);
        Opening::new(async move {
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
                    embeddings::EmbeddingResponse::new(embeddings)
                })
                .map_err(|err| ProviderError::Provider(err.to_string()));
            // A failed embed fails the reply, as a transport failure does.
            Ok(Opened::new(futures::stream::iter(
                [embedded.map(Step::End)],
            )))
        })
    }
}
