//! Local embedding model integration backed by `fastembed`.
//!
//! A loaded `fastembed` model is the [`Fastembed`] transport; the
//! [`TextEmbeddings`] wire embeds through it in the calling process. The
//! default feature set enables Hugging Face model downloads and ONNX Runtime
//! binary downloads.
//!
//! ```no_run
//! use rig_core::Model;
//! use rig_fastembed::{Fastembed, FastembedModel, TextEmbeddings};
//!
//! # fn run() -> Result<(), rig_fastembed::FastembedError> {
//! let model = Model::new(
//!     TextEmbeddings::for_model(&FastembedModel::AllMiniLML6V2Q, None)?,
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
use rig_core::driver::{Observation, Opened, Transport};
use rig_core::embeddings;
use rig_core::error::{EncodeError, ProviderError};
use rig_core::operation::{Embedding, EmbeddingCapabilities, One};
use rig_core::providers::internal::wire::{self, TypedEvent, WireEvent};
use rig_core::wire::{Decoder, Mode, Sink, Wire};

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

/// The text-embedding endpoint of one Fastembed model, at a width.
#[derive(Clone, Debug, PartialEq)]
pub struct TextEmbeddings {
    pub model: FastembedModel,
    pub ndims: usize,
    /// The model's name as telemetry spells it.
    label: String,
}

impl TextEmbeddings {
    /// The endpoint for `model` at `ndims` dimensions.
    pub fn new(model: FastembedModel, ndims: usize) -> Self {
        Self {
            label: format!("{model:?}"),
            model,
            ndims,
        }
    }

    /// The endpoint for `model` at width `ndims`. `None` takes the width
    /// from the model metadata, which errors for models `fastembed` does not
    /// know.
    pub fn for_model(model: &FastembedModel, ndims: Option<usize>) -> Result<Self, FastembedError> {
        let ndims = match ndims {
            Some(ndims) => ndims,
            None => TextEmbedding::get_model_info(model)
                .map(|info| info.dim)
                .map_err(|_| FastembedError::UnknownModel(model.clone()))?,
        };
        Ok(Self::new(model.clone(), ndims))
    }
}

/// A loaded Fastembed model: the transport that embeds in the calling
/// process. Clones share the loaded model.
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

impl Wire for TextEmbeddings {
    type Op = Embedding;
    type Payload = Vec<String>;
    type Frame = (Vec<String>, Vec<Vec<f32>>);
    type Decoder = FastembedDecoder;

    fn name(&self) -> &str {
        "fastembed"
    }

    fn model(&self) -> Option<&str> {
        Some(&self.label)
    }

    fn capabilities(&self) -> EmbeddingCapabilities {
        EmbeddingCapabilities::new(1024, self.ndims)
    }

    fn encode(&self, texts: Vec<String>, _mode: Mode) -> Result<Vec<String>, EncodeError> {
        Ok(texts)
    }

    fn decoder(&self, _mode: Mode) -> FastembedDecoder {
        FastembedDecoder
    }
}

impl Transport<TextEmbeddings> for Fastembed {
    fn send(
        &self,
        texts: Vec<String>,
        _mode: Mode,
        _observation: Option<Observation>,
    ) -> Result<
        impl Future<Output = Opened<Vec<String>, (Vec<String>, Vec<Vec<f32>>)>> + Send + 'static + use<>,
        ProviderError,
    > {
        let embedder = Arc::clone(&self.embedder);
        Ok(async move {
            let embedded = embedder
                .embed(texts.iter().map(String::as_str).collect(), None)
                .map(|vectors| (texts, vectors))
                .map_err(|err| ProviderError::Provider(err.to_string()));
            Opened::new(futures::stream::iter([embedded]))
        })
    }
}

/// Pairs each text with its vector. In-process execution reports no raw
/// payload, usage, or request id.
pub struct FastembedDecoder;

impl Decoder<Embedding, (Vec<String>, Vec<Vec<f32>>)> for FastembedDecoder {
    type Event = (Vec<String>, Vec<Vec<f32>>);

    fn classify(&self, frame: Self::Event) -> WireEvent<Self::Event> {
        wire::classify_typed_event(TypedEvent::Modeled(frame))
    }

    fn interpret(&mut self, (texts, vectors): Self::Event, out: &mut One<Embedding>) {
        let embeddings = texts
            .into_iter()
            .zip(vectors)
            .map(|(document, vector)| embeddings::Embedding {
                document,
                vec: vector.into_iter().map(f64::from).collect(),
            })
            .collect();
        out.push(Ok(embeddings::EmbeddingResponse::new(
            embeddings,
            "fastembed",
        )));
    }
}
