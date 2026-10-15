//! Local text and image embedding models backed by `fastembed`.
//!
//! A loaded `fastembed` model is a transport, the runtime behind a local
//! embedding wire that embeds in the calling process: [`Fastembed`] for
//! [`text_embeddings`] and [`FastembedImage`] for [`image_embeddings`]. A CLIP
//! pair ([`FastembedModel::ClipVitB32`] and
//! [`FastembedImageModel::ClipVitB32`]) embeds texts and images into one space,
//! so a text query finds images. The default feature set enables Hugging Face
//! model downloads and ONNX Runtime binary downloads.
//!
//! ```no_run
//! use rig_core::Model;
//! use rig_fastembed::{Fastembed, FastembedModel, text_embeddings};
//!
//! # fn run() -> Result<(), rig_fastembed::FastembedError> {
//! let model = Fastembed::load(&FastembedModel::AllMiniLML6V2Q)?.embedding(&FastembedModel::AllMiniLML6V2Q, None)?;
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
pub use fastembed::ImageEmbeddingModel as FastembedImageModel;
use fastembed::{
    ImageEmbedding as ImageRuntime, ImageInitOptionsUserDefined, InitOptionsUserDefined,
    TextEmbedding, UserDefinedEmbeddingModel, UserDefinedImageEmbeddingModel,
};
#[cfg(feature = "hf-hub")]
use fastembed::{ImageInitOptions, InitOptions};
use rig_core::driver::{Exchange, Local, Model, Opened, Opening, Step, Transport};
use rig_core::embeddings;
use rig_core::error::ProviderError;
use rig_core::operation::{Embedding, ImageEmbedding};
use rig_core::wire::Capabilities;

/// Errors raised while resolving or initializing a Fastembed model.
#[derive(Debug, Clone)]
pub enum FastembedError {
    /// `fastembed` has no metadata for the requested model.
    UnknownModel(FastembedModel),
    /// `fastembed` has no metadata for the requested image model.
    UnknownImageModel(FastembedImageModel),
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
            FastembedError::UnknownImageModel(model) => {
                write!(
                    f,
                    "Failed to resolve FastEmbed image model metadata for {model:?}"
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
        .with_capabilities(Capabilities::embedding(1024, ndims)))
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

    /// The embedding model of `model` at `ndims` dimensions, on this
    /// runtime: the [`text_embeddings`] wire, which fails for a model
    /// `fastembed` does not know when `ndims` is `None`. `model` names what
    /// spans and capabilities report; this runtime embeds with whatever it
    /// loaded.
    pub fn embedding(
        &self,
        model: &FastembedModel,
        ndims: Option<usize>,
    ) -> Result<Model<Local<Embedding>, Self>, FastembedError> {
        Ok(Model::new(text_embeddings(model, ndims)?, self.clone()))
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

/// The local image-embedding wire of `model` at `ndims` dimensions, named
/// `fastembed` and addressing the model by its `fastembed` name. `None`
/// takes the width from the model metadata, which errors for models
/// `fastembed` does not know.
pub fn image_embeddings(
    model: &FastembedImageModel,
    ndims: Option<usize>,
) -> Result<Local<ImageEmbedding>, FastembedError> {
    let ndims = match ndims {
        Some(ndims) => ndims,
        None => ImageRuntime::list_supported_models()
            .into_iter()
            .find(|info| &info.model == model)
            .map(|info| info.dim)
            .ok_or_else(|| FastembedError::UnknownImageModel(model.clone()))?,
    };
    Ok(Local::new("fastembed")
        .with_id(format!("{model:?}"))
        .with_capabilities(Capabilities::embedding(256, ndims)))
}

/// A loaded Fastembed image model: the transport that embeds encoded image
/// files (PNG, JPEG, ...) in the calling process. Clones share the loaded
/// model. Pair it with the [`image_embeddings`] wire of the model it
/// loaded, as [`Fastembed`] pairs with [`text_embeddings`]. A file that does
/// not decode as an image fails the whole batch.
#[derive(Clone)]
pub struct FastembedImage {
    embedder: Arc<ImageRuntime>,
}

impl FastembedImage {
    /// Loads `model`, downloading it when necessary and reporting download
    /// progress on standard output.
    #[cfg(feature = "hf-hub")]
    pub fn load(model: &FastembedImageModel) -> Result<Self, FastembedError> {
        let embedder = ImageRuntime::try_new(
            ImageInitOptions::new(model.to_owned()).with_show_download_progress(true),
        )
        .map_err(|err| FastembedError::Initialization(err.to_string()))?;
        Ok(Self {
            embedder: Arc::new(embedder),
        })
    }

    /// The image-embedding model of `model` at `ndims` dimensions, on this
    /// runtime: the [`image_embeddings`] wire. `model` names what spans and
    /// capabilities report; this runtime embeds with whatever it loaded.
    pub fn embedding(
        &self,
        model: &FastembedImageModel,
        ndims: Option<usize>,
    ) -> Result<Model<Local<ImageEmbedding>, Self>, FastembedError> {
        Ok(Model::new(image_embeddings(model, ndims)?, self.clone()))
    }

    /// Loads a caller-supplied ONNX model and its preprocessor
    /// configuration.
    pub fn from_user_defined(
        user_defined_model: UserDefinedImageEmbeddingModel,
    ) -> Result<Self, FastembedError> {
        let embedder = ImageRuntime::try_new_from_user_defined(
            user_defined_model,
            ImageInitOptionsUserDefined::default(),
        )
        .map_err(|err| FastembedError::Initialization(err.to_string()))?;
        Ok(Self {
            embedder: Arc::new(embedder),
        })
    }
}

impl Transport<Local<ImageEmbedding>> for FastembedImage {
    fn send(&self, images: Vec<Vec<u8>>, _exchange: Exchange) -> Opening<Step<ImageEmbedding>> {
        let embedder = Arc::clone(&self.embedder);
        Opening::new(async move {
            let images: Vec<&[u8]> = images.iter().map(Vec::as_slice).collect();
            // The operation pairs each vector with its image's identifier.
            let embedded = embedder
                .embed_bytes(&images, None)
                .map(|vectors| {
                    embeddings::EmbeddingResponse::new(
                        vectors
                            .into_iter()
                            .map(|vector| embeddings::Embedding {
                                document: String::new(),
                                vec: vector.into_iter().map(f64::from).collect(),
                            })
                            .collect(),
                    )
                })
                .map_err(|err| ProviderError::Provider(err.to_string()));
            Ok(Opened::new(futures::stream::iter(
                [embedded.map(Step::End)],
            )))
        })
    }
}
