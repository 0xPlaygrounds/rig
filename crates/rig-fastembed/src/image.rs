//! Local image embeddings from encoded image bytes through Fastembed.
//!
//! ```no_run
//! use rig_fastembed::{FastembedImage, FastembedImageModel};
//!
//! # #[cfg(feature = "hf-hub")]
//! # fn run() -> Result<(), rig_fastembed::FastembedError> {
//! let runtime = FastembedImage::load(&FastembedImageModel::ClipVitB32)?;
//! let model = runtime.embedding(&FastembedImageModel::ClipVitB32, None);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

pub use fastembed::ImageEmbeddingModel as FastembedImageModel;
#[cfg(feature = "hf-hub")]
use fastembed::ImageInitOptions;
use fastembed::{
    ImageEmbedding as Embedder, ImageInitOptionsUserDefined, UserDefinedImageEmbeddingModel,
};
use rig_core::driver::{Exchange, Local, Model, Opened, Opening, Step, Transport};
use rig_core::embeddings::{Embedding, ImageEmbeddingResponse};
use rig_core::error::ProviderError;
use rig_core::operation::ImageEmbedding;
use rig_core::wire::Capabilities;

use crate::FastembedError;

/// The local image embedding wire for `model`, with the expected vector width
/// `ndims`. `None` uses the model metadata. A positive width is checked against
/// every returned vector; it does not resize the model's output.
pub fn image_embeddings(
    model: &FastembedImageModel,
    ndims: Option<usize>,
) -> Local<ImageEmbedding> {
    let ndims = ndims.unwrap_or_else(|| Embedder::get_model_info(model).dim);
    Local::new("fastembed")
        .with_id(format!("{model:?}"))
        .with_capabilities(Capabilities::embedding(1024, ndims).declaring(Some(ndims)))
}

/// A loaded Fastembed image model. Clones share the loaded model. Pair this
/// transport with the wire for the model it loaded: the wire names the model
/// and expected width, while the transport embeds with whatever it loaded.
/// Inputs are encoded image file bytes. In-process execution reports no raw
/// payload, usage, or request id.
#[derive(Clone)]
pub struct FastembedImage {
    embedder: Arc<Embedder>,
}

impl FastembedImage {
    /// Load `model`, downloading it when necessary and reporting download
    /// progress on standard output. Return an initialization error if loading
    /// or downloading fails.
    #[cfg(feature = "hf-hub")]
    pub fn load(model: &FastembedImageModel) -> Result<Self, FastembedError> {
        let embedder = Embedder::try_new(
            ImageInitOptions::new(model.to_owned()).with_show_download_progress(true),
        )
        .map_err(|error| FastembedError::Initialization(error.to_string()))?;
        Ok(Self {
            embedder: Arc::new(embedder),
        })
    }

    /// The [`image_embeddings`] wire on this runtime. `model` and `ndims`
    /// describe the loaded model; this runtime always uses the model it loaded.
    pub fn embedding(
        &self,
        model: &FastembedImageModel,
        ndims: Option<usize>,
    ) -> Model<Local<ImageEmbedding>, Self> {
        Model::new(image_embeddings(model, ndims), self.clone())
    }

    /// Load a caller-supplied ONNX model and image preprocessor configuration.
    /// Return an initialization error if either cannot be loaded.
    pub fn from_user_defined(
        user_defined_model: UserDefinedImageEmbeddingModel,
    ) -> Result<Self, FastembedError> {
        let embedder = Embedder::try_new_from_user_defined(
            user_defined_model,
            ImageInitOptionsUserDefined::default(),
        )
        .map_err(|error| FastembedError::Initialization(error.to_string()))?;
        Ok(Self {
            embedder: Arc::new(embedder),
        })
    }
}

impl Transport<Local<ImageEmbedding>> for FastembedImage {
    fn send(&self, images: Vec<Vec<u8>>, _exchange: Exchange) -> Opening<Step<ImageEmbedding>> {
        let embedder = Arc::clone(&self.embedder);
        Opening::new(async move {
            let vectors = if images.is_empty() {
                Ok(Vec::new())
            } else {
                let images: Vec<&[u8]> = images.iter().map(Vec::as_slice).collect();
                embedder
                    .embed_bytes(&images, None)
                    .map_err(|error| ProviderError::Provider(error.to_string()))
            };
            let embedded = vectors.map(|vectors| {
                ImageEmbeddingResponse::new(
                    vectors
                        .into_iter()
                        .map(|vector| Embedding {
                            document: String::new(),
                            vec: vector.into_iter().map(f64::from).collect(),
                        })
                        .collect(),
                )
            });
            Ok(Opened::new(futures::stream::iter(
                [embedded.map(Step::End)],
            )))
        })
    }
}
