use std::sync::Arc;

pub use fastembed::EmbeddingModel as FastembedTextModel;
pub use fastembed::ImageEmbeddingModel as FastembedImageModel;
use fastembed::{
    ImageEmbedding, ImageInitOptions, ImageInitOptionsUserDefined, InitOptions,
    InitOptionsUserDefined, ModelInfo, TextEmbedding, UserDefinedEmbeddingModel,
    UserDefinedImageEmbeddingModel,
};
use rig::{
    embeddings::{self, EmbeddingError, EmbeddingsBuilder},
    Embed,
};

#[derive(Clone)]
pub struct Client;

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}

impl Client {
    /// Create a new Fastembed client.
    pub fn new() -> Self {
        Self
    }

    /// Create a text embedding model with the given name.
    /// Note: default embedding dimension of 0 will be used if model is not known.
    /// If this is the case, it's better to use function `embedding_model_with_ndims`
    ///
    /// # Example
    /// ```
    /// use rig_fastembed::{Client, FastembedModel};
    ///
    /// // Initialize the OpenAI client
    /// let fastembed_client = Client::new("your-open-ai-api-key");
    ///
    /// let embedding_model = fastembed_client.embedding_model(&FastembedModel::AllMiniLML6V2Q);
    /// ```
    pub fn text_embedding_model(&self, model: &FastembedTextModel) -> TextEmbeddingModel {
        TextEmbeddingModel::new(model)
    }

    /// Create an image embedding model with the given name.
    /// Note: default embedding dimension of 0 will be used if model is not known.
    /// If this is the case, it's better to use function `embedding_model_with_ndims`
    ///
    /// # Example
    /// ```
    /// use rig_fastembed::{Client, FastembedModel};
    ///
    /// // Initialize the OpenAI client
    /// let fastembed_client = Client::new("your-open-ai-api-key");
    ///
    /// let embedding_model = fastembed_client.image_embedding_model(&FastembedModel::AllMiniLML6V2Q);
    /// ```
    pub fn image_embedding_model(
        &self,
        model: &fastembed::ImageEmbeddingModel,
    ) -> ImageEmbeddingModel {
        ImageEmbeddingModel::new(model)
    }

    /// Create an embedding builder with the given embedding model.
    ///
    /// # Example
    /// ```
    /// use rig_fastembed::{Client, FastembedModel};
    ///
    /// // Initialize the Fastembed client
    /// let fastembed_client = Client::new();
    ///
    /// let embeddings = fastembed_client.embeddings(FastembedModel::AllMiniLML6V2Q)
    ///     .simple_document("doc0", "Hello, world!")
    ///     .simple_document("doc1", "Goodbye, world!")
    ///     .build()
    ///     .await
    ///     .expect("Failed to embed documents");
    /// ```
    pub fn text_embeddings<D: Embed>(
        &self,
        model: &FastembedTextModel,
    ) -> EmbeddingsBuilder<TextEmbeddingModel, D> {
        EmbeddingsBuilder::new(self.text_embedding_model(model))
    }
}

#[derive(Clone)]
pub struct TextEmbeddingModel {
    embedder: Arc<TextEmbedding>,
    pub model: FastembedTextModel,
    ndims: usize,
}

impl TextEmbeddingModel {
    pub fn new(model: &fastembed::EmbeddingModel) -> Self {
        let embedder = Arc::new(
            TextEmbedding::try_new(
                InitOptions::new(model.to_owned()).with_show_download_progress(true),
            )
            .unwrap(),
        );

        let ndims = TextEmbedding::get_model_info(model).unwrap().dim;

        Self {
            embedder,
            model: model.to_owned(),
            ndims,
        }
    }

    pub fn new_from_user_defined(
        user_defined_model: UserDefinedEmbeddingModel,
        ndims: usize,
        model_info: &ModelInfo<FastembedTextModel>,
    ) -> Self {
        let fastembed_embedding_model = TextEmbedding::try_new_from_user_defined(
            user_defined_model,
            InitOptionsUserDefined::default(),
        )
        .unwrap();

        let embedder = Arc::new(fastembed_embedding_model);

        Self {
            embedder,
            model: model_info.model.to_owned(),
            ndims,
        }
    }
}

impl embeddings::EmbeddingModel for TextEmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents_as_strings: Vec<String> = documents.into_iter().collect();

        let documents_as_vec = self
            .embedder
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

#[derive(Clone)]
pub struct ImageEmbeddingModel {
    embedder: Arc<ImageEmbedding>,
    pub model: FastembedImageModel,
    ndims: usize,
}

impl ImageEmbeddingModel {
    pub fn new(model: &FastembedImageModel) -> Self {
        let embedder = Arc::new(
            ImageEmbedding::try_new(
                ImageInitOptions::new(model.to_owned()).with_show_download_progress(true),
            )
            .unwrap(),
        );

        let ndims = ImageEmbedding::get_model_info(model).dim;

        Self {
            embedder,
            model: model.to_owned(),
            ndims,
        }
    }

    pub fn new_from_user_defined(
        user_defined_model: UserDefinedImageEmbeddingModel,
        ndims: usize,
        model_info: &ModelInfo<FastembedImageModel>,
    ) -> Self {
        let fastembed_embedding_model = ImageEmbedding::try_new_from_user_defined(
            user_defined_model,
            ImageInitOptionsUserDefined::default(),
        )
        .unwrap();

        let embedder = Arc::new(fastembed_embedding_model);

        Self {
            embedder,
            model: model_info.model.to_owned(),
            ndims,
        }
    }
}

impl rig::embeddings::embedding::ImageEmbeddingModel for ImageEmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_images(
        &self,
        documents: impl IntoIterator<Item = Vec<u8>>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let images: Vec<Vec<u8>> = documents.into_iter().collect();
        let images: Vec<&[u8]> = images.iter().map(|x| x.as_slice()).collect();

        let images_as_vec = self.embedder.embed_bytes(&images, None).unwrap();

        let docs = images_as_vec
            .into_iter()
            .map(|embedding| embeddings::Embedding {
                document: String::new(),
                vec: embedding.into_iter().map(|f| f as f64).collect(),
            })
            .collect::<Vec<embeddings::Embedding>>();

        Ok(docs)
    }
}
