use crate::client::{AsEmbeddings, ProviderClient};
use crate::Embed;
use crate::embeddings::{EmbeddingModel, EmbeddingsBuilder};
use crate::embeddings::embedding::EmbeddingModelDyn;

pub trait EmbeddingsClient: ProviderClient {
    type EmbeddingModel: EmbeddingModel;
    fn embedding_model(&self, model: &str) -> Self::EmbeddingModel;
    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> Self::EmbeddingModel;
    fn embeddings<D: Embed>(&self, model: &str) -> EmbeddingsBuilder<Self::EmbeddingModel, D> {
        EmbeddingsBuilder::new(self.embedding_model(model))
    }

    fn embeddings_with_ndims<D: Embed>(
        &self,
        model: &str,
        ndims: usize,
    ) -> EmbeddingsBuilder<Self::EmbeddingModel, D> {
        EmbeddingsBuilder::new(self.embedding_model_with_ndims(model, ndims))
    }
}

pub trait EmbeddingsClientDyn: ProviderClient {
    fn embedding_model<'a>(&'a self, model: &'a str) -> Box<dyn EmbeddingModelDyn + 'a>;
    fn embedding_model_with_ndims<'a>(
        &'a self,
        model: &'a str,
        ndims: usize,
    ) -> Box<dyn EmbeddingModelDyn + 'a>;
    
    
}

impl<T: EmbeddingsClient> EmbeddingsClientDyn for T {
    fn embedding_model<'a>(&'a self, model: &'a str) -> Box<dyn EmbeddingModelDyn + 'a> {
        Box::new(self.embedding_model(model))
    }

    fn embedding_model_with_ndims<'a>(
        &'a self,
        model: &'a str,
        ndims: usize,
    ) -> Box<dyn EmbeddingModelDyn + 'a> {
        Box::new(self.embedding_model_with_ndims(model, ndims))
    }
}

impl<T: EmbeddingsClientDyn> AsEmbeddings for T {
    fn as_embeddings(&self) -> Option<Box<&dyn EmbeddingsClientDyn>> {
        Some(Box::new(self))
    }
}
