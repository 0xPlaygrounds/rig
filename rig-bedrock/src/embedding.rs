use aws_smithy_types::Blob;
use rig::{
    embeddings::{self, Embedding, EmbeddingError},
    models,
};
use serde::{Deserialize, Serialize};

use crate::{client::Client, types::errors::AwsSdkInvokeModelError};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingRequest {
    pub input_text: String,
    pub dimensions: usize,
    pub normalize: bool,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingResponse {
    pub embedding: Vec<f64>,
    pub input_text_token_count: usize,
}

models! {
    pub enum EmbeddingModels {
        /// `amazon.titan-embed-text-v1`
        AmazonTitanEmbedTextV1 => "amazon.titan-embed-text-v1",
        /// `amazon.titan-embed-text-v2:0`
        AmazonTitanEmbedTextV2 => "amazon.titan-embed-text-v2:0",
        /// `amazon.titan-embed-image-v1`
        AmazonTitanEmbedImageV1 => "amazon.titan-embed-image-v1",
        /// `cohere.embed-english-v3`
        CohereEmbedEnglishV3 => "cohere.embed-english-v3",
        /// `cohere.embed-multilingual-v3`
        CohereEmbedMultilingualV3 => "cohere.embed-multilingual-v3",
    }
}
pub use EmbeddingModels::*;

#[derive(Clone)]
pub struct EmbeddingModel {
    client: Client,
    model: String,
    ndims: Option<usize>,
}

impl EmbeddingModel {
    pub fn new(client: Client, model: &str, ndims: Option<usize>) -> Self {
        Self {
            client,
            model: model.to_string(),
            ndims,
        }
    }

    pub fn with_model(client: Client, model: &str, ndims: Option<usize>) -> Self {
        Self {
            client,
            model: model.to_string(),
            ndims,
        }
    }

    pub async fn document_to_embeddings(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        let input_document = serde_json::to_string(&request).map_err(EmbeddingError::JsonError)?;

        let model_response = self
            .client
            .get_inner()
            .await
            .invoke_model()
            .model_id(self.model.as_str())
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(input_document))
            .send()
            .await;

        let response = model_response
            .map_err(|sdk_error| AwsSdkInvokeModelError(sdk_error).into())
            .map_err(|e: EmbeddingError| e)?;

        let response_str = String::from_utf8(response.body.into_inner())
            .map_err(|e| EmbeddingError::ResponseError(e.to_string()))?;

        let result: EmbeddingResponse =
            serde_json::from_str(&response_str).map_err(EmbeddingError::JsonError)?;

        Ok(result)
    }
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;

    type Client = Client;
    type Models = String;

    fn make(client: &Self::Client, model: Self::Models, dims: Option<usize>) -> Self {
        Self::new(client.clone(), model.as_str(), dims)
    }

    fn make_custom(client: &Self::Client, model: &str, dims: Option<usize>) -> Self {
        Self::with_model(client.clone(), model, dims)
    }

    fn ndims(&self) -> usize {
        self.ndims.unwrap_or_default()
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String> + Send,
    ) -> Result<Vec<Embedding>, EmbeddingError> {
        let documents: Vec<_> = documents.into_iter().collect();

        let mut results = Vec::new();
        let mut errors = Vec::new();

        let mut iterator = documents.into_iter();
        while let Some(embedding) = iterator.next().map(|doc| async move {
            let request = EmbeddingRequest {
                input_text: doc.to_owned(),
                dimensions: self.ndims(),
                normalize: true,
            };
            self.document_to_embeddings(request)
                .await
                .map(|embeddings| Embedding {
                    document: doc.to_owned(),
                    vec: embeddings.embedding,
                })
        }) {
            match embedding.await {
                Ok(embedding) => results.push(embedding),
                Err(err) => errors.push(err),
            }
        }

        match errors.as_slice() {
            [] => Ok(results),
            [err, ..] => Err(EmbeddingError::ResponseError(err.to_string())),
        }
    }
}
