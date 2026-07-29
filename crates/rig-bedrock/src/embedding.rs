use aws_smithy_types::Blob;
use rig_core::embeddings::{self, Embedding, EmbeddingError};
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

/// `amazon.titan-embed-text-v1`
pub const AMAZON_TITAN_EMBED_TEXT_V1: &str = "amazon.titan-embed-text-v1";
/// `amazon.titan-embed-text-v2:0`
pub const AMAZON_TITAN_EMBED_TEXT_V2_0: &str = "amazon.titan-embed-text-v2:0";
/// `amazon.titan-embed-image-v1`
pub const AMAZON_TITAN_EMBED_IMAGE_V1: &str = "amazon.titan-embed-image-v1";
/// `cohere.embed-english-v3`
pub const COHERE_EMBED_ENGLISH_V3: &str = "cohere.embed-english-v3";
/// `cohere.embed-multilingual-v3`
pub const COHERE_EMBED_MULTILINGUAL_V3: &str = "cohere.embed-multilingual-v3";

#[derive(Clone)]
pub struct EmbeddingModel {
    client: Client,
    model: String,
    ndims: Option<usize>,
}

impl EmbeddingModel {
    pub fn new(client: Client, model: impl Into<String>, ndims: Option<usize>) -> Self {
        Self {
            client,
            model: model.into(),
            ndims,
        }
    }

    pub async fn document_to_embeddings(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        invoke_embedding(self.client.get_inner().await, self.model.as_str(), request).await
    }
}

/// Invoke `model` for one embedding document over the AWS SDK.
///
/// Extracted from [`EmbeddingModel::document_to_embeddings`], which is
/// rewired through it (single source of truth); also drives
/// [`crate::functions::embed`].
pub(crate) async fn invoke_embedding(
    client: &aws_sdk_bedrockruntime::Client,
    model: &str,
    request: EmbeddingRequest,
) -> Result<EmbeddingResponse, EmbeddingError> {
    let input_document = serde_json::to_string(&request).map_err(EmbeddingError::JsonError)?;

    let model_response = client
        .invoke_model()
        .model_id(model)
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

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;

    type Client = Client;

    fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self {
        Self::new(client.clone(), model, dims)
    }

    fn ndims(&self) -> usize {
        self.ndims.unwrap_or_default()
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String> + Send,
    ) -> Result<Vec<Embedding>, EmbeddingError> {
        let documents: Vec<_> = documents.into_iter().collect();
        let client = self.client.get_inner().await;
        let response = crate::functions::embed(client, &self.model, self.ndims, documents).await?;
        Ok(response.embeddings)
    }
}
