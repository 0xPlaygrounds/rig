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

// The model-id string values are canonically defined in `crate::completion`;
// these aliases keep this module's historical public names.
pub use crate::completion::{
    AMAZON_TITAN_EMBEDDINGS_G1_TEXT as AMAZON_TITAN_EMBED_TEXT_V1,
    AMAZON_TITAN_MULTIMODAL_EMBEDDINGS_G1 as AMAZON_TITAN_EMBED_IMAGE_V1,
    AMAZON_TITAN_TEXT_EMBEDDINGS_V2 as AMAZON_TITAN_EMBED_TEXT_V2_0,
    COHERE_EMBED_ENGLISH as COHERE_EMBED_ENGLISH_V3,
    COHERE_EMBED_MULTILINGUAL as COHERE_EMBED_MULTILINGUAL_V3,
};

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
        let documents: Vec<String> = documents.into_iter().collect();

        // Deliberately sequential: issuing the requests one at a time keeps
        // Bedrock's per-account throttling behavior unchanged.
        let mut results = Vec::new();
        let mut first_error = None;
        for doc in documents {
            let request = EmbeddingRequest {
                input_text: doc.clone(),
                dimensions: self.ndims(),
                normalize: true,
            };
            match self.document_to_embeddings(request).await {
                Ok(embeddings) => results.push(Embedding {
                    document: doc,
                    vec: embeddings.embedding,
                }),
                Err(err) => {
                    first_error.get_or_insert(err);
                }
            }
        }

        match first_error {
            None => Ok(results),
            Some(err) => Err(EmbeddingError::ResponseError(err.to_string())),
        }
    }
}
