use aws_smithy_types::Blob;
use rig_core::embeddings::{self, Embedding};
use rig_core::error::ProviderError;
use serde::{Deserialize, Serialize};

use crate::types::assistant_content::PROVIDER_NAME;
use crate::{client::Client, types::errors::AwsSdkInvokeModelError};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingRequest {
    pub input_text: String,
    pub dimensions: usize,
    pub normalize: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingResponse {
    pub embedding: Vec<f64>,
    pub input_text_token_count: usize,
}

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
    ) -> Result<EmbeddingResponse, ProviderError> {
        let input_document = serde_json::to_string(&request).map_err(ProviderError::Json)?;

        let model_response = self
            .client
            .inner()
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
            .map_err(|e: ProviderError| e)?;

        let response_str = String::from_utf8(response.body.into_inner())
            .map_err(|e| ProviderError::Response(e.to_string()))?;

        let result: EmbeddingResponse =
            serde_json::from_str(&response_str).map_err(ProviderError::Json)?;

        Ok(result)
    }
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    fn max_documents(&self) -> usize {
        1024
    }

    fn ndims(&self) -> usize {
        self.ndims.unwrap_or_default()
    }

    async fn embed_texts_response(
        &self,
        documents: impl IntoIterator<Item = String> + Send,
    ) -> Result<embeddings::EmbeddingResponse, ProviderError> {
        rig_core::telemetry::instrument_modality::<rig_core::operation::Embedding, _>(
            PROVIDER_NAME,
            &self.model,
            async {
                let documents: Vec<String> = documents.into_iter().collect();

                // Sequential requests limit concurrent load against account quotas.
                let mut results = Vec::new();
                let mut raw = Vec::new();
                let mut usage = rig_core::completion::Usage::default();
                let mut first_error = None;
                for doc in documents {
                    let request = EmbeddingRequest {
                        input_text: doc.clone(),
                        dimensions: self.ndims(),
                        normalize: true,
                    };
                    match self.document_to_embeddings(request).await {
                        Ok(response) => {
                            let tokens = response.input_text_token_count as u64;
                            usage += rig_core::completion::Usage {
                                input_tokens: Some(tokens),
                                total_tokens: Some(tokens),
                                ..Default::default()
                            };
                            raw.push(serde_json::to_value(&response)?);
                            results.push(Embedding {
                                document: doc,
                                vec: response.embedding,
                            });
                        }
                        Err(err) => {
                            first_error.get_or_insert(err);
                        }
                    }
                }

                match first_error {
                    None => Ok(embeddings::EmbeddingResponse::new(results, PROVIDER_NAME)
                        .with_usage(usage)
                        .with_raw(serde_json::Value::Array(raw))),
                    Some(err) => Err(ProviderError::Response(err.to_string())),
                }
            },
        )
        .await
    }
}
