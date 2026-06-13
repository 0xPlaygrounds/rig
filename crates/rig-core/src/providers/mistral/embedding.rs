use serde::Deserialize;
use serde_json::json;

use crate::{
    embeddings::{self, EmbeddingError},
    http_client::{self, HttpClientExt},
};

use super::client::{ApiResponse, Client, Usage};

// ================================================================
// Mistral Embedding API
// ================================================================
pub const MISTRAL_EMBED: &str = "mistral-embed";

pub const MAX_DOCUMENTS: usize = 1024;

#[derive(Clone)]
pub struct EmbeddingModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
    ndims: usize,
}

impl<T> EmbeddingModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>, ndims: usize) -> Self {
        Self {
            client,
            model: model.into(),
            ndims,
        }
    }

    pub fn with_model(client: Client<T>, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.to_string(),
            ndims,
        }
    }
}

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    type Client = Client<T>;

    const MAX_DOCUMENTS: usize = MAX_DOCUMENTS;

    fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self {
        Self::new(client.clone(), model, dims.unwrap_or_default())
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents = documents.into_iter().collect::<Vec<_>>();

        let body = serde_json::to_vec(&json!({
            "model": self.model,
            "input": documents
        }))?;

        let req = self
            .client
            .post("v1/embeddings")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| EmbeddingError::HttpError(e.into()))?;

        let response = self.client.send(req).await?;

        let status = response.status();
        if status.is_success() {
            let response_body: Vec<u8> = response.into_body().await?;
            let parsed: ApiResponse<EmbeddingResponse> = serde_json::from_slice(&response_body)?;

            match parsed {
                ApiResponse::Ok(response) => {
                    tracing::debug!(target: "rig",
                        "Mistral embedding token usage: {}",
                        response.usage
                    );

                    if response.data.len() != documents.len() {
                        return Err(EmbeddingError::ResponseError(
                            "Response data length does not match input length".into(),
                        ));
                    }

                    Ok(response
                        .data
                        .into_iter()
                        .zip(documents.into_iter())
                        .map(|(embedding, document)| embeddings::Embedding {
                            document,
                            vec: embedding
                                .embedding
                                .into_iter()
                                .filter_map(|n| n.as_f64())
                                .collect(),
                        })
                        .collect())
                }
                ApiResponse::Err(err) => {
                    tracing::warn!(message = %err.message, "provider returned an error response");
                    Err(EmbeddingError::ProviderResponse(
                        crate::provider_response::ProviderResponseError {
                            status: Some(status),
                            body: String::from_utf8_lossy(&response_body).into_owned(),
                        },
                    ))
                }
            }
        } else {
            let text = http_client::text(response).await?;
            Err(EmbeddingError::HttpError(
                http_client::Error::InvalidStatusCodeWithMessage(status, text),
            ))
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    pub usage: Usage,
    pub data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<serde_json::Number>,
    pub index: usize,
}
