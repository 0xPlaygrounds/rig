use super::{Client, client::ApiResponse};

use crate::{
    embeddings::{self, EmbeddingError},
    http_client::HttpClientExt,
    wasm_compat::*,
};

use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
pub struct EmbeddingResponse {
    #[serde(default)]
    pub response_type: Option<String>,
    pub id: String,
    pub embeddings: Vec<Vec<f64>>,
    pub texts: Vec<String>,
    #[serde(default)]
    pub meta: Option<Meta>,
}

#[derive(Deserialize)]
pub struct Meta {
    pub api_version: ApiVersion,
    pub billed_units: BilledUnits,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Deserialize)]
pub struct ApiVersion {
    pub version: String,
    #[serde(default)]
    pub is_deprecated: Option<bool>,
    #[serde(default)]
    pub is_experimental: Option<bool>,
}

#[derive(Deserialize, Debug)]
pub struct BilledUnits {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub search_units: u32,
    #[serde(default)]
    pub classifications: u32,
}

impl std::fmt::Display for BilledUnits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Input tokens: {}\nOutput tokens: {}\nSearch units: {}\nClassifications: {}",
            self.input_tokens, self.output_tokens, self.search_units, self.classifications
        )
    }
}

#[derive(Clone)]
pub struct EmbeddingModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
    pub input_type: String,
    ndims: usize,
}

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
    T: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    const MAX_DOCUMENTS: usize = 96;

    fn ndims(&self) -> usize {
        self.ndims
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents = documents.into_iter().collect::<Vec<_>>();

        let body = json!({
            "model": self.model,
            "texts": documents,
            "input_type": self.input_type
        });

        let body = serde_json::to_vec(&body)?;

        let req = self
            .client
            .post("/v1/embed")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| EmbeddingError::HttpError(e.into()))?;

        let response = self
            .client
            .send::<_, Vec<u8>>(req)
            .await
            .map_err(EmbeddingError::HttpError)?;

        if response.status().is_success() {
            let body: ApiResponse<EmbeddingResponse> =
                serde_json::from_slice(response.into_body().await?.as_slice())?;

            match body {
                ApiResponse::Ok(response) => {
                    match response.meta {
                        Some(meta) => tracing::info!(target: "rig",
                            "Cohere embeddings billed units: {}",
                            meta.billed_units,
                        ),
                        None => tracing::info!(target: "rig",
                            "Cohere embeddings billed units: n/a",
                        ),
                    };

                    if response.embeddings.len() != documents.len() {
                        return Err(EmbeddingError::DocumentError(
                            format!(
                                "Expected {} embeddings, got {}",
                                documents.len(),
                                response.embeddings.len()
                            )
                            .into(),
                        ));
                    }

                    Ok(response
                        .embeddings
                        .into_iter()
                        .zip(documents.into_iter())
                        .map(|(embedding, document)| embeddings::Embedding {
                            document,
                            vec: embedding,
                        })
                        .collect())
                }
                ApiResponse::Err(error) => Err(EmbeddingError::ProviderError(error.message)),
            }
        } else {
            let text = String::from_utf8_lossy(&response.into_body().await?).into();
            Err(EmbeddingError::ProviderError(text))
        }
    }
}

impl<T> EmbeddingModel<T> {
    pub fn new(client: Client<T>, model: &str, input_type: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.to_string(),
            input_type: input_type.to_string(),
            ndims,
        }
    }
}
