use super::{
    Client, Usage,
    client::{ApiErrorResponse, ApiResponse},
};
use crate::embeddings::EmbeddingError;
use crate::http_client::HttpClientExt;
use crate::wasm_compat::WasmCompatSend;
use crate::{embeddings, http_client};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Option<Usage>,
    pub id: Option<String>,
}

impl From<ApiErrorResponse> for EmbeddingError {
    fn from(err: ApiErrorResponse) -> Self {
        EmbeddingError::ProviderError(err.message)
    }
}

impl From<ApiResponse<EmbeddingResponse>> for Result<EmbeddingResponse, EmbeddingError> {
    fn from(value: ApiResponse<EmbeddingResponse>) -> Self {
        match value {
            ApiResponse::Ok(response) => Ok(response),
            ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
        }
    }
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EncodingFormat {
    Float,
    Base64,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: usize,
}

#[derive(Clone)]
pub struct EmbeddingModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
    pub encoding_format: Option<EncodingFormat>,
    pub user: Option<String>,
    ndims: usize,
}

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + WasmCompatSend + 'static,
{
    const MAX_DOCUMENTS: usize = 1024;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>, ndims: Option<usize>) -> Self {
        let model = model.into();
        let dims = ndims.unwrap_or_default();

        Self::new(client.clone(), model, dims)
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents = documents.into_iter().collect::<Vec<_>>();

        let mut body = json!({
            "model": self.model,
            "input": documents,
        });

        if self.ndims > 0 {
            body["dimensions"] = json!(self.ndims);
        }

        if let Some(encoding_format) = &self.encoding_format {
            body["encoding_format"] = json!(encoding_format);
        }

        if let Some(user) = &self.user {
            body["user"] = json!(user);
        }

        let body = serde_json::to_vec(&body)?;

        let req = self
            .client
            .post("/embeddings")?
            .body(body)
            .map_err(|e| EmbeddingError::HttpError(e.into()))?;

        let response = self.client.send(req).await?;

        if response.status().is_success() {
            let body: Vec<u8> = response.into_body().await?;
            let body: ApiResponse<EmbeddingResponse> = serde_json::from_slice(&body)?;

            match body {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "OpenRouter embedding token usage: {:?}",
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
                            vec: embedding.embedding,
                        })
                        .collect())
                }
                ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
            }
        } else {
            let text = http_client::text(response).await?;
            Err(EmbeddingError::ProviderError(text))
        }
    }
}

impl<T> EmbeddingModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>, ndims: usize) -> Self {
        Self {
            client,
            model: model.into(),
            encoding_format: None,
            ndims,
            user: None,
        }
    }

    pub fn with_model(client: Client<T>, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.into(),
            encoding_format: None,
            ndims,
            user: None,
        }
    }

    pub fn with_encoding_format(
        client: Client<T>,
        model: &str,
        ndims: usize,
        encoding_format: EncodingFormat,
    ) -> Self {
        Self {
            client,
            model: model.into(),
            encoding_format: Some(encoding_format),
            ndims,
            user: None,
        }
    }

    pub fn encoding_format(mut self, encoding_format: EncodingFormat) -> Self {
        self.encoding_format = Some(encoding_format);
        self
    }

    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }
}
