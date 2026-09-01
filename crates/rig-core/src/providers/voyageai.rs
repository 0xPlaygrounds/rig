use crate::client::{self, BearerAuth, DebugExt, Provider};
use crate::embeddings;
use crate::embeddings::EmbeddingError;
use crate::http_client::HttpClientExt;
use crate::rerank;
use crate::rerank::RerankError;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::json;

// ================================================================
// Main Voyage AI Client
// ================================================================
const VOYAGEAI_API_BASE_URL: &str = "https://api.voyageai.com/v1";

#[derive(Debug, Default, Clone, Copy)]
pub struct VoyageExt;

#[derive(Debug, Default, Clone, Copy)]
pub struct VoyageBuilder;

type VoyageApiKey = BearerAuth;

impl Provider for VoyageExt {
    type Builder = VoyageBuilder;

    /// There is currently no way to verify a Voyage api key without consuming tokens
    const VERIFY_PATH: &'static str = "";
}

client::impl_capabilities!(
    VoyageExt,
    embeddings = EmbeddingModel<H>,
    rerank = RerankModel<H>,
);

impl DebugExt for VoyageExt {}

client::impl_default_provider_builder!(
    VoyageBuilder => VoyageExt,
    api_key = VoyageApiKey,
    base_url = VOYAGEAI_API_BASE_URL,
);

pub type Client<H> = client::Client<VoyageExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<VoyageBuilder, VoyageApiKey, H>;

client::impl_provider_from_env!(VoyageExt, input = String, api_key_env = "VOYAGE_API_KEY");

impl<T> EmbeddingModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>, ndims: usize) -> Self {
        Self {
            client,
            model: model.into(),
            ndims,
            options: EmbeddingOptions::default(),
        }
    }

    pub fn with_model(client: Client<T>, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.into(),
            ndims,
            options: EmbeddingOptions::default(),
        }
    }

    /// Set optional request parameters for every embedding call made through
    /// this model. Defaults to [`EmbeddingOptions::default()`] (all `None`).
    pub fn with_options(mut self, options: EmbeddingOptions) -> Self {
        self.options = options;
        self
    }
}

// ================================================================
// Voyage AI Embedding API
// ================================================================

/// `voyage-3-large` embedding model (Voyage AI)
pub const VOYAGE_3_LARGE: &str = "voyage-3-large";
/// `voyage-3.5` embedding model (Voyage AI)
pub const VOYAGE_3_5: &str = "voyage-3.5";
/// `voyage-3.5-lite` embedding model (Voyage AI)
pub const VOYAGE_3_5_LITE: &str = "voyage.3-5.lite";
/// `voyage-code-3` embedding model (Voyage AI)
pub const VOYAGE_CODE_3: &str = "voyage-code-3";
/// `voyage-finance-2` embedding model (Voyage AI)
pub const VOYAGE_FINANCE_2: &str = "voyage-finance-2";
/// `voyage-law-2` embedding model (Voyage AI)
pub const VOYAGE_LAW_2: &str = "voyage-law-2";
/// `voyage-code-2` embedding model (Voyage AI)
pub const VOYAGE_CODE_2: &str = "voyage-code-2";

pub fn model_dimensions_from_identifier(model_identifier: &str) -> Option<usize> {
    match model_identifier {
        "voyage-code-2" => Some(1536),
        "voyage-3-large" | "voyage-3.5" | "voyage.3-5.lite" | "voyage-code-3"
        | "voyage-finance-2" | "voyage-law-2" => Some(1024),
        _ => None,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

impl embeddings::NormalizeEmbeddingResponse for EmbeddingResponse {
    fn normalize(
        self,
        provider: &str,
        documents: Vec<String>,
    ) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
        if self.data.len() != documents.len() {
            return Err(EmbeddingError::ResponseError(
                "Response data length does not match input length".into(),
            ));
        }

        let usage = crate::completion::Usage {
            input_tokens: self.usage.total_tokens as u64,
            output_tokens: 0,
            total_tokens: self.usage.total_tokens as u64,
            ..crate::completion::Usage::new()
        };

        let embeddings = self
            .data
            .into_iter()
            .zip(documents)
            .map(|(embedding, document)| embeddings::Embedding {
                document,
                vec: embedding.embedding,
            })
            .collect();

        Ok(embeddings::EmbeddingResponse::new(embeddings, provider)
            .with_model(self.model)
            .with_usage(usage))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Usage {
    pub total_tokens: usize,
}

#[derive(Debug)]
pub struct ApiErrorResponse {
    /// Provider error message; tolerant of `{"message": "..."}`,
    /// `{"error": "..."}`, nested `{"error": {"message": ...}}`, and bodies
    /// carrying both keys. Used for logging only — the raw body is preserved
    /// on the returned error.
    pub(crate) message: String,
}

impl<'de> Deserialize<'de> for ApiErrorResponse {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self {
            message: crate::providers::internal::envelope::error_message(deserializer)?,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: usize,
}

/// Optional request parameters for Voyage AI embedding calls.
///
/// All fields default to `None`, which matches Voyage's own server defaults:
/// no `input_type`, `truncation` enabled, and the model's default output
/// dimension.
///
/// TODO: `output_dtype` (`float` | `int8` | `uint8` | `binary` | `ubinary`) is
/// intentionally not implemented yet. Quantized dtypes return integer vectors
/// instead of floats, so they need a separate response-parsing change before
/// they can be supported here.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EmbeddingOptions {
    /// Prepends a retrieval prompt to the input text. Use `"document"` when
    /// embedding stored chunks and `"query"` when embedding search queries.
    ///
    /// Embeddings produced with and without `input_type` are compatible.
    pub input_type: Option<String>,
    /// Whether to truncate inputs that exceed the model's maximum context
    /// length. Voyage's server default is `true`.
    pub truncation: Option<bool>,
    /// Dimensionality of the returned embeddings. Defaults to the model's
    /// default output dimension when unset.
    pub output_dimension: Option<usize>,
}

#[derive(Clone)]
pub struct EmbeddingModel<T> {
    client: Client<T>,
    pub model: String,
    ndims: usize,
    options: EmbeddingOptions,
}

impl<T> EmbeddingModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    /// Perform the request and return Voyage AI's native response instead of
    /// the normalized [`embeddings::EmbeddingResponse`]. Same request,
    /// transport, parser, and error path as
    /// [`embeddings::EmbeddingModel::embed_texts_response`].
    pub async fn raw_embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        let documents: Vec<String> = documents.into_iter().collect();
        self.raw_embed_texts_slice(&documents).await
    }

    /// Borrow-shaped twin of [`Self::raw_embed_texts`]: the batch is only
    /// serialized into the request body, so callers that keep their documents
    /// (the normalize path) can lend them instead of cloning the batch.
    async fn raw_embed_texts_slice(
        &self,
        documents: &[String],
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        let mut request = json!({
            "model": self.model,
            "input": documents,
        });

        let request_obj = request.as_object_mut().ok_or_else(|| {
            EmbeddingError::ResponseError("embedding request body must be a JSON object".into())
        })?;

        if let Some(input_type) = &self.options.input_type {
            request_obj.insert("input_type".to_owned(), json!(input_type));
        }
        if let Some(truncation) = self.options.truncation {
            request_obj.insert("truncation".to_owned(), json!(truncation));
        }
        if let Some(output_dimension) = self.options.output_dimension {
            request_obj.insert("output_dimension".to_owned(), json!(output_dimension));
        }

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/embeddings")?
            .body(body)
            .map_err(|x| EmbeddingError::HttpError(x.into()))?;

        let response = self.client.send::<_, Bytes>(req).await?;
        let status = response.status();
        let response_body = response.into_body().into_future().await?.to_vec();

        if status.is_success() {
            match serde_json::from_slice::<ApiResponse<EmbeddingResponse>>(&response_body)? {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "VoyageAI embedding token usage: {}",
                        response.usage.total_tokens
                    );
                    Ok(response)
                }
                ApiResponse::Err(err) => {
                    tracing::warn!(message = %err.message, "provider returned an error response");
                    Err(EmbeddingError::from_http_response(
                        status,
                        String::from_utf8_lossy(&response_body),
                    ))
                }
            }
        } else {
            Err(EmbeddingError::from_http_response(
                status,
                String::from_utf8_lossy(&response_body),
            ))
        }
    }
}

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    fn max_documents(&self) -> usize {
        1024
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts_response(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
        crate::telemetry::instrument_modality(
            "voyageai",
            &self.model,
            crate::telemetry::ModalityOperation::Embeddings,
            async {
                use embeddings::NormalizeEmbeddingResponse as _;

                let documents: Vec<String> = documents.into_iter().collect();
                // Voyage AI reports no transport request-id header.
                let response = self.raw_embed_texts_slice(&documents).await?;
                let captured = serde_json::to_value(&response)?;
                Ok(response
                    .normalize("voyageai", documents)?
                    .with_raw(captured))
            },
        )
        .await
    }
}

impl<T> crate::client::ConstructEmbeddingModel<Client<T>> for EmbeddingModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    fn construct(client: &Client<T>, model: String, dims: Option<usize>) -> Self {
        let dims = dims
            .or(model_dimensions_from_identifier(&model))
            .unwrap_or_default();

        Self::new(client.clone(), model, dims)
    }
}

// ================================================================
// Voyage AI Rerank API
// ================================================================

/// `rerank-2.5` reranker model (Voyage AI)
pub const RERANK_2_5: &str = "rerank-2.5";
/// `rerank-2.5-lite` reranker model (Voyage AI)
pub const RERANK_2_5_LITE: &str = "rerank-2.5-lite";
/// `rerank-2` reranker model (Voyage AI)
pub const RERANK_2: &str = "rerank-2";
/// `rerank-2-lite` reranker model (Voyage AI)
pub const RERANK_2_LITE: &str = "rerank-2-lite";
/// `rerank-1` reranker model (Voyage AI)
pub const RERANK_1: &str = "rerank-1";
/// `rerank-lite-1` reranker model (Voyage AI)
pub const RERANK_LITE_1: &str = "rerank-lite-1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankApiResponse {
    pub data: Vec<RerankApiData>,
    pub model: String,
    pub usage: RerankApiUsage,
}

impl rerank::NormalizeRerankResponse for RerankApiResponse {
    fn normalize(self, provider: &str) -> Result<rerank::RerankResponse, RerankError> {
        let usage = crate::completion::Usage {
            input_tokens: self.usage.total_tokens as u64,
            total_tokens: self.usage.total_tokens as u64,
            ..crate::completion::Usage::new()
        };
        let results = self
            .data
            .into_iter()
            .map(|d| rerank::RerankResult {
                index: d.index,
                document: d.document,
                relevance_score: d.relevance_score,
            })
            .collect();
        Ok(rerank::RerankResponse::new(results, provider)
            .with_model(self.model)
            .with_usage(usage))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankApiUsage {
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankApiData {
    pub index: usize,
    pub relevance_score: f64,
    #[serde(default)]
    pub document: Option<String>,
}

#[derive(Clone)]
pub struct RerankModel<T> {
    client: Client<T>,
    pub model: String,
    pub top_k: Option<usize>,
    pub return_documents: bool,
    pub truncation: Option<bool>,
}

impl<T> RerankModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            top_k: None,
            return_documents: false,
            truncation: None,
        }
    }

    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn return_documents(mut self, return_documents: bool) -> Self {
        self.return_documents = return_documents;
        self
    }

    pub fn truncation(mut self, truncation: bool) -> Self {
        self.truncation = Some(truncation);
        self
    }
}

impl<T> RerankModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    /// Perform the request and return Voyage AI's native response instead of
    /// the normalized [`rerank::RerankResponse`]. Same request, transport,
    /// parser, and error path as [`rerank::RerankModel::rerank`].
    pub async fn raw_rerank(
        &self,
        query: &str,
        documents: Vec<String>,
    ) -> Result<RerankApiResponse, RerankError> {
        let mut body = json!({
            "query": query,
            "documents": documents,
            "model": self.model,
        });

        let body_obj = body.as_object_mut().ok_or_else(|| {
            RerankError::ResponseError("rerank request body must be a JSON object".into())
        })?;

        if let Some(top_k) = self.top_k {
            body_obj.insert("top_k".to_owned(), json!(top_k));
        }

        body_obj.insert("return_documents".to_owned(), json!(self.return_documents));

        if let Some(truncation) = self.truncation {
            body_obj.insert("truncation".to_owned(), json!(truncation));
        }

        let body = serde_json::to_vec(&body)?;

        let req = self
            .client
            .post("/rerank")?
            .body(body)
            .map_err(|x| RerankError::HttpError(x.into()))?;

        let response = self.client.send::<_, Bytes>(req).await?;
        let status = response.status();
        let response_body = response.into_body().into_future().await?.to_vec();

        if status.is_success() {
            match serde_json::from_slice::<ApiResponse<RerankApiResponse>>(&response_body)? {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "VoyageAI rerank token usage: {}",
                        response.usage.total_tokens
                    );
                    Ok(response)
                }
                ApiResponse::Err(err) => {
                    tracing::warn!(message = %err.message, "provider returned an error response");
                    Err(RerankError::from_http_response(
                        status,
                        String::from_utf8_lossy(&response_body),
                    ))
                }
            }
        } else {
            Err(RerankError::from_http_response(
                status,
                String::from_utf8_lossy(&response_body),
            ))
        }
    }
}

impl<T> rerank::RerankModel for RerankModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    fn max_documents(&self) -> usize {
        1000
    }

    async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
    ) -> Result<rerank::RerankResponse, RerankError> {
        crate::telemetry::instrument_modality(
            "voyageai",
            &self.model,
            crate::telemetry::ModalityOperation::Rerank,
            async {
                use rerank::NormalizeRerankResponse as _;

                // Voyage AI reports no transport request-id header.
                let response = self.raw_rerank(query, documents).await?;
                let captured = serde_json::to_value(&response)?;
                Ok(response.normalize("voyageai")?.with_raw(captured))
            },
        )
        .await
    }
}

impl<T> crate::client::ConstructRerankModel<Client<T>> for RerankModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    fn construct(client: &Client<T>, model: String) -> Self {
        Self::new(client.clone(), model)
    }
}

#[cfg(test)]
mod tests;
