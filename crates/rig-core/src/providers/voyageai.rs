use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
    ProviderClient,
};
use crate::embeddings;
use crate::embeddings::EmbeddingError;
use crate::http_client::{self, HttpClientExt};
use crate::rerank;
use crate::rerank::RerankError;
use bytes::Bytes;
use serde::Deserialize;
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

impl<H> Capabilities<H> for VoyageExt {
    type Completion = Nothing;
    type Embeddings = Capable<EmbeddingModel<H>>;
    type Rerank = Capable<RerankModel<H>>;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;

    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for VoyageExt {}

impl ProviderBuilder for VoyageBuilder {
    type Extension<H>
        = VoyageExt
    where
        H: HttpClientExt;
    type ApiKey = VoyageApiKey;

    const BASE_URL: &'static str = VOYAGEAI_API_BASE_URL;

    fn build<H>(
        _builder: &crate::client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(VoyageExt)
    }
}

pub type Client<H = reqwest::Client> = client::Client<VoyageExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<VoyageBuilder, VoyageApiKey, H>;

impl ProviderClient for Client {
    type Input = String;
    type Error = crate::client::ProviderClientError;

    /// Create a new OpenAI client from the `OPENAI_API_KEY` environment variable.
    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("VOYAGE_API_KEY")?;
        Self::new(&api_key).map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(&input).map_err(Into::into)
    }
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
            model: model.into(),
            ndims,
        }
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

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
    pub total_tokens: usize,
}

#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    pub(crate) message: String,
}

impl From<ApiErrorResponse> for EmbeddingError {
    fn from(err: ApiErrorResponse) -> Self {
        EmbeddingError::ProviderError(err.message)
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

impl From<ApiResponse<EmbeddingResponse>> for Result<EmbeddingResponse, EmbeddingError> {
    fn from(value: ApiResponse<EmbeddingResponse>) -> Self {
        match value {
            ApiResponse::Ok(response) => Ok(response),
            ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: usize,
}

#[derive(Clone)]
pub struct EmbeddingModel<T> {
    client: Client<T>,
    pub model: String,
    ndims: usize,
}

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + 'static,
{
    const MAX_DOCUMENTS: usize = 1024;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self {
        let model = model.into();
        let dims = dims
            .or(model_dimensions_from_identifier(&model))
            .unwrap_or_default();

        Self::new(client.clone(), model, dims)
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents: Vec<String> = documents.into_iter().collect();
        let response = self.embed_texts_with_usage(documents).await?;
        Ok(response.embeddings)
    }

    async fn embed_texts_with_usage(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
        let documents: Vec<String> = documents.into_iter().collect();
        let request = json!({
            "model": self.model,
            "input": documents,
        });

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

                    if response.data.len() != documents.len() {
                        return Err(EmbeddingError::ResponseError(
                            "Response data length does not match input length".into(),
                        ));
                    }

                    let usage = crate::completion::Usage {
                        input_tokens: response.usage.total_tokens as u64,
                        output_tokens: 0,
                        total_tokens: response.usage.total_tokens as u64,
                        cached_input_tokens: 0,
                        cache_creation_input_tokens: 0,
                        tool_use_prompt_tokens: 0,
                        reasoning_tokens: 0,
                    };

                    let embeddings = response
                        .data
                        .into_iter()
                        .zip(documents.into_iter())
                        .map(|(embedding, document)| embeddings::Embedding {
                            document,
                            vec: embedding.embedding,
                        })
                        .collect();

                    Ok(embeddings::EmbeddingResponse { embeddings, usage })
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
            Err(EmbeddingError::HttpError(
                crate::http_client::Error::InvalidStatusCodeWithMessage(
                    status,
                    String::from_utf8_lossy(&response_body).to_string(),
                ),
            ))
        }
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

#[derive(Debug, Deserialize)]
pub struct RerankApiResponse {
    pub data: Vec<RerankApiData>,
    pub model: String,
    pub usage: RerankApiUsage,
}

#[derive(Debug, Deserialize)]
pub struct RerankApiUsage {
    pub total_tokens: usize,
}

#[derive(Debug, Deserialize)]
pub struct RerankApiData {
    pub index: usize,
    pub relevance_score: f64,
    #[serde(default)]
    pub document: Option<String>,
}

impl From<ApiErrorResponse> for RerankError {
    fn from(err: ApiErrorResponse) -> Self {
        RerankError::ProviderError(err.message)
    }
}

#[derive(Clone)]
pub struct RerankModel<T = reqwest::Client> {
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

impl<T> rerank::RerankModel for RerankModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + 'static,
{
    const MAX_DOCUMENTS: usize = 1000;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
    ) -> Result<rerank::RerankResponse, RerankError> {
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

                    let usage = crate::completion::Usage {
                        input_tokens: response.usage.total_tokens as u64,
                        output_tokens: 0,
                        total_tokens: response.usage.total_tokens as u64,
                        cached_input_tokens: 0,
                        cache_creation_input_tokens: 0,
                        reasoning_tokens: 0,
                        tool_use_prompt_tokens: 0,
                    };

                    let results = response
                        .data
                        .into_iter()
                        .map(|d| rerank::RerankResult {
                            index: d.index,
                            document: d.document,
                            relevance_score: d.relevance_score,
                        })
                        .collect();

                    Ok(rerank::RerankResponse {
                        results,
                        model: response.model,
                        usage,
                    })
                }
                ApiResponse::Err(err) => Err(RerankError::ProviderError(err.message)),
            }
        } else {
            Err(RerankError::ProviderError(
                String::from_utf8_lossy(&response_body).to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::voyageai::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::voyageai::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }
}
