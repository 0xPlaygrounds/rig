use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
    ProviderClient,
};
use crate::embeddings;
use crate::embeddings::EmbeddingError;
use crate::http_client::{self, HttpClientExt};
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

    fn build<H>(
        _: &crate::client::ClientBuilder<
            Self::Builder,
            <Self::Builder as crate::client::ProviderBuilder>::ApiKey,
            H,
        >,
    ) -> http_client::Result<Self> {
        Ok(Self)
    }
}

impl<H> Capabilities<H> for VoyageExt {
    type Completion = Nothing;
    type Embeddings = Capable<EmbeddingModel<H>>;
    type Transcription = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;

    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for VoyageExt {}

impl ProviderBuilder for VoyageBuilder {
    type Output = VoyageExt;
    type ApiKey = VoyageApiKey;

    const BASE_URL: &'static str = VOYAGEAI_API_BASE_URL;
}

pub type Client<H = reqwest::Client> = client::Client<VoyageExt, H>;
pub type ClientBuilder<H = reqwest::Client> = client::ClientBuilder<VoyageBuilder, VoyageApiKey, H>;

impl ProviderClient for Client {
    type Input = String;

    /// Create a new OpenAI client from the `OPENAI_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("VOYAGE_API_KEY").expect("VOYAGE_API_KEY not set");
        Self::new(&api_key).unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(&input).unwrap()
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
    pub prompt_tokens: usize,
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
        let documents = documents.into_iter().collect::<Vec<_>>();
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
            Err(EmbeddingError::ProviderError(
                String::from_utf8_lossy(&response_body).to_string(),
            ))
        }
    }
}
