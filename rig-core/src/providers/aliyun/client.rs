use crate::{
    embeddings::{self},
    Embed,
};
use serde::Deserialize;

use super::embedding::EmbeddingModel;

// ================================================================
// Aliyun Gemini Client
// ================================================================
const ALIYUN_API_BASE_URL: &str = "https://dashscope.aliyuncs.com";

#[derive(Clone)]
pub struct Client {
    base_url: String,
    api_key: String,
    http_client: reqwest::Client,
}

impl Client {
    /// Create a new Aliyun client with the given API key.
    ///
    /// # Example
    /// ```
    /// use rig::providers::aliyun::Client;
    ///
    /// // Initialize the Aliyun client
    /// let aliyun = Client::new("your-dashscope-api-key");
    /// ```
    pub fn new(api_key: &str) -> Self {
        Self::from_url(api_key, ALIYUN_API_BASE_URL)
    }

    /// Create a new Aliyun client with the given API key and base URL.
    ///
    /// # Example
    /// ```
    /// use rig::providers::aliyun::Client;
    ///
    /// // Initialize the Aliyun client with a custom base URL
    /// let aliyun = Client::from_url("your-dashscope-api-key", "https://custom-dashscope-url.com");
    /// ```
    pub fn from_url(api_key: &str, base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            http_client: reqwest::Client::builder()
                .default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        reqwest::header::CONTENT_TYPE,
                        "application/json".parse().unwrap(),
                    );
                    headers
                })
                .build()
                .expect("Aliyun reqwest client should build"),
        }
    }

    /// Create a new Aliyun client from the `DASHSCOPE_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    ///
    /// # Example
    /// ```
    /// use rig::providers::aliyun::Client;
    ///
    /// // Initialize the Aliyun client from environment variable
    /// let aliyun = Client::from_env();
    /// ```
    /// # Panics
    /// This function will panic if the `DASHSCOPE_API_KEY` environment variable is not set.
    pub fn from_env() -> Self {
        let api_key = std::env::var("DASHSCOPE_API_KEY").expect("DASHSCOPE_API_KEY not set");
        Self::new(&api_key)
    }

    /// Create a POST request to the specified API endpoint path.
    /// The Authorization header with the API key will be automatically added.
    ///
    /// # Arguments
    /// * `path` - The API endpoint path to append to the base URL
    ///
    /// # Returns
    /// A reqwest::RequestBuilder instance that can be further customized before sending
    pub fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");

        tracing::debug!("POST {}/{}", self.base_url, path);
        self.http_client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
    }

    /// Create an embedding model with the given name.
    /// Note: default embedding dimension of 0 will be used if model is not known.
    /// If this is the case, it's better to use function `embedding_model_with_ndims`
    ///
    /// # Example
    /// ```
    /// use rig::providers::aliyun::{Client, self};
    ///
    /// // Initialize the Aliyun client
    /// let aliyun = Client::new("your-dashscope-api-key");
    ///
    /// let embedding_model = aliyun.embedding_model("your-model-name");
    /// ```
    pub fn embedding_model(&self, model: &str) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, None)
    }

    /// Create an embedding model with the given name and the number of dimensions in the embedding generated by the model.
    ///
    /// # Example
    /// ```
    /// use rig::providers::aliyun::{Client, self};
    ///
    /// // Initialize the Aliyun client
    /// let aliyun = Client::new("your-dashscope-api-key");
    ///
    /// let embedding_model = aliyun.embedding_model_with_ndims("model-unknown-to-rig", 1024);
    /// ```
    pub fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, Some(ndims))
    }

    /// Create an embedding builder with the given embedding model.
    ///
    /// # Example
    /// ```
    /// use rig::providers::aliyun::{Client, self};
    ///
    /// // Initialize the Aliyun client
    /// let aliyun = Client::new("your-dashscope-api-key");
    ///
    /// let embeddings = aliyun.embeddings("your-model-name")
    ///     .simple_document("doc0", "Hello, world!")
    ///     .simple_document("doc1", "Goodbye, world!")
    ///     .build()
    ///     .await
    ///     .expect("Failed to embed documents");
    /// ```
    pub fn embeddings<D: Embed>(
        &self,
        model: &str,
    ) -> embeddings::EmbeddingsBuilder<EmbeddingModel, D> {
        embeddings::EmbeddingsBuilder::new(self.embedding_model(model))
    }
}

#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    pub message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}
