//! HTTP client for Cloudflare Vectorize API.

use crate::error::VectorizeError;
use crate::types::{ApiResponse, QueryRequest, QueryResult};
use reqwest::Client;
use tracing::instrument;

/// HTTP client wrapper for Vectorize API operations.
#[derive(Debug, Clone)]
pub struct VectorizeClient {
    http_client: Client,
    account_id: String,
    index_name: String,
    api_token: String,
    base_url: String,
}

impl VectorizeClient {
    /// Creates a new Vectorize client.
    ///
    /// # Arguments
    /// * `account_id` - Cloudflare account ID
    /// * `index_name` - Name of the Vectorize index
    /// * `api_token` - Cloudflare API token with Vectorize permissions
    pub fn new(account_id: impl Into<String>, index_name: impl Into<String>, api_token: impl Into<String>) -> Self {
        Self {
            http_client: Client::new(),
            account_id: account_id.into(),
            index_name: index_name.into(),
            api_token: api_token.into(),
            base_url: "https://api.cloudflare.com/client/v4".to_string(),
        }
    }

    /// Creates a new Vectorize client with a custom base URL.
    /// Useful for testing or enterprise deployments.
    pub fn with_base_url(
        account_id: impl Into<String>,
        index_name: impl Into<String>,
        api_token: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            http_client: Client::new(),
            account_id: account_id.into(),
            index_name: index_name.into(),
            api_token: api_token.into(),
            base_url: base_url.into(),
        }
    }

    /// Returns the base URL for the index endpoints.
    fn index_url(&self) -> String {
        format!(
            "{}/accounts/{}/vectorize/v2/indexes/{}",
            self.base_url, self.account_id, self.index_name
        )
    }

    /// Performs a vector similarity query.
    #[instrument(skip(self, request), fields(index = %self.index_name, top_k = request.top_k))]
    pub async fn query(&self, request: QueryRequest) -> Result<QueryResult, VectorizeError> {
        let url = format!("{}/query", self.index_url());

        tracing::debug!("Sending query request to Vectorize");

        let response = self
            .http_client
            .post(&url)
            .bearer_auth(&self.api_token)
            .json(&request)
            .send()
            .await?;

        // Get raw response text for debugging
        let response_text = response.text().await?;
        tracing::debug!("Raw Vectorize response: {}", response_text);

        let api_response: ApiResponse<QueryResult> = serde_json::from_str(&response_text)?;

        if !api_response.success {
            let error = api_response
                .errors
                .first()
                .map(|e| VectorizeError::ApiError {
                    code: e.code,
                    message: e.message.clone(),
                })
                .unwrap_or_else(|| VectorizeError::ApiError {
                    code: 0,
                    message: "Unknown error".to_string(),
                });
            return Err(error);
        }

        api_response.result.ok_or_else(|| VectorizeError::ApiError {
            code: 0,
            message: "No result in successful response".to_string(),
        })
    }
}
