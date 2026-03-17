//! Cloudflare Vectorize HTTP client.
//!
//! This module contains the HTTP client for interacting with the Cloudflare Vectorize API.
//! It is designed to be potentially extracted into a standalone crate in the future.

mod error;
mod filter;
mod types;

pub use error::VectorizeError;
pub use filter::VectorizeFilter;
pub use types::{
    DeleteByIdsRequest, DeleteResult, ListVectorsResult, QueryRequest, QueryResult, ReturnMetadata,
    UpsertRequest, UpsertResult, VectorIdEntry, VectorInput, VectorMatch,
};

use reqwest::Client;
use tracing::instrument;
use types::ApiResponse;

/// Base URL for the Cloudflare API.
const CLOUDFLARE_API_BASE_URL: &str = "https://api.cloudflare.com/client/v4";

/// HTTP client wrapper for Vectorize API operations.
#[derive(Debug, Clone)]
pub struct VectorizeClient {
    http_client: Client,
    account_id: String,
    index_name: String,
    api_token: String,
}

impl VectorizeClient {
    /// Creates a new Vectorize client.
    ///
    /// # Arguments
    /// * `account_id` - Cloudflare account ID
    /// * `index_name` - Name of the Vectorize index
    /// * `api_token` - Cloudflare API token with Vectorize permissions
    pub fn new(
        account_id: impl Into<String>,
        index_name: impl Into<String>,
        api_token: impl Into<String>,
    ) -> Self {
        Self {
            http_client: Client::new(),
            account_id: account_id.into(),
            index_name: index_name.into(),
            api_token: api_token.into(),
        }
    }

    /// Returns the base URL for the index endpoints.
    fn index_url(&self) -> String {
        format!(
            "{}/accounts/{}/vectorize/v2/indexes/{}",
            CLOUDFLARE_API_BASE_URL, self.account_id, self.index_name
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

    /// Upserts vectors (inserts or updates if ID already exists).
    ///
    /// This is the preferred method for inserting documents as it's idempotent.
    /// Up to 5000 vectors can be upserted per request via the HTTP API.
    #[instrument(skip(self, request), fields(index = %self.index_name, count = request.vectors.len()))]
    pub async fn upsert(&self, request: UpsertRequest) -> Result<UpsertResult, VectorizeError> {
        let url = format!("{}/upsert", self.index_url());

        tracing::debug!(
            "Sending upsert request to Vectorize with {} vectors",
            request.vectors.len()
        );

        let response = self
            .http_client
            .post(&url)
            .bearer_auth(&self.api_token)
            .json(&request)
            .send()
            .await?;

        let response_text = response.text().await?;
        tracing::debug!("Raw Vectorize upsert response: {}", response_text);

        let api_response: ApiResponse<UpsertResult> = serde_json::from_str(&response_text)?;

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
            message: "No result in successful upsert response".to_string(),
        })
    }

    /// Deletes vectors by their IDs.
    ///
    /// Up to 1000 vector IDs can be deleted per request.
    #[instrument(skip(self, ids), fields(index = %self.index_name, count = ids.len()))]
    pub async fn delete_by_ids(&self, ids: Vec<String>) -> Result<DeleteResult, VectorizeError> {
        let url = format!("{}/delete_by_ids", self.index_url());

        let request = DeleteByIdsRequest { ids };

        let response = self
            .http_client
            .post(&url)
            .bearer_auth(&self.api_token)
            .json(&request)
            .send()
            .await?;

        let response_text = response.text().await?;
        tracing::debug!("Raw Vectorize delete response: {}", response_text);

        let api_response: ApiResponse<DeleteResult> = serde_json::from_str(&response_text)?;

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
            message: "No result in successful delete response".to_string(),
        })
    }

    /// Lists vector IDs in the index (paginated).
    ///
    /// Returns up to `limit` vector IDs (max 1000, default 100).
    /// Use the `next_cursor` from the response to fetch the next page.
    #[instrument(skip(self), fields(index = %self.index_name))]
    pub async fn list_vectors(
        &self,
        limit: Option<u32>,
        cursor: Option<&str>,
    ) -> Result<ListVectorsResult, VectorizeError> {
        let mut url = format!("{}/list", self.index_url());

        let mut query_params = Vec::new();
        if let Some(limit) = limit {
            query_params.push(format!("count={}", limit));
        }
        if let Some(cursor) = cursor {
            query_params.push(format!("cursor={}", cursor));
        }
        if !query_params.is_empty() {
            url = format!("{}?{}", url, query_params.join("&"));
        }

        let response = self
            .http_client
            .get(&url)
            .bearer_auth(&self.api_token)
            .send()
            .await?;

        let response_text = response.text().await?;
        tracing::debug!("Raw Vectorize list response: {}", response_text);

        let api_response: ApiResponse<ListVectorsResult> = serde_json::from_str(&response_text)?;

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
            message: "No result in successful list response".to_string(),
        })
    }
}
