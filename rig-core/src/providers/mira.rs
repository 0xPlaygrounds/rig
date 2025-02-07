//! Mira API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::mira;
//!
//! let client = mira::Client::new("YOUR_API_KEY");
//!
//! ```
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::string::FromUtf8Error;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MiraError {
    #[error("Invalid API key")]
    InvalidApiKey,
    #[error("API error: {0}")]
    ApiError(u16),
    #[error("Request error: {0}")]
    RequestError(#[from] reqwest::Error),
    #[error("UTF-8 error: {0}")]
    Utf8Error(#[from] FromUtf8Error),
}

#[derive(Debug, Serialize)]
pub struct AiRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    pub message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    id: String,
}

/// Client for interacting with the Mira API
pub struct Client {
    base_url: String,
    client: reqwest::Client,
    headers: HeaderMap,
}

impl Client {
    /// Create a new Mira client with the given API key
    pub fn new(api_key: impl AsRef<str>) -> Result<Self, MiraError> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key.as_ref()))
                .map_err(|_| MiraError::InvalidApiKey)?,
        );

        Ok(Self {
            base_url: "https://apis.mira.network".to_string(),
            client: reqwest::Client::new(),
            headers,
        })
    }

    /// Create a new Mira client with a custom base URL and API key
    pub fn new_with_base_url(
        api_key: impl AsRef<str>,
        base_url: impl Into<String>,
    ) -> Result<Self, MiraError> {
        let mut client = Self::new(api_key)?;
        client.base_url = base_url.into();
        Ok(client)
    }

    /// Generate a chat completion
    pub async fn generate(&self, request: AiRequest) -> Result<ChatResponse, MiraError> {
        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .headers(self.headers.clone())
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(MiraError::ApiError(response.status().as_u16()));
        }

        Ok(response.json().await?)
    }

    /// List available models
    pub async fn list_models(&self) -> Result<Vec<String>, MiraError> {
        let response = self
            .client
            .get(format!("{}/v1/models", self.base_url))
            .headers(self.headers.clone())
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(MiraError::ApiError(response.status().as_u16()));
        }

        let models: ModelsResponse = response.json().await?;
        Ok(models.data.into_iter().map(|model| model.id).collect())
    }

    /// Get user credits information
    pub async fn get_user_credits(&self) -> Result<Value, MiraError> {
        let response = self
            .client
            .get(format!("{}/user-credits", self.base_url))
            .headers(self.headers.clone())
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(MiraError::ApiError(response.status().as_u16()));
        }

        Ok(response.json().await?)
    }

    /// Get credits history
    pub async fn get_credits_history(&self) -> Result<Vec<Value>, MiraError> {
        let response = self
            .client
            .get(format!("{}/user-credits-history", self.base_url))
            .headers(self.headers.clone())
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(MiraError::ApiError(response.status().as_u16()));
        }

        Ok(response.json().await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generate() {
        let client = Client::new("mira-api-key").unwrap();

        // First get available models to ensure we use a valid one
        let _models = client.list_models().await.unwrap();
        // println!("Available models: {:?}", models);

        let request = AiRequest {
            model: "deepseek-r1".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello, What can you do?".to_string(),
            }],
            temperature: Some(0.7),
            max_tokens: Some(100),
            stream: None,
        };

        let response = client.generate(request).await.unwrap();
        println!("Response: {:?}", response);
        assert!(!response.choices.is_empty());
    }

    #[tokio::test]
    async fn test_list_models() {
        let client = Client::new("mira-api-key").unwrap();
        let models = client.list_models().await.unwrap();
        println!("Models: {:?}", models);
        assert!(!models.is_empty());
        assert!(models.iter().any(|model| model == "gpt-4o"
            || model == "deepseek-r1"
            || model == "claude-3.5-sonnet"));
    }

    #[tokio::test]
    async fn test_get_user_credits() {
        let client = Client::new("mira-api-key").unwrap();
        let credits = client.get_user_credits().await.unwrap();
        println!("Credits: {:?}", credits);
    }

    #[tokio::test]
    async fn test_get_credits_history() {
        let client = Client::new("mira-api-key").unwrap();
        let history = client.get_credits_history().await.unwrap();
        println!("History: {:?}", history);
    }
}
