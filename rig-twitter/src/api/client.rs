use crate::auth::user_auth::TwitterAuth;
use crate::error::{Result, TwitterError};
use crate::models::Tweet;
use reqwest::{Client, Method};
use serde::de::DeserializeOwned;
use std::time::Duration;

pub struct TwitterApiClient {
    pub client: Client,
    pub auth: Box<dyn TwitterAuth + Send + Sync>,
}

impl TwitterApiClient {
    pub fn new(auth: Box<dyn TwitterAuth + Send + Sync>) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .cookie_store(true)
            .build()?;

        Ok(Self { client, auth })
    }

    pub async fn send_tweet(&self, text: &str, media_ids: Option<Vec<String>>) -> Result<Tweet> {
        let mut params = serde_json::json!({
            "text": text,
        });

        if let Some(ids) = media_ids {
            params["media"] = serde_json::json!({ "media_ids": ids });
        }

        let endpoint = "https://api.twitter.com/2/tweets";
        self.post(endpoint, Some(params)).await
    }

    pub async fn get_tweet(&self, tweet_id: &str) -> Result<Tweet> {
        let endpoint = format!("https://api.twitter.com/2/tweets/{}", tweet_id);
        self.get(&endpoint).await
    }

    pub async fn get_user_tweets(&self, user_id: &str, limit: usize) -> Result<Vec<Tweet>> {
        let endpoint = format!("https://api.twitter.com/2/users/{}/tweets", user_id);
        let params = serde_json::json!({
            "max_results": limit,
            "tweet.fields": "created_at,author_id,conversation_id,public_metrics"
        });
        self.get_with_params(&endpoint, Some(params)).await
    }

    pub async fn get<T: DeserializeOwned>(&self, endpoint: &str) -> Result<T> {
        self.request(Method::GET, endpoint, None).await
    }

    pub async fn get_with_params<T: DeserializeOwned>(
        &self,
        endpoint: &str,
        params: Option<serde_json::Value>,
    ) -> Result<T> {
        self.request(Method::GET, endpoint, params).await
    }

    pub async fn post<T: DeserializeOwned>(
        &self,
        endpoint: &str,
        params: Option<serde_json::Value>,
    ) -> Result<T> {
        self.request(Method::POST, endpoint, params).await
    }

    pub async fn request<T: DeserializeOwned>(
        &self,
        method: Method,
        endpoint: &str,
        params: Option<serde_json::Value>,
    ) -> Result<T> {
        let mut headers = reqwest::header::HeaderMap::new();
        self.auth.install_headers(&mut headers).await?;

        let mut request = self.client.request(method, endpoint);
        request = request.headers(headers);

        if let Some(params) = params {
            request = request.json(&params);
        }

        let response = request.send().await?;

        if response.status().is_success() {
            Ok(response.json().await?)
        } else {
            Err(TwitterError::Api(format!(
                "Request failed with status: {}",
                response.status()
            )))
        }
    }
}
