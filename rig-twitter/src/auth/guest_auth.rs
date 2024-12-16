use super::TwitterAuth;
use crate::error::{Result, TwitterError};
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use cookie::CookieJar;
use reqwest::header::{HeaderMap, HeaderValue};
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct TwitterGuestAuth {
    bearer_token: String,
    guest_token: Option<String>,
    cookie_jar: Arc<Mutex<CookieJar>>,
    created_at: Option<DateTime<Utc>>,
    client: reqwest::Client,
}

impl TwitterGuestAuth {
    pub fn new(bearer_token: String) -> Self {
        Self {
            bearer_token,
            guest_token: None,
            cookie_jar: Arc::new(Mutex::new(CookieJar::new())),
            created_at: None,
            client: reqwest::Client::new(),
        }
    }

    async fn update_guest_token(&mut self) -> Result<()> {
        let url = "https://api.twitter.com/1.1/guest/activate.json";
        
        let mut headers = HeaderMap::new();
        headers.insert(
            "Authorization",
            HeaderValue::from_str(&format!("Bearer {}", self.bearer_token))
                .map_err(|e| TwitterError::Auth(e.to_string()))?,
        );

        let response = self.client
            .post(url)
            .headers(headers)
            .send()
            .await
            .map_err(TwitterError::Network)?;

        if !response.status().is_success() {
            return Err(TwitterError::Auth("Failed to get guest token".into()));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(TwitterError::Network)?;

        let guest_token = json["guest_token"]
            .as_str()
            .ok_or_else(|| TwitterError::Auth("Guest token not found in response".into()))?;

        self.guest_token = Some(guest_token.to_string());
        self.created_at = Some(Utc::now());

        Ok(())
    }

    fn should_update_token(&self) -> bool {
        match (self.guest_token.as_ref(), self.created_at) {
            (Some(_), Some(created_at)) => {
                Utc::now() - created_at > Duration::hours(3)
            }
            _ => true,
        }
    }
}