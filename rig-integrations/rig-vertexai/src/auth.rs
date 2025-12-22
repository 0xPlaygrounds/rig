//! GCP authentication for Vertex AI API requests
//!
//! This module provides utilities for adding GCP authentication to HTTP clients
//! for Vertex AI streaming requests.

use google_cloud_auth::project::Config as AuthConfig;
use google_cloud_auth::token::DefaultTokenSourceProvider;
use http::Extensions;
use reqwest_middleware::{Middleware, Next};
use std::sync::Arc;
use tokio::sync::Mutex;
use token_source::TokenSourceProvider as _;
use tracing::debug;

/// Cached bearer token with optional expiration info
#[derive(Clone, Debug)]
pub struct BearerToken(pub String);

impl BearerToken {
    /// Create a bearer token from a raw token string
    pub fn new(token: String) -> Self {
        Self(token)
    }
}

/// Middleware that dynamically fetches and injects GCP Bearer tokens
pub struct GcpAuthMiddleware {
    token_cache: Arc<Mutex<Option<String>>>,
}

impl GcpAuthMiddleware {
    /// Create a new GCP auth middleware with dynamic token fetching
    pub fn new() -> Self {
        Self {
            token_cache: Arc::new(Mutex::new(None)),
        }
    }

    /// Get a valid GCP token (cached if available)
    async fn get_token(&self) -> reqwest_middleware::Result<String> {
        // Check cache first
        {
            let cached = self.token_cache.lock().await;
            if let Some(token) = cached.as_ref() {
                return Ok(token.clone());
            }
        }

        // Fetch new token
        debug!("Fetching fresh GCP token for Vertex AI");
        let scopes = [
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/aiplatform",
        ];

        let auth_config = AuthConfig::default()
            .with_scopes(&scopes)
            .with_use_id_token(false);
        let tsp = DefaultTokenSourceProvider::new(auth_config)
            .await
            .map_err(|e| reqwest_middleware::Error::Middleware(anyhow::anyhow!("Auth config error: {e}")))?;
        let ts = tsp.token_source();
        let mut token = ts
            .token()
            .await
            .map_err(|e| reqwest_middleware::Error::Middleware(anyhow::anyhow!("Token fetch error: {e}")))?;

        if token.to_lowercase().starts_with("bearer ") {
            token = token[7..].to_string();
        }

        // Cache the token
        *self.token_cache.lock().await = Some(token.clone());

        Ok(token)
    }
}

impl Default for GcpAuthMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Middleware for GcpAuthMiddleware {
    async fn handle(
        &self,
        mut req: reqwest::Request,
        _extensions: &mut Extensions,
        next: Next<'_>,
    ) -> reqwest_middleware::Result<reqwest::Response> {
        // Get token (cached if available)
        let token = self.get_token().await?;
        debug!("Injecting GCP Bearer token for Vertex AI request");

        // Inject Bearer token into Authorization header
        req.headers_mut().insert(
            "Authorization",
            format!("Bearer {token}")
                .parse()
                .map_err(|_| reqwest_middleware::Error::Middleware(anyhow::anyhow!(
                    "Failed to parse authorization header"
                )))?,
        );

        // Continue with next middleware/handler
        next.run(req, _extensions).await
    }
}
