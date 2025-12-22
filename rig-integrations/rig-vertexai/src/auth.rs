//! GCP authentication for Vertex AI API requests
//!
//! This module provides utilities for adding GCP authentication to HTTP clients
//! for Vertex AI streaming requests.

use http::Extensions;
use reqwest_middleware::{Middleware, Next};
use std::sync::Arc;
use tokio::sync::Mutex;
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

/// Middleware that injects pre-cached GCP Bearer tokens into outgoing HTTP requests
pub struct GcpAuthMiddleware {
    token: Arc<Mutex<BearerToken>>,
}

impl GcpAuthMiddleware {
    /// Create a new GCP auth middleware with a pre-fetched token
    pub fn new(token: BearerToken) -> Self {
        Self {
            token: Arc::new(Mutex::new(token)),
        }
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
        // Get cached token
        let token = self.token.lock().await;
        debug!("Injecting GCP Bearer token for Vertex AI request");

        // Inject Bearer token into Authorization header
        req.headers_mut().insert(
            "Authorization",
            format!("Bearer {}", token.0)
                .parse()
                .map_err(|_| reqwest_middleware::Error::Middleware(anyhow::anyhow!(
                    "Failed to parse authorization header"
                )))?,
        );

        // Continue with next middleware/handler
        next.run(req, _extensions).await
    }
}
