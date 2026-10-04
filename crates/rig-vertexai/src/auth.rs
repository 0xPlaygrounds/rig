//! Google Cloud credentials for the Vertex AI client: explicit ones, or
//! Application Default Credentials with optional impersonation.

use std::sync::Arc;

use google_cloud_auth::credentials::{self, Credentials};

use crate::client::VertexAiClientError;

/// Returns explicit credentials or resolves ADC with optional impersonation.
/// ADC construction requires a Tokio runtime that stays alive and driven while
/// credentials are used, because their refresh task runs on that runtime.
pub(crate) fn build_credentials(
    explicit_creds: Option<Credentials>,
) -> Result<Credentials, VertexAiClientError> {
    if let Some(creds) = explicit_creds {
        Ok(creds)
    } else {
        // ADC construction spawns refresh work; refuse before reading any
        // credential source when the host has supplied no runtime context.
        tokio::runtime::Handle::try_current().map_err(|_| VertexAiClientError::RuntimeRequired)?;
        let source_credentials = credentials::Builder::default()
            .build()
            .map_err(|e| VertexAiClientError::SourceCredentials(Arc::new(e)))?;

        if let Ok(service_account) = std::env::var("GOOGLE_CLOUD_SERVICE_ACCOUNT") {
            credentials::impersonated::Builder::from_source_credentials(source_credentials)
                .with_target_principal(service_account)
                .build()
                .map_err(|e| VertexAiClientError::ImpersonatedCredentials(Arc::new(e)))
        } else {
            Ok(source_credentials)
        }
    }
}
