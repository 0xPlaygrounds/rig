use crate::completion::CompletionModel;
use google_cloud_aiplatform_v1 as vertexai;
use google_cloud_auth::credentials;
use google_cloud_auth::credentials::Credentials;
use rig::client::{CompletionClient, Nothing};
use rig::prelude::*;
use std::sync::Arc;
use tokio::sync::OnceCell;

// Env vars and terminology (location, project) chosen to match google genai client
// https://googleapis.github.io/python-genai/genai.html#genai.client.Client

/// Default location for Vertex AI Gemini models.
///
/// The `global` endpoint is recommended for Gemini models as it provides higher availability
/// and reduces resource exhaustion errors. Regional endpoints (e.g., `us-central1`, `europe-west4`)
/// are also supported and can be specified via `ClientBuilder::with_location()`.
/// Regional endpoints may be preferred for data residency requirements or to use regional quotas.
pub const DEFAULT_LOCATION: &str = "global";

/// Helper function to build credentials with optional service account impersonation.
fn build_credentials(explicit_creds: Option<Credentials>) -> Result<Credentials, String> {
    if let Some(creds) = explicit_creds {
        Ok(creds)
    } else {
        // Build default credentials
        let source_credentials = credentials::Builder::default()
            .build()
            .map_err(|e| format!("Failed to build source credentials: {e}"))?;

        // Check for service account impersonation
        if let Ok(service_account) = std::env::var("GOOGLE_CLOUD_SERVICE_ACCOUNT") {
            credentials::impersonated::Builder::from_source_credentials(source_credentials)
                .with_target_principal(service_account)
                .build()
                .map_err(|e| format!("Failed to build impersonated credentials: {e}"))
        } else {
            Ok(source_credentials)
        }
    }
}

#[derive(Clone, Debug)]
pub struct ClientBuilder {
    project: Option<String>,
    location: Option<String>,
    credentials: Option<Credentials>,
}

impl ClientBuilder {
    pub fn new() -> Self {
        Self {
            project: None,
            location: None,
            credentials: None,
        }
    }

    /// Set the Google Cloud project ID explicitly.
    ///
    /// If not set, will fall back to `GOOGLE_CLOUD_PROJECT` environment variable.
    pub fn with_project(mut self, project: &str) -> Self {
        self.project = Some(project.to_string());
        self
    }

    /// Set the Google Cloud location explicitly.
    ///
    /// If not set, will fall back to `GOOGLE_CLOUD_LOCATION` environment variable,
    /// or default to "global" if the env var is also not set.
    pub fn with_location(mut self, location: &str) -> Self {
        self.location = Some(location.to_string());
        self
    }

    /// Set credentials explicitly.
    ///
    /// If not set, will build credentials from Application Default Credentials (ADC),
    /// with optional service account impersonation if `GOOGLE_CLOUD_SERVICE_ACCOUNT` is set.
    pub fn with_credentials(mut self, credentials: Credentials) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Build the client with the configured values, falling back to environment variables where not set.
    ///
    /// The Vertex AI client is built lazily on first use via `get_inner()`.
    pub fn build(self) -> Result<Client, String> {
        let project = self
            .project
            .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok())
            .ok_or_else(|| {
                "Google Cloud project is required. Set it via ClientBuilder::with_project() or GOOGLE_CLOUD_PROJECT environment variable".to_string()
            })?;

        let location = self
            .location
            .or_else(|| std::env::var("GOOGLE_CLOUD_LOCATION").ok())
            .unwrap_or_else(|| DEFAULT_LOCATION.to_string());

        let credentials = build_credentials(self.credentials)?;

        Ok(Client {
            project,
            location,
            credentials,
            vertex_client: Arc::new(OnceCell::new()),
        })
    }
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct Client {
    project: String,
    location: String,
    credentials: Credentials,
    pub(crate) vertex_client: Arc<OnceCell<vertexai::client::PredictionService>>,
}

impl Client {
    /// Create a new client builder that uses environment variables as defaults.
    ///
    /// You can override any values using the builder methods:
    /// - `.with_project()` - override project
    /// - `.with_location()` - override location
    /// - `.with_credentials()` - override credentials
    ///
    /// Example:
    /// ```no_run
    /// # use rig_vertexai::Client;
    /// # fn example() -> Result<(), String> {
    /// // Use all env vars
    /// let client = Client::builder().build()?;
    ///
    /// // Override just the location
    /// let client = Client::builder().with_location("us-central1").build()?;
    ///
    /// // Override project and location
    /// let client = Client::builder()
    ///     .with_project("my-project")
    ///     .with_location("us-central1")
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// Create a new client using environment variables for project, location, and credentials.
    ///
    /// Reads from:
    /// - `GOOGLE_CLOUD_PROJECT` (required)
    /// - `GOOGLE_CLOUD_LOCATION` (optional, defaults to "global")
    /// - `GOOGLE_CLOUD_SERVICE_ACCOUNT` (optional, for service account impersonation)
    ///
    /// Panics if the environment is improperly configured. For error handling, use `Client::builder().build()`.
    pub fn new() -> Self {
        ClientBuilder::new()
            .build()
            .expect("Failed to build Vertex AI client. Make sure GOOGLE_CLOUD_PROJECT is set and credentials are configured (e.g., via 'gcloud auth application-default login')")
    }

    /// Create a client using environment variables for project, location, and credentials.
    ///
    /// This is a convenience method that calls the `ProviderClient::from_env()` trait method.
    /// Reads from:
    /// - `GOOGLE_CLOUD_PROJECT` (required)
    /// - `GOOGLE_CLOUD_LOCATION` (optional, defaults to "global")
    /// - `GOOGLE_CLOUD_SERVICE_ACCOUNT` (optional, for service account impersonation)
    pub fn from_env() -> Self {
        <Self as ProviderClient>::from_env()
    }

    pub fn project(&self) -> &str {
        &self.project
    }

    pub fn location(&self) -> &str {
        &self.location
    }

    pub async fn get_inner(&self) -> &vertexai::client::PredictionService {
        let credentials = self.credentials.clone();
        self.vertex_client
            .get_or_init(|| async {
                let mut builder = vertexai::client::PredictionService::builder();
                builder = builder.with_credentials(credentials);
                builder
                    .build()
                    .await
                    .expect("Failed to build Vertex AI client. Make sure you have Google Cloud credentials configured (e.g., via 'gcloud auth application-default login')")
            })
            .await
    }
}

impl Default for Client {
    fn default() -> Self {
        Client::new()
    }
}

impl ProviderClient for Client {
    type Input = Nothing;

    fn from_env() -> Self
    where
        Self: Sized,
    {
        Client::new()
    }

    fn from_val(_: Self::Input) -> Self
    where
        Self: Sized,
    {
        panic!(
            "Vertex AI uses Application Default Credentials (ADC). Use `Client::from_env()` for default credentials, or `Client::new().with_credentials(...).build()` for custom credentials."
        );
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;

    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel {
        CompletionModel::new(self.clone(), model.into())
    }
}

impl VerifyClient for Client {
    async fn verify(&self) -> Result<(), VerifyError> {
        // No API endpoint to verify credentials - they're validated on first use
        Ok(())
    }
}
