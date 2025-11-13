use crate::completion::CompletionModel;
use google_cloud_aiplatform_v1 as vertexai;
use google_cloud_auth::credentials::Credentials;
use rig::client::{CompletionClient, ProviderValue};
use rig::impl_conversion_traits;
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

    pub fn with_project(mut self, project: &str) -> Self {
        self.project = Some(project.to_string());
        self
    }

    pub fn with_location(mut self, location: &str) -> Self {
        self.location = Some(location.to_string());
        self
    }

    pub fn with_credentials(mut self, credentials: Credentials) -> Self {
        self.credentials = Some(credentials);
        self
    }

    pub async fn build(self) -> Result<Client, String> {
        // Resolve project: use explicit value, or fall back to GOOGLE_CLOUD_PROJECT env var
        let project = self
            .project
            .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok())
            .ok_or_else(|| {
                "Google Cloud project is required. Set it via ClientBuilder::with_project() or GOOGLE_CLOUD_PROJECT environment variable".to_string()
            })?;

        // Resolve location: use explicit value, or fall back to GOOGLE_CLOUD_LOCATION env var, or default to "global"
        let location = self
            .location
            .or_else(|| std::env::var("GOOGLE_CLOUD_LOCATION").ok())
            .unwrap_or_else(|| DEFAULT_LOCATION.to_string());

        let credentials = self.credentials.clone();
        let mut builder = vertexai::client::PredictionService::builder();
        if let Some(ref creds) = credentials {
            builder = builder.with_credentials(creds.clone());
        }

        let vertex_client = builder
            .build()
            .await
            .map_err(|e| format!("Failed to build Vertex AI client: {e}. Make sure you have Google Cloud credentials configured (e.g., via 'gcloud auth application-default login')"))?;

        Ok(Client {
            project: Some(project),
            location,
            credentials,
            vertex_client: Arc::new(OnceCell::from(vertex_client)),
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
    project: Option<String>,
    location: String,
    credentials: Option<Credentials>,
    pub(crate) vertex_client: Arc<OnceCell<vertexai::client::PredictionService>>,
}

impl Client {
    pub fn new() -> Self {
        let project = std::env::var("GOOGLE_CLOUD_PROJECT").ok();
        let location = std::env::var("GOOGLE_CLOUD_LOCATION")
            .ok()
            .unwrap_or_else(|| DEFAULT_LOCATION.to_string());
        Self {
            project,
            location,
            credentials: None,
            vertex_client: Arc::new(OnceCell::new()),
        }
    }

    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    pub fn project(&self) -> Option<&str> {
        self.project.as_deref()
    }

    pub fn location(&self) -> &str {
        &self.location
    }

    pub async fn get_inner(&self) -> &vertexai::client::PredictionService {
        let credentials = self.credentials.clone();
        self.vertex_client
            .get_or_init(|| async {
                let mut builder = vertexai::client::PredictionService::builder();
                if let Some(creds) = credentials {
                    builder = builder.with_credentials(creds);
                }
                builder
                    .build()
                    .await
                    .expect("Failed to build Vertex AI client. Make sure you have Google Cloud credentials configured (e.g., via 'gcloud auth application-default login')")
            })
            .await
    }
}

impl ProviderClient for Client {
    fn from_env() -> Self
    where
        Self: Sized,
    {
        Client::new()
    }

    fn from_val(_: ProviderValue) -> Self
    where
        Self: Sized,
    {
        panic!(
            "Vertex AI uses Application Default Credentials (ADC). Use `Client::from_env()` for default credentials, or `Client::builder().with_credentials(...).build().await` for custom credentials."
        );
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;

    fn completion_model(&self, model: &str) -> Self::CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl VerifyClient for Client {
    async fn verify(&self) -> Result<(), VerifyError> {
        // No API endpoint to verify credentials - they're validated on first use
        Ok(())
    }
}

impl_conversion_traits!(
    AsTranscription,
    AsEmbeddings,
    AsImageGeneration,
    AsAudioGeneration for Client
);
