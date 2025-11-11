use crate::completion::CompletionModel;
use google_cloud_aiplatform_v1 as vertexai;
use google_cloud_auth::credentials::Credentials;
use rig::client::{CompletionClient, ProviderValue};
use rig::impl_conversion_traits;
use rig::prelude::*;
use std::sync::Arc;
use tokio::sync::OnceCell;

/// Default region for Vertex AI Gemini models.
///
/// The `global` endpoint is recommended for Gemini models as it provides higher availability
/// and reduces resource exhaustion errors. Regional endpoints (e.g., `us-central1`, `europe-west4`)
/// are also supported and can be specified via `ClientBuilder::with_google_cloud_region()`.
/// Regional endpoints may be preferred for data residency requirements or to use regional quotas.
pub const DEFAULT_REGION: &str = "global";

#[derive(Clone, Debug)]
pub struct ClientBuilder {
    project_id: Option<String>,
    region: Option<String>,
    credentials: Option<Credentials>,
}

impl ClientBuilder {
    pub fn new() -> Self {
        Self {
            project_id: None,
            region: None,
            credentials: None,
        }
    }

    pub fn with_google_cloud_project(mut self, project_id: &str) -> Self {
        self.project_id = Some(project_id.to_string());
        self
    }

    pub fn with_google_cloud_region(mut self, region: &str) -> Self {
        self.region = Some(region.to_string());
        self
    }

    pub fn with_credentials(mut self, credentials: Credentials) -> Self {
        self.credentials = Some(credentials);
        self
    }

    pub async fn build(self) -> Result<Client, String> {
        // Resolve project_id: use explicit value, or fall back to GOOGLE_CLOUD_PROJECT env var
        let project_id = self
            .project_id
            .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok())
            .ok_or_else(|| {
                "Google Cloud project ID is required. Set it via ClientBuilder::with_google_cloud_project() or GOOGLE_CLOUD_PROJECT environment variable".to_string()
            })?;

        let region = self
            .region
            .clone()
            .unwrap_or_else(|| DEFAULT_REGION.to_string());

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
            project_id: Some(project_id),
            region,
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
    project_id: Option<String>,
    region: String,
    credentials: Option<Credentials>,
    pub(crate) vertex_client: Arc<OnceCell<vertexai::client::PredictionService>>,
}

impl Client {
    pub fn new() -> Self {
        let project_id = std::env::var("GOOGLE_CLOUD_PROJECT").ok();
        Self {
            project_id,
            region: DEFAULT_REGION.to_string(),
            credentials: None,
            vertex_client: Arc::new(OnceCell::new()),
        }
    }

    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    pub fn project_id(&self) -> Option<&str> {
        self.project_id.as_deref()
    }

    pub fn region(&self) -> &str {
        &self.region
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
