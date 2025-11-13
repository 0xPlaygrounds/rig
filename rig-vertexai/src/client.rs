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
/// are also supported and can be specified via `ClientBuilder::new()`.
/// Regional endpoints may be preferred for data residency requirements or to use regional quotas.
pub const DEFAULT_LOCATION: &str = "global";

#[derive(Clone, Debug)]
pub struct ClientBuilder {
    project: String,
    location: String,
    credentials: Option<Credentials>,
}

impl ClientBuilder {
    pub fn new(project: &str, location: &str) -> Self {
        Self {
            project: project.to_string(),
            location: location.to_string(),
            credentials: None,
        }
    }

    pub fn with_credentials(mut self, credentials: Credentials) -> Self {
        self.credentials = Some(credentials);
        self
    }

    pub async fn build(self) -> Result<Client, String> {
        let project = self.project;
        let location = self.location;

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
            project,
            location,
            credentials,
            vertex_client: Arc::new(OnceCell::from(vertex_client)),
        })
    }
}

#[derive(Clone, Debug)]
pub struct Client {
    project: String,
    location: String,
    credentials: Option<Credentials>,
    pub(crate) vertex_client: Arc<OnceCell<vertexai::client::PredictionService>>,
}

impl Client {
    pub fn new() -> Self {
        let project = std::env::var("GOOGLE_CLOUD_PROJECT")
            .expect("GOOGLE_CLOUD_PROJECT environment variable must be set");
        let location =
            std::env::var("GOOGLE_CLOUD_LOCATION").unwrap_or_else(|_| DEFAULT_LOCATION.to_string());
        Self {
            project,
            location,
            credentials: None,
            vertex_client: Arc::new(OnceCell::new()),
        }
    }

    pub fn builder(project: &str, location: &str) -> ClientBuilder {
        ClientBuilder::new(project, location)
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
