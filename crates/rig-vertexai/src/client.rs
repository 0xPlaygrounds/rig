use google_cloud_aiplatform_v1 as vertexai;
use google_cloud_auth::credentials;
use google_cloud_auth::credentials::Credentials;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::OnceCell;

// Env vars and terminology (location, project) chosen to match google genai client
// https://googleapis.github.io/python-genai/genai.html#genai.client.Client

/// Default model-resource location, used when neither the builder nor environment sets one.
pub const DEFAULT_LOCATION: &str = "global";

#[derive(Clone, Debug, Error)]
pub enum VertexAiClientError {
    #[error(
        "Google Cloud project is required. Set it via `VertexAiBuilder::with_project()` or `GOOGLE_CLOUD_PROJECT`"
    )]
    MissingProject,
    #[error(
        "construct Vertex ADC credentials inside a Tokio runtime context; the host must retain and drive that runtime"
    )]
    RuntimeRequired,
    #[error("failed to build source credentials: {0}")]
    SourceCredentials(String),
    #[error("failed to build impersonated credentials: {0}")]
    ImpersonatedCredentials(String),
    #[error("failed to build Vertex AI prediction service: {0}")]
    PredictionService(String),
    #[error(
        "a supplied `PredictionService` already carries its own credentials; drop either `VertexAiBuilder::with_credentials()` or `VertexAiBuilder::with_prediction_service()`"
    )]
    ConflictingCredentials,
}

/// Returns explicit credentials or resolves ADC with optional impersonation.
/// ADC construction requires a Tokio runtime that stays alive and driven while
/// credentials are used, because their refresh task runs on that runtime.
fn build_credentials(
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
            .map_err(|e| VertexAiClientError::SourceCredentials(e.to_string()))?;

        if let Ok(service_account) = std::env::var("GOOGLE_CLOUD_SERVICE_ACCOUNT") {
            credentials::impersonated::Builder::from_source_credentials(source_credentials)
                .with_target_principal(service_account)
                .build()
                .map_err(|e| VertexAiClientError::ImpersonatedCredentials(e.to_string()))
        } else {
            Ok(source_credentials)
        }
    }
}

#[derive(Clone, Debug)]
pub struct VertexAiBuilder {
    project: Option<String>,
    location: Option<String>,
    credentials: Option<Credentials>,
    prediction_service: Option<vertexai::client::PredictionService>,
}

impl VertexAiBuilder {
    pub fn new() -> Self {
        Self {
            project: None,
            location: None,
            credentials: None,
            prediction_service: None,
        }
    }

    /// Set the Google Cloud project ID explicitly.
    ///
    /// If not set, will fall back to `GOOGLE_CLOUD_PROJECT` environment variable.
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Set the Google Cloud location explicitly.
    ///
    /// If not set, will fall back to `GOOGLE_CLOUD_LOCATION` environment variable,
    /// or default to "global" if the env var is also not set.
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
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

    /// Use a Vertex AI prediction service the host already built.
    ///
    /// The SDK client carries its own endpoint, credentials, transport,
    /// retry and universe-domain settings: this crate takes it as given and
    /// never rebuilds it, resolves Application Default Credentials for it, or
    /// picks an endpoint of its own. `project` and `location` still have to be
    /// configured here (explicitly or from the environment), because they name
    /// the model resource in the request rather than the connection.
    ///
    /// Combining this with [`Self::with_credentials`] makes [`Self::build`]
    /// return [`VertexAiClientError::ConflictingCredentials`].
    ///
    /// ```no_run
    /// # use google_cloud_aiplatform_v1::client::PredictionService;
    /// # async fn example() -> anyhow::Result<()> {
    /// let service = PredictionService::builder()
    ///     .with_endpoint("https://us-central1-aiplatform.googleapis.com")
    ///     .build()
    ///     .await?;
    /// let client = rig_vertexai::VertexAi::builder()
    ///     .with_project("my-project")
    ///     .with_location("us-central1")
    ///     .with_prediction_service(service)
    ///     .build()?;
    /// # let _ = client;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_prediction_service(
        mut self,
        prediction_service: vertexai::client::PredictionService,
    ) -> Self {
        self.prediction_service = Some(prediction_service);
        self
    }

    /// Build the client with the configured values, falling back to environment variables where not set.
    ///
    /// Without a supplied prediction service this resolves credentials now
    /// (see [`VertexAi::from_env`] for the runtime that entails) and builds the
    /// Vertex AI client lazily on first use via [`VertexAi::inner`].
    pub fn build(self) -> Result<VertexAi, VertexAiClientError> {
        let project = self
            .project
            .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok())
            .ok_or(VertexAiClientError::MissingProject)?;

        let location = self
            .location
            .or_else(|| std::env::var("GOOGLE_CLOUD_LOCATION").ok())
            .unwrap_or_else(|| DEFAULT_LOCATION.to_string());

        let service = match self.prediction_service {
            Some(prediction_service) => {
                if self.credentials.is_some() {
                    return Err(VertexAiClientError::ConflictingCredentials);
                }
                PredictionServiceSource::Supplied(prediction_service)
            }
            None => PredictionServiceSource::Deferred {
                credentials: build_credentials(self.credentials)?,
                client: Arc::new(OnceCell::new()),
            },
        };

        Ok(VertexAi {
            project,
            location,
            service,
        })
    }
}

impl Default for VertexAiBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Supplied or lazily initialized prediction service.
/// Construction does not verify connectivity or authenticate requests.
#[derive(Clone, Debug)]
enum PredictionServiceSource {
    /// Builds the SDK client on first use and shares the completed result,
    /// including errors, across clones. Correcting settings requires a new client.
    Deferred {
        credentials: Credentials,
        client: Arc<OnceCell<Result<vertexai::client::PredictionService, VertexAiClientError>>>,
    },
    /// The host built the SDK client and owns its lifetime, endpoint and
    /// credentials. Cancelling one completion never touches it.
    Supplied(vertexai::client::PredictionService),
}

#[derive(Clone, Debug)]
pub struct VertexAi {
    project: String,
    location: String,
    service: PredictionServiceSource,
}

impl VertexAi {
    /// Creates a builder using environment values for unset project and location.
    /// See [`VertexAiBuilder::build`] for credential and runtime requirements.
    pub fn builder() -> VertexAiBuilder {
        VertexAiBuilder::new()
    }

    /// Create a new client using environment variables for project, location, and credentials.
    ///
    /// Reads from:
    /// - `GOOGLE_CLOUD_PROJECT` (required)
    /// - `GOOGLE_CLOUD_LOCATION` (optional, defaults to "global")
    /// - `GOOGLE_CLOUD_SERVICE_ACCOUNT` (optional, for service account impersonation)
    ///
    pub fn new() -> Result<Self, VertexAiClientError> {
        VertexAiBuilder::new().build()
    }

    /// Create a client using environment variables for project, location, and credentials.
    ///
    /// Reads from:
    /// - `GOOGLE_CLOUD_PROJECT` (required)
    /// - `GOOGLE_CLOUD_LOCATION` (optional, defaults to "global")
    /// - `GOOGLE_CLOUD_SERVICE_ACCOUNT` (optional, for service account impersonation)
    ///
    /// Requires a Tokio runtime context or returns [`VertexAiClientError::RuntimeRequired`].
    /// Keep that runtime alive and driven for the client's lifetime: credential
    /// construction spawns a refresh task on it. A supplied service through
    /// [`VertexAiBuilder::with_prediction_service`] bypasses credential resolution.
    pub fn from_env() -> Result<Self, VertexAiClientError> {
        VertexAi::new()
    }

    pub fn project(&self) -> &str {
        &self.project
    }

    pub fn location(&self) -> &str {
        &self.location
    }

    /// The underlying Vertex AI prediction service, built on first use unless
    /// the host supplied one. A completed initialization error is cached across
    /// all clones; correct construction settings by building a new client.
    /// Request retry policy remains owned by the SDK.
    pub async fn inner(&self) -> Result<&vertexai::client::PredictionService, VertexAiClientError> {
        match &self.service {
            PredictionServiceSource::Supplied(service) => Ok(service),
            PredictionServiceSource::Deferred {
                credentials,
                client,
            } => client
                .get_or_init(|| async {
                    vertexai::client::PredictionService::builder()
                        .with_credentials(credentials.clone())
                        .build()
                        .await
                        .map_err(|error| VertexAiClientError::PredictionService(error.to_string()))
                })
                .await
                .as_ref()
                .map_err(Clone::clone),
        }
    }
}

