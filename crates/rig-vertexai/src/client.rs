use crate::completion::CompletionModel;
use google_cloud_aiplatform_v1 as vertexai;
use google_cloud_auth::credentials;
use google_cloud_auth::credentials::Credentials;
use rig_core::client::VerifyError;
use rig_core::driver::CompletionProvider;
use std::sync::Arc;
use thiserror::Error;
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

#[derive(Clone, Debug, Error)]
pub enum VertexAiClientError {
    #[error(
        "Google Cloud project is required. Set it via `ClientBuilder::with_project()` or `GOOGLE_CLOUD_PROJECT`"
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
        "a supplied `PredictionService` already carries its own credentials; drop either `ClientBuilder::with_credentials()` or `ClientBuilder::with_prediction_service()`"
    )]
    ConflictingCredentials,
}

/// Helper function to build credentials with optional service account impersonation.
///
/// Every Application Default Credentials branch (`mds`, `service_account`,
/// `user_account`, `external_account`, `impersonated`) wraps its token
/// provider in `google-cloud-auth`'s token cache, and that cache calls
/// `tokio::spawn` for its refresh task while it is being constructed. So this
/// function must run inside a Tokio runtime context, and the runtime that
/// accepted the spawn has to stay alive and driven for as long as the
/// credentials are used — see [`Client::from_env`].
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

        // Check for service account impersonation
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
pub struct ClientBuilder {
    project: Option<String>,
    location: Option<String>,
    credentials: Option<Credentials>,
    prediction_service: Option<vertexai::client::PredictionService>,
}

impl ClientBuilder {
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
    /// Combining this with [`Self::with_credentials`] is a contradiction —
    /// the supplied client's credentials are already fixed — and
    /// [`Self::build`] rejects it with
    /// [`VertexAiClientError::ConflictingCredentials`].
    ///
    /// ```no_run
    /// # use google_cloud_aiplatform_v1::client::PredictionService;
    /// # async fn example() -> anyhow::Result<()> {
    /// let service = PredictionService::builder()
    ///     .with_endpoint("https://us-central1-aiplatform.googleapis.com")
    ///     .build()
    ///     .await?;
    /// let client = rig_vertexai::Client::builder()
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
    /// (see [`Client::from_env`] for the runtime that entails) and builds the
    /// Vertex AI client lazily on first use via [`Client::inner`].
    pub fn build(self) -> Result<Client, VertexAiClientError> {
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

        Ok(Client {
            project,
            location,
            service,
        })
    }
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Where this client's Vertex AI prediction service comes from.
///
/// The two cases are genuinely different owners, so they are different
/// variants rather than a pre-filled cell: a supplied client is already built
/// by the host and has nothing left for Rig to initialize. Construction alone
/// does not verify connectivity or authenticate a request.
#[derive(Clone, Debug)]
enum PredictionServiceSource {
    /// Rig resolved the credentials and builds the SDK client on first use,
    /// on whichever runtime drives that first completion. The result — error
    /// included — is shared permanently by every clone of the client. Correct
    /// invalid construction settings by building a fresh client; this cell is
    /// not a request-retry policy.
    Deferred {
        credentials: Credentials,
        client: Arc<OnceCell<Result<vertexai::client::PredictionService, VertexAiClientError>>>,
    },
    /// The host built the SDK client and owns its lifetime, endpoint and
    /// credentials. Cancelling one completion never touches it.
    Supplied(vertexai::client::PredictionService),
}

#[derive(Clone, Debug)]
pub struct Client {
    project: String,
    location: String,
    service: PredictionServiceSource,
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
    /// # fn example() -> Result<(), rig_vertexai::client::VertexAiClientError> {
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
    pub fn new() -> Result<Self, VertexAiClientError> {
        ClientBuilder::new().build()
    }

    /// Create a client using environment variables for project, location, and credentials.
    ///
    /// Reads from:
    /// - `GOOGLE_CLOUD_PROJECT` (required)
    /// - `GOOGLE_CLOUD_LOCATION` (optional, defaults to "global")
    /// - `GOOGLE_CLOUD_SERVICE_ACCOUNT` (optional, for service account impersonation)
    ///
    /// # Runtime
    ///
    /// Resolving Application Default Credentials constructs
    /// `google-cloud-auth`'s token cache, which `tokio::spawn`s a refresh task
    /// as part of construction. Call this from inside a Tokio runtime context
    /// — otherwise this returns [`VertexAiClientError::RuntimeRequired`] — and keep that runtime alive and driven
    /// for as long as the client is used: the refresh task belongs to the
    /// runtime that accepted it, not to any one completion, and dropping the
    /// runtime drops it. A host that builds the client on a temporary runtime
    /// and then completes on another will find the credentials un-refreshable.
    ///
    /// Hosts that would rather own that lifetime themselves can build a
    /// [`google_cloud_aiplatform_v1::client::PredictionService`] and pass it to
    /// [`ClientBuilder::with_prediction_service`], which resolves no
    /// credentials here at all.
    pub fn from_env() -> Result<Self, VertexAiClientError> {
        Client::new()
    }

    /// Vertex AI exposes no credential-check endpoint: Application Default
    /// Credentials are validated on first use, so there is nothing to call
    /// here.
    pub async fn verify(&self) -> Result<(), VerifyError> {
        Ok(())
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

impl CompletionProvider for Client {
    type Model = CompletionModel;

    fn completion(&self, model: impl Into<String>) -> Self::Model {
        CompletionModel::new(self.clone(), model.into())
    }
}
