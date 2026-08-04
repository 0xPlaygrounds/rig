//! Vertex AI as config + free functions — the crate's entry point.
//!
//! Vertex AI is a non-HTTP transport: requests go through the
//! `google-cloud-aiplatform-v1` SDK's `GenerateContent` RPC rather than a
//! rig-owned HTTP runtime, and authentication runs through Google's OAuth
//! credential chain. The face is therefore a serde [`Config`] describing how
//! to *build* a [`Client`] (never holding one), a [`client_from_config`]
//! constructor over [`crate::client::ClientBuilder`], and free
//! [`complete`]/[`open_stream`] functions taking that live handle.
//!
//! Credential material never lives in the config: like rig-bedrock's
//! [`Config`](https://docs.rs/rig-bedrock), [`CredentialSource`] describes
//! *where* credentials come from (Google Application Default Credentials,
//! optionally impersonating a service account) and resolution happens at
//! [`client_from_config`] time.

use google_cloud_auth::credentials;
use rig_core::completion::{self, CompletionError, CompletionRequest};
use rig_core::providers::descriptor::ProviderDescriptor;
use rig_core::streaming::CompletionStream;
use serde::{Deserialize, Serialize};

use crate::client::{Client, VertexAiClientError};
use crate::completion::rpc_error;
use crate::types::{
    completion_request::VertexCompletionRequest, completion_response::VertexGenerateContentOutput,
};

/// Vertex AI's capability sheet.
///
/// `supports_response_format` is false: the request conversion forwards
/// provider-specific `generation_config` extras but does not wire the typed
/// `output_schema` surface. The streaming and OpenAI-wire fields do not apply
/// to the SDK transport (streaming is unsupported in this integration).
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor::named("vertexai").with_tools(true);

/// Where [`client_from_config`] resolves Google credentials from.
///
/// Never carries key material; both variants defer to Google Application
/// Default Credentials (ADC) at construction time.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CredentialSource {
    /// Application Default Credentials, with optional service-account
    /// impersonation when `GOOGLE_CLOUD_SERVICE_ACCOUNT` is set — exactly
    /// the [`Client::from_env`] behavior.
    #[default]
    ApplicationDefault,
    /// ADC source credentials impersonating an explicit service account
    /// (the config-carried equivalent of `GOOGLE_CLOUD_SERVICE_ACCOUNT`).
    Impersonated {
        /// Target principal, e.g. `my-sa@my-project.iam.gserviceaccount.com`.
        service_account: String,
    },
}

/// Plain-data description of how to build a Vertex AI [`Client`].
///
/// This describes client *construction* (project / location / credential
/// source) as data; it never holds a live SDK client or credentials.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Config {
    /// Google Cloud project ID (`None` defers to `GOOGLE_CLOUD_PROJECT`,
    /// matching [`Client::from_env`]; construction fails without either).
    pub project: Option<String>,
    /// Google Cloud location (`None` defers to `GOOGLE_CLOUD_LOCATION`,
    /// then the `global` endpoint default).
    pub location: Option<String>,
    /// Model identifier requests are built for.
    pub model: String,
    /// Credential resolution strategy.
    #[serde(default)]
    pub credentials: CredentialSource,
}

impl Config {
    /// Config for `model` deferring project/location/credentials to the
    /// environment, matching [`Client::from_env`].
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            project: None,
            location: None,
            model: model.into(),
            credentials: CredentialSource::default(),
        }
    }

    /// Pin the Google Cloud project ID.
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Pin the Google Cloud location (e.g. `us-central1`).
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Impersonate `service_account` from ADC source credentials.
    pub fn with_impersonated_service_account(mut self, service_account: impl Into<String>) -> Self {
        self.credentials = CredentialSource::Impersonated {
            service_account: service_account.into(),
        };
        self
    }
}

/// Build a [`Client`] from `cfg`.
///
/// Resolves credentials per [`Config::credentials`] and falls back to the
/// same environment variables as [`Client::from_env`] for unset fields.
/// Cheap apart from credential construction; the underlying SDK channel is
/// still built lazily on first use.
pub fn client_from_config(cfg: &Config) -> Result<Client, VertexAiClientError> {
    let mut builder = Client::builder();
    if let Some(project) = &cfg.project {
        builder = builder.with_project(project);
    }
    if let Some(location) = &cfg.location {
        builder = builder.with_location(location);
    }
    if let CredentialSource::Impersonated { service_account } = &cfg.credentials {
        let source_credentials = credentials::Builder::default()
            .build()
            .map_err(|e| VertexAiClientError::SourceCredentials(e.to_string()))?;
        let impersonated =
            credentials::impersonated::Builder::from_source_credentials(source_credentials)
                .with_target_principal(service_account.clone())
                .build()
                .map_err(|e| VertexAiClientError::ImpersonatedCredentials(e.to_string()))?;
        builder = builder.with_credentials(impersonated);
    }
    builder.build()
}

/// The full Vertex AI model resource path for `model`. Pure.
pub fn model_path(project: &str, location: &str, model: &str) -> String {
    format!("projects/{project}/locations/{location}/publishers/google/models/{model}")
}

/// Send `request` through the `GenerateContent` RPC and return the
/// normalized response.
///
/// The request conversion itself (`VertexCompletionRequest`'s accessors and
/// [`model_path`]) is pure; only the RPC does IO.
pub async fn complete(
    client: &Client,
    model: &str,
    request: CompletionRequest,
) -> Result<completion::CompletionResponse, CompletionError> {
    tracing::debug!(
        target: "rig_core::vertexai",
        "Vertex AI completion request: {request:?}"
    );

    let vertex_request = VertexCompletionRequest(request);

    let contents = vertex_request.contents()?;
    let generation_config = vertex_request.generation_config()?;
    let system_instruction = vertex_request.system_instruction();
    let tools = vertex_request.tools();
    let tool_config = vertex_request.tool_config();
    let path = model_path(client.project(), client.location(), model);

    let mut request_builder = client
        .get_inner()
        .await
        .map_err(|error| CompletionError::ProviderError(error.to_string()))?
        .generate_content()
        .set_model(&path)
        .set_contents(contents);

    if let Some(config) = generation_config {
        request_builder = request_builder.set_generation_config(config);
    }

    if let Some(system_instruction) = system_instruction {
        request_builder = request_builder.set_system_instruction(system_instruction);
    }

    if let Some(tools) = tools {
        request_builder = request_builder.set_tools([tools]);
    }

    if let Some(tool_config) = tool_config {
        request_builder = request_builder.set_tool_config(tool_config);
    }

    let response = request_builder.send().await.map_err(rpc_error)?;

    tracing::debug!(
        target: "rig_core::vertexai",
        "Vertex AI completion response: {response:?}"
    );

    let vertex_output = VertexGenerateContentOutput(response);
    let completion_response = vertex_output.try_into()?;

    Ok(completion_response)
}

/// Open a streaming completion.
///
/// Streaming is not supported by this integration; this always errors. It
/// exists so hosts can treat Vertex AI uniformly with streaming providers.
pub async fn open_stream(
    _client: &Client,
    _model: &str,
    _request: CompletionRequest,
) -> Result<CompletionStream, CompletionError> {
    Err(CompletionError::ProviderError(
        "Streaming is not supported for Vertex AI in this integration".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_defer_to_environment() {
        let cfg = Config::new("gemini-2.5-flash");
        assert_eq!(cfg.project, None);
        assert_eq!(cfg.location, None);
        assert_eq!(cfg.model, "gemini-2.5-flash");
        assert_eq!(cfg.credentials, CredentialSource::ApplicationDefault);
    }

    #[test]
    fn config_serde_round_trip() {
        let cfg = Config::new("gemini-2.5-pro")
            .with_project("my-project")
            .with_location("us-central1")
            .with_impersonated_service_account("sa@my-project.iam.gserviceaccount.com");
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: Config = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, cfg);
    }

    #[test]
    fn model_path_matches_publisher_resource_shape() {
        assert_eq!(
            model_path("my-project", "global", "gemini-2.5-flash"),
            "projects/my-project/locations/global/publishers/google/models/gemini-2.5-flash"
        );
    }

    #[test]
    // Pinning const capability values is the point of this test.
    #[allow(clippy::assertions_on_constants)]
    fn descriptor_is_honest() {
        assert_eq!(DESCRIPTOR.name, "vertexai");
        assert!(DESCRIPTOR.supports_tools);
        assert!(!DESCRIPTOR.supports_response_format);
        assert!(!DESCRIPTOR.stream_include_usage);
        assert!(!DESCRIPTOR.emits_complete_single_chunk_tool_calls);
    }
}
