//! Azure OpenAI API integration.
//!
//! Azure does not fit the plain `base_url + path` shape: the deployment is
//! routed through the URL and the API is versioned by a query parameter, so
//! [`functions::Config`] carries `endpoint` + `api_version` rather than a base
//! URL, and it covers completions, transcription, image generation, audio
//! generation, and embeddings.
//!
//! # Example
//! ```no_run
//! use rig_core::providers::azure;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! # let request = rig_core::completion::CompletionRequest::from_prompt("hello");
//! let cfg = azure::functions::Config::new(
//!     "https://my-resource.openai.azure.com", // add your endpoint here!
//!     azure::GPT_4O,
//! )
//! .with_api_key("test");
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//! let response = azure::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Authentication
//! Credential presentation is data on the config:
//! [`functions::AuthScheme::ApiKeyHeader`] (the default) sends the resolved
//! credential in Azure's `api-key` header, while
//! [`functions::AuthScheme::Bearer`] sends it as `Authorization: Bearer` — the
//! Entra ID path. [`functions::Config::from_env`] picks the scheme for you:
//! `AZURE_API_KEY` maps to `ApiKeyHeader` and `AZURE_TOKEN` to `Bearer`.

use crate::embeddings::{self, EmbeddingError};
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

// ================================================================
// Azure OpenAI Embedding API
// ================================================================

/// `text-embedding-3-large` embedding model
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
/// `text-embedding-3-small` embedding model
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// `text-embedding-ada-002` embedding model
pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

/// Known embedding dimensionality for a built-in Azure OpenAI embedding
/// deployment.
///
/// Callers that need the vector width (index creation, store schemas) read it
/// from here and, when the model accepts it, pass it to
/// [`functions::EmbeddingConfig::with_dimensions`].
pub fn model_dimensions_from_identifier(identifier: &str) -> Option<usize> {
    match identifier {
        TEXT_EMBEDDING_3_LARGE => Some(3_072),
        TEXT_EMBEDDING_3_SMALL | TEXT_EMBEDDING_ADA_002 => Some(1_536),
        _ => None,
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

impl From<Usage> for crate::completion::Usage {
    fn from(value: Usage) -> crate::completion::Usage {
        let mut usage = crate::completion::Usage::new();

        usage.input_tokens = value.prompt_tokens as u64;
        usage.total_tokens = value.total_tokens as u64;
        usage.output_tokens = usage.total_tokens - usage.input_tokens;

        usage
    }
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {} Total tokens: {}",
            self.prompt_tokens, self.total_tokens
        )
    }
}

/// Build the serialized Azure embeddings request body (`input` +
/// optional `dimensions`). Pure; used by [`functions::embed`].
pub(crate) fn build_embedding_body(
    texts: &[String],
    dimensions: Option<usize>,
) -> Result<Vec<u8>, EmbeddingError> {
    let mut body = json!({
        "input": texts,
    });
    let body_object = body.as_object_mut().ok_or_else(|| {
        EmbeddingError::ResponseError("embedding request body must be a JSON object".into())
    })?;
    if let Some(dimensions) = dimensions {
        body_object.insert("dimensions".to_owned(), json!(dimensions));
    }
    Ok(serde_json::to_vec(&body)?)
}

/// Parse an Azure embeddings response into the normalized
/// [`embeddings::EmbeddingResponse`], zipping vectors back onto
/// `documents`. Pure; used by [`functions::embed`].
pub(crate) fn parse_embedding_response(
    status: http::StatusCode,
    body: &str,
    documents: Vec<String>,
) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
    if !status.is_success() {
        return Err(EmbeddingError::from_http_response(status, body.to_string()));
    }
    let parsed: ApiResponse<EmbeddingResponse> = serde_json::from_str(body)?;
    match parsed {
        ApiResponse::Ok(response) => {
            tracing::info!(target: "rig",
                "Azure embedding token usage: {}",
                response.usage
            );

            if response.data.len() != documents.len() {
                return Err(EmbeddingError::ResponseError(
                    "Response data length does not match input length".into(),
                ));
            }

            let usage = response.usage.clone().into();
            let embeddings = response
                .data
                .into_iter()
                .zip(documents)
                .map(|(embedding, document)| embeddings::Embedding {
                    document,
                    vec: embedding.embedding,
                })
                .collect();
            Ok(embeddings::EmbeddingResponse { embeddings, usage })
        }
        ApiResponse::Err(err) => {
            tracing::warn!(message = %err.message, "provider returned an error response");
            Err(EmbeddingError::from_http_response(status, body.to_string()))
        }
    }
}

// ================================================================
// Azure OpenAI Completion API
// ================================================================

/// `o1` completion model
pub const O1: &str = "o1";
/// `o1-preview` completion model
pub const O1_PREVIEW: &str = "o1-preview";
/// `o1-mini` completion model
pub const O1_MINI: &str = "o1-mini";
/// `gpt-4o` completion model
pub const GPT_4O: &str = "gpt-4o";
/// `gpt-4o-mini` completion model
pub const GPT_4O_MINI: &str = "gpt-4o-mini";
/// `gpt-4o-realtime-preview` completion model
pub const GPT_4O_REALTIME_PREVIEW: &str = "gpt-4o-realtime-preview";
/// `gpt-4-turbo` completion model
pub const GPT_4_TURBO: &str = "gpt-4";
/// `gpt-4` completion model
pub const GPT_4: &str = "gpt-4";
/// `gpt-4-32k` completion model
pub const GPT_4_32K: &str = "gpt-4-32k";
/// `gpt-4-32k` completion model
pub const GPT_4_32K_0613: &str = "gpt-4-32k";
/// `gpt-3.5-turbo` completion model
pub const GPT_35_TURBO: &str = "gpt-3.5-turbo";
/// `gpt-3.5-turbo-instruct` completion model
pub const GPT_35_TURBO_INSTRUCT: &str = "gpt-3.5-turbo-instruct";
/// `gpt-3.5-turbo-16k` completion model
pub const GPT_35_TURBO_16K: &str = "gpt-3.5-turbo-16k";
#[cfg(test)]
mod azure_tests {
    use super::*;
    use crate::OneOrMany;
    use crate::completion::{CompletionError, CompletionRequest};
    use crate::http_runtime::HttpRuntime;

    const TEST_ENDPOINT: &str = "https://example.openai.azure.com";

    fn test_config(model: &str) -> functions::Config {
        functions::Config::new(TEST_ENDPOINT, model).with_api_key("test-key")
    }

    #[cfg(feature = "image")]
    #[tokio::test]
    async fn image_generation_non_success_response_preserves_status_and_body() {
        use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"invalid image request"}}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::with_error_response(
            http::StatusCode::BAD_REQUEST,
            body,
        ));

        let error = functions::generate_image(
            &test_config("dall-e-3"),
            &rt,
            ImageGenerationRequest {
                prompt: "draw a cat".to_string(),
                width: 256,
                height: 256,
                additional_params: None,
            },
        )
        .await
        .expect_err("image generation should fail with non-success status");

        assert!(matches!(error, ImageGenerationError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::BAD_REQUEST)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[cfg(feature = "audio")]
    #[tokio::test]
    async fn audio_generation_non_success_response_preserves_status_and_body() {
        use crate::audio_generation::{AudioGenerationError, AudioGenerationRequest};
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"invalid voice"}}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::with_error_response(
            http::StatusCode::UNPROCESSABLE_ENTITY,
            body,
        ));

        let error = match functions::generate_audio(
            &test_config("tts-1"),
            &rt,
            AudioGenerationRequest {
                text: "hello".to_string(),
                voice: "alloy".to_string(),
                speed: 1.0,
                additional_params: None,
            },
        )
        .await
        {
            Err(error) => error,
            Ok(_) => panic!("audio generation should fail with non-success status"),
        };

        assert!(matches!(error, AudioGenerationError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::UNPROCESSABLE_ENTITY)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn transcription_http_non_success_preserves_status_and_body() {
        use crate::test_utils::RecordingHttpClient;
        use crate::transcription::{TranscriptionError, TranscriptionRequest};

        let body = r#"{"error":{"message":"bad audio","type":"invalid_request_error"}}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::with_error_response(
            http::StatusCode::BAD_REQUEST,
            body,
        ));

        let error = match functions::transcribe(
            &test_config("whisper"),
            &rt,
            TranscriptionRequest::new(vec![0u8; 16]),
        )
        .await
        {
            Err(error) => error,
            Ok(_) => panic!("transcription should fail with non-success status"),
        };

        assert!(matches!(error, TranscriptionError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::BAD_REQUEST)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn embedding_http_non_success_preserves_status_and_body() {
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"bad embedding","type":"invalid_request_error"}}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::with_error_response(
            http::StatusCode::BAD_REQUEST,
            body,
        ));
        let cfg = functions::EmbeddingConfig::new(TEST_ENDPOINT, TEXT_EMBEDDING_3_SMALL)
            .with_api_key("test-key");

        let error = match functions::embed(&cfg, &rt, vec!["Hello, world!".to_string()]).await {
            Err(error) => error,
            Ok(_) => panic!("embedding should fail with non-success status"),
        };

        assert!(matches!(error, EmbeddingError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::BAD_REQUEST)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[test]
    fn completion_pins_deployment_url_under_model_override() {
        let cfg = test_config(GPT_4O_MINI);

        let req = functions::build_request(
            &cfg,
            &CompletionRequest {
                model: Some("other-deployment".to_string()),
                chat_history: OneOrMany::one("Hello!".into()),
                documents: vec![],
                max_tokens: None,
                temperature: None,
                tools: vec![],
                tool_choice: None,
                additional_params: None,
                output_schema: None,
                record_telemetry_content: false,
            },
            false,
        )
        .expect("request should build");

        // The deployment URL stays pinned to the configured model; the
        // override only changes the body.
        let uri = req.uri().to_string();
        assert!(
            uri.contains("/openai/deployments/gpt-4o-mini/chat/completions"),
            "unexpected uri: {uri}"
        );
        let body: serde_json::Value =
            serde_json::from_slice(req.body()).expect("body should be JSON");
        assert_eq!(body["model"], "other-deployment");
    }

    #[tokio::test]
    async fn completion_http_non_success_preserves_status_and_body() {
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"bad completion","type":"invalid_request_error"}}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::with_error_response(
            http::StatusCode::BAD_REQUEST,
            body,
        ));

        let error = match functions::complete(
            &test_config(GPT_4O_MINI),
            &rt,
            CompletionRequest {
                model: None,
                chat_history: OneOrMany::many(vec![
                    crate::message::Message::system("You are a helpful assistant.".to_string()),
                    "Hello!".into(),
                ])
                .expect("non-empty"),
                documents: vec![],
                max_tokens: Some(100),
                temperature: Some(0.0),
                tools: vec![],
                tool_choice: None,
                additional_params: None,
                output_schema: None,
                record_telemetry_content: false,
            },
        )
        .await
        {
            Err(error) => error,
            Ok(_) => panic!("completion should fail with non-success status"),
        };

        assert!(matches!(error, CompletionError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::BAD_REQUEST)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    #[ignore]
    async fn test_azure_embedding() -> anyhow::Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let cfg = functions::EmbeddingConfig::from_env(TEXT_EMBEDDING_3_SMALL)?;
        let rt = HttpRuntime::new();
        let embeddings = functions::embed(&cfg, &rt, vec!["Hello, world!".to_string()]).await?;

        tracing::info!("Azure embedding: {:?}", embeddings);
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_azure_embedding_dimensions() -> anyhow::Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let ndims = 256;
        let cfg =
            functions::EmbeddingConfig::from_env(TEXT_EMBEDDING_3_SMALL)?.with_dimensions(ndims);
        let rt = HttpRuntime::new();
        let response = functions::embed(&cfg, &rt, vec!["Hello, world!".to_string()]).await?;
        let embedding = response
            .embeddings
            .first()
            .ok_or_else(|| anyhow::anyhow!("no embedding returned"))?;

        anyhow::ensure!(
            embedding.vec.len() == ndims,
            "expected embedding dimensions {ndims}, got {}",
            embedding.vec.len()
        );

        tracing::info!("Azure dimensions embedding: {:?}", embedding);
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_azure_completion() -> anyhow::Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let cfg = functions::Config::from_env(GPT_4O_MINI)?;
        let rt = HttpRuntime::new();
        let completion = functions::complete(
            &cfg,
            &rt,
            CompletionRequest {
                model: None,
                chat_history: OneOrMany::many(vec![
                    crate::message::Message::system("You are a helpful assistant.".to_string()),
                    "Hello!".into(),
                ])
                .expect("non-empty"),
                documents: vec![],
                max_tokens: Some(100),
                temperature: Some(0.0),
                tools: vec![],
                tool_choice: None,
                additional_params: None,
                output_schema: None,
                record_telemetry_content: false,
            },
        )
        .await?;

        tracing::info!("Azure completion: {:?}", completion);
        Ok(())
    }
}

pub mod functions {
    //! Azure OpenAI chat completions as config + pure functions.
    //!
    //! The data-oriented face of the Azure OpenAI provider, mirroring
    //! [`crate::providers::openai::functions`]. Azure does not fit the
    //! simple `base_url + path` shape: the deployment (model) is routed
    //! through the URL and the API is versioned via a query parameter, so
    //! [`Config`] carries `endpoint` + `api_version` instead of a
    //! `base_url`, and [`build_request`] assembles that absolute URL via
    //! `completion_path`.
    //!
    //! Authentication is data too: [`AuthScheme::ApiKeyHeader`] (the default)
    //! sends the resolved credential in the `api-key` header, and
    //! [`AuthScheme::Bearer`] sends it as `Authorization: Bearer` — the Entra
    //! ID / `AZURE_TOKEN` path.

    use serde::{Deserialize, Serialize};

    use crate::completion::{self, CompletionError, CompletionRequest};
    use crate::http_runtime::HttpRuntime;
    use crate::providers::descriptor::ChatCompletionsDialect;
    use crate::providers::descriptor::{
        ApiKeyLocation, ConfigError, ProviderDescriptor, optional_env_var, required_env_var,
    };
    use crate::providers::openai::completion::CompletionModelOptions;
    use crate::providers::openai::functions as openai_functions;

    /// Default Azure OpenAI API version (GA).
    pub const DEFAULT_API_VERSION: &str = "2024-10-21";

    /// Azure OpenAI's Chat Completions streaming dialect (OpenAI's own).
    pub(crate) const STREAM_DIALECT: ChatCompletionsDialect =
        ChatCompletionsDialect::from_descriptor(&DESCRIPTOR);

    /// Azure OpenAI's capability sheet.
    pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
        name: "azure.openai",
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        emits_complete_single_chunk_tool_calls: false,
        composes_native_output_with_tools: true,
        max_embedding_documents: Some(1024),
        verify_path: None,
    };

    /// How an Azure credential is presented on the wire.
    ///
    /// Resource keys go in `api-key`, Entra ID tokens go in
    /// `Authorization: Bearer`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
    pub enum AuthScheme {
        /// Send the credential in the `api-key` header (Azure resource keys).
        #[default]
        ApiKeyHeader,
        /// Send the credential as `Authorization: Bearer` (Entra ID tokens —
        /// `AZURE_TOKEN`).
        Bearer,
    }

    /// Resolve which Azure credential variable to read and how to present it.
    ///
    /// `AZURE_API_KEY` wins over `AZURE_TOKEN`.
    ///
    /// # Errors
    /// [`ConfigError::InvalidConfiguration`] when neither variable is set.
    fn resolve_env_credential() -> Result<(&'static str, AuthScheme), ConfigError> {
        if optional_env_var("AZURE_API_KEY")?.is_some() {
            Ok(("AZURE_API_KEY", AuthScheme::ApiKeyHeader))
        } else if optional_env_var("AZURE_TOKEN")?.is_some() {
            Ok(("AZURE_TOKEN", AuthScheme::Bearer))
        } else {
            Err(ConfigError::InvalidConfiguration(
                "either `AZURE_API_KEY` or `AZURE_TOKEN` must be set",
            ))
        }
    }

    /// Apply the configured credential header to `builder`.
    fn apply_auth_header(
        builder: http::request::Builder,
        scheme: AuthScheme,
        key: String,
    ) -> http::request::Builder {
        match scheme {
            AuthScheme::ApiKeyHeader => builder.header("api-key", key),
            AuthScheme::Bearer => {
                builder.header(http::header::AUTHORIZATION, format!("Bearer {key}"))
            }
        }
    }

    /// Plain-data Azure OpenAI provider configuration.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[non_exhaustive]
    pub struct Config {
        /// Resource endpoint, e.g. `https://my-resource.openai.azure.com`.
        pub endpoint: String,
        /// API version query parameter (defaults to [`DEFAULT_API_VERSION`]).
        pub api_version: String,
        /// Credential location.
        pub api_key: ApiKeyLocation,
        /// How the credential is presented (`api-key` header by default).
        #[serde(default)]
        pub auth_scheme: AuthScheme,
        /// Deployment identifier requests are built for (routed through the URL).
        pub model: String,
        /// Extra headers attached to every request.
        pub extra_headers: Vec<(String, String)>,
    }

    impl Config {
        /// Config for `model` on `endpoint`, reading `AZURE_API_KEY` from the
        /// environment.
        pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
            Self {
                endpoint: endpoint.into(),
                api_version: DEFAULT_API_VERSION.to_string(),
                api_key: ApiKeyLocation::Env("AZURE_API_KEY".to_string()),
                auth_scheme: AuthScheme::ApiKeyHeader,
                model: model.into(),
                extra_headers: Vec::new(),
            }
        }

        /// Config for `model` built entirely from the process environment.
        ///
        /// Reads `AZURE_ENDPOINT` and `AZURE_API_VERSION` (both required) and
        /// takes the credential from `AZURE_API_KEY`, falling back to
        /// `AZURE_TOKEN` — the same variables the deleted
        /// `azure::Client::from_env` read. The credential is validated eagerly
        /// but stored as [`ApiKeyLocation::Env`], so the secret is read at
        /// request time rather than held inside the config.
        ///
        /// Credential presentation follows the variable:
        /// `AZURE_API_KEY` is sent in the `api-key` header
        /// ([`AuthScheme::ApiKeyHeader`]), while `AZURE_TOKEN` is sent as
        /// `Authorization: Bearer` ([`AuthScheme::Bearer`], Entra ID).
        ///
        /// # Errors
        /// [`ConfigError`] when a required variable is missing or invalid, or
        /// when neither credential variable is set.
        pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
            let endpoint = required_env_var("AZURE_ENDPOINT")?;
            let api_version = required_env_var("AZURE_API_VERSION")?;
            let (key_var, auth_scheme) = resolve_env_credential()?;
            let mut cfg = Self::new(endpoint, model);
            cfg.api_key = ApiKeyLocation::Env(key_var.to_string());
            cfg.auth_scheme = auth_scheme;
            cfg.api_version = api_version;
            Ok(cfg)
        }

        /// Select the Azure credential variable and its wire presentation.
        pub fn with_auth_scheme(mut self, auth_scheme: AuthScheme) -> Self {
            self.auth_scheme = auth_scheme;
            self
        }

        /// Config with an explicit API key.
        pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
            self.api_key = ApiKeyLocation::Inline(key.into());
            self
        }

        /// Override the API version.
        pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
            self.api_version = api_version.into();
            self
        }
    }

    /// The absolute deployment-scoped chat-completions URL, e.g.
    /// `{endpoint}/openai/deployments/{model}/chat/completions?api-version={v}`.
    ///
    /// Azure routes the deployment (model) through the URL path and versions
    /// the API via a query parameter, so this is a complete URL rather than a
    /// path relative to a base URL.
    pub(crate) fn completion_path(endpoint: &str, api_version: &str, model: &str) -> String {
        format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            endpoint,
            model.trim_start_matches('/'),
            api_version
        )
    }

    /// Azure OpenAI's chat-completions body assembly.
    ///
    /// Identical to OpenAI's: Azure's wire dialect has no body-level quirks
    /// (the deployment and API version live in the URL, not the body).
    pub(crate) fn build_body(
        model: &str,
        request: &CompletionRequest,
        options: CompletionModelOptions,
        stream: bool,
    ) -> Result<Vec<u8>, CompletionError> {
        let typed =
            openai_functions::compatible_typed_request(model, request, &DESCRIPTOR, options)?;
        let body = openai_functions::compatible_body_value(&typed, &DESCRIPTOR, stream)?;
        Ok(serde_json::to_vec(&body)?)
    }

    /// Build the serialized chat-completions request body for `request`. Pure.
    pub fn build_request_body(
        cfg: &Config,
        request: &CompletionRequest,
        stream: bool,
    ) -> Result<Vec<u8>, CompletionError> {
        build_body(
            &cfg.model,
            request,
            CompletionModelOptions::default(),
            stream,
        )
    }

    /// Build the complete HTTP request (URL, headers, body) for `request`.
    ///
    /// Pure except for credential resolution (`ApiKeyLocation::Env` reads the
    /// environment).
    pub fn build_request(
        cfg: &Config,
        request: &CompletionRequest,
        stream: bool,
    ) -> Result<http::Request<Vec<u8>>, CompletionError> {
        // Absolute deployment-scoped URL, e.g.
        // `{endpoint}/openai/deployments/{model}/chat/completions?api-version={v}`.
        let url = completion_path(
            cfg.endpoint.trim_end_matches('/'),
            &cfg.api_version,
            &cfg.model,
        );
        let body = build_request_body(cfg, request, stream)?;

        let mut builder =
            http::Request::post(url).header(http::header::CONTENT_TYPE, "application/json");
        if let Some(key) = cfg
            .api_key
            .resolve()
            .map_err(|e| CompletionError::RequestError(Box::new(e)))?
        {
            builder = apply_auth_header(builder, cfg.auth_scheme, key);
        }
        for (name, value) in &cfg.extra_headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        builder
            .body(body)
            .map_err(|e| CompletionError::RequestError(Box::new(e)))
    }

    /// Parse a chat-completions response body into the normalized
    /// [`completion::CompletionResponse`]. Pure.
    pub fn parse_response(
        status: http::StatusCode,
        body: &str,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        openai_functions::compatible_parse_response::<crate::providers::openai::CompletionResponse>(
            status,
            body,
            DESCRIPTOR.name,
        )
    }

    /// Open a streaming completion for `request`.
    pub async fn open_stream(
        cfg: &Config,
        rt: &HttpRuntime,
        request: CompletionRequest,
    ) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError> {
        let req = build_request(cfg, &request, true)?;
        Ok(openai_functions::compatible_open_stream(
            rt,
            req,
            STREAM_DIALECT,
        ))
    }

    /// The deployment-scoped modality URL, e.g.
    /// `{endpoint}/openai/deployments/{model}/{suffix}?api-version={v}`.
    fn deployment_url(cfg: &Config, suffix: &str) -> String {
        format!(
            "{}/openai/deployments/{}/{}?api-version={}",
            cfg.endpoint.trim_end_matches('/'),
            cfg.model.trim_start_matches('/'),
            suffix,
            cfg.api_version
        )
    }

    fn api_key_request(
        cfg: &Config,
        url: String,
        json_content_type: bool,
    ) -> Result<http::request::Builder, crate::http_client::Error> {
        let mut builder = http::Request::post(url);
        if json_content_type {
            builder = builder.header(http::header::CONTENT_TYPE, "application/json");
        }
        if let Some(key) = cfg
            .api_key
            .resolve()
            .map_err(|e| crate::http_client::Error::Instance(e.into()))?
        {
            builder = apply_auth_header(builder, cfg.auth_scheme, key);
        }
        for (name, value) in &cfg.extra_headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        Ok(builder)
    }

    /// Build the multipart form for a transcription `request`. Pure.
    ///
    /// The deployment rides in the URL, so the form carries no `model`
    /// field, and Azure's translations route takes no `language`.
    pub fn build_transcription_form(
        request: crate::transcription::TranscriptionRequest,
    ) -> Result<crate::http_client::MultipartForm, crate::transcription::TranscriptionError> {
        use crate::http_client::{MultipartForm, multipart::Part};
        use crate::transcription::TranscriptionError;

        let mut body =
            MultipartForm::new().part(Part::bytes("file", request.data).filename(request.filename));
        if let Some(prompt) = request.prompt {
            body = body.text("prompt", prompt);
        }
        if let Some(ref temperature) = request.temperature {
            body = body.text("temperature", temperature.to_string());
        }
        if let Some(ref additional_params) = request.additional_params {
            let params = additional_params.as_object().ok_or_else(|| {
                TranscriptionError::RequestError(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "additional transcription parameters must be a JSON object",
                )))
            })?;
            for (key, value) in params {
                body = body.text(key.to_owned(), value.to_string());
            }
        }
        Ok(body)
    }

    /// Transcribe `request` with the deployment's `audio/translations`
    /// endpoint. Responses share OpenAI's transcription envelope.
    pub async fn transcribe(
        cfg: &Config,
        rt: &HttpRuntime,
        request: crate::transcription::TranscriptionRequest,
    ) -> Result<
        crate::transcription::TranscriptionResponse<
            crate::providers::openai::TranscriptionResponse,
        >,
        crate::transcription::TranscriptionError,
    > {
        let form = build_transcription_form(request)?;
        let url = deployment_url(cfg, "audio/translations");
        let req = api_key_request(cfg, url, false)?
            .body(form)
            .map_err(|e| crate::transcription::TranscriptionError::RequestError(Box::new(e)))?;
        let (status, body) = rt.send_multipart(req).await?;
        openai_functions::parse_transcription_response(status, &body)
    }

    /// Build the serialized image-generation request body. Pure.
    ///
    /// Azure always requests `b64_json` (no `gpt-image` carve-out).
    #[cfg(feature = "image")]
    pub fn build_image_generation_body(
        model: &str,
        request: &crate::image_generation::ImageGenerationRequest,
    ) -> Result<Vec<u8>, crate::image_generation::ImageGenerationError> {
        Ok(serde_json::to_vec(&serde_json::json!({
            "model": model,
            "prompt": request.prompt,
            "size": format!("{}x{}", request.width, request.height),
            "response_format": "b64_json",
        }))?)
    }

    /// Generate an image with the deployment's `images/generations` endpoint.
    /// Responses share OpenAI's image-generation envelope.
    #[cfg(feature = "image")]
    pub async fn generate_image(
        cfg: &Config,
        rt: &HttpRuntime,
        request: crate::image_generation::ImageGenerationRequest,
    ) -> Result<
        crate::image_generation::ImageGenerationResponse<
            crate::providers::openai::ImageGenerationResponse,
        >,
        crate::image_generation::ImageGenerationError,
    > {
        let body = build_image_generation_body(&cfg.model, &request)?;
        let url = deployment_url(cfg, "images/generations");
        let req = api_key_request(cfg, url, true)?.body(body).map_err(|e| {
            crate::image_generation::ImageGenerationError::RequestError(Box::new(e))
        })?;
        let (status, body) = rt.send_bytes(req).await?;
        openai_functions::parse_image_generation_response(status, &body)
    }

    /// Generate speech with the deployment's `audio/speech` endpoint. The
    /// request body shares OpenAI's TTS shape; success bodies are raw audio.
    #[cfg(feature = "audio")]
    pub async fn generate_audio(
        cfg: &Config,
        rt: &HttpRuntime,
        request: crate::audio_generation::AudioGenerationRequest,
    ) -> Result<
        crate::audio_generation::AudioGenerationResponse<bytes::Bytes>,
        crate::audio_generation::AudioGenerationError,
    > {
        let body = openai_functions::build_audio_generation_body(&cfg.model, &request)?;
        let url = deployment_url(cfg, "audio/speech");
        let req = api_key_request(cfg, url, true)?.body(body).map_err(|e| {
            crate::audio_generation::AudioGenerationError::RequestError(Box::new(e))
        })?;
        let (status, body) = rt.send_bytes(req).await?;
        openai_functions::parse_audio_generation_response(status, body)
    }

    /// Send `request` to Azure OpenAI and return the normalized response.
    pub async fn complete(
        cfg: &Config,
        rt: &HttpRuntime,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        let req = build_request(cfg, &request, false)?;
        let (status, body) = rt.send(req).await?;
        parse_response(status, &body)
    }

    // ================================================================
    // Embeddings
    // ================================================================

    /// Plain-data Azure OpenAI embeddings configuration.
    ///
    /// A sibling of [`Config`]: embeddings target their own deployment
    /// (`model`) and optionally request a dimension count, which do not
    /// belong on the completion configuration.
    #[derive(Debug, Clone, PartialEq, serde::Serialize, Deserialize)]
    #[non_exhaustive]
    pub struct EmbeddingConfig {
        /// Resource endpoint, e.g. `https://my-resource.openai.azure.com`.
        pub endpoint: String,
        /// API version query parameter (defaults to [`DEFAULT_API_VERSION`]).
        pub api_version: String,
        /// Credential location.
        pub api_key: crate::providers::descriptor::ApiKeyLocation,
        /// How the credential is presented (`api-key` header by default).
        #[serde(default)]
        pub auth_scheme: AuthScheme,
        /// Embedding deployment identifier (routed through the URL).
        pub model: String,
        /// Requested embedding dimensions, sent verbatim as `dimensions`
        /// when set (models that reject the field, like
        /// `text-embedding-ada-002`, should leave it unset).
        pub dimensions: Option<usize>,
        /// Extra headers attached to every request.
        pub extra_headers: Vec<(String, String)>,
    }

    impl EmbeddingConfig {
        /// Config for the `model` deployment on `endpoint`, reading
        /// `AZURE_API_KEY` from the environment.
        pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
            Self {
                endpoint: endpoint.into(),
                api_version: DEFAULT_API_VERSION.to_string(),
                api_key: ApiKeyLocation::Env("AZURE_API_KEY".to_string()),
                auth_scheme: AuthScheme::ApiKeyHeader,
                model: model.into(),
                dimensions: None,
                extra_headers: Vec::new(),
            }
        }

        /// Embedding config for the `model` deployment, built entirely from the
        /// process environment.
        ///
        /// Same variables as [`Config::from_env`]: `AZURE_ENDPOINT` and
        /// `AZURE_API_VERSION` (both required), plus `AZURE_API_KEY` with
        /// `AZURE_TOKEN` as a fallback. The credential is validated eagerly but
        /// stored as [`ApiKeyLocation::Env`]; `AZURE_API_KEY` is sent in the
        /// `api-key` header and `AZURE_TOKEN` as `Authorization: Bearer`.
        ///
        /// # Errors
        /// [`ConfigError`] when a required variable is missing or invalid, or
        /// when neither credential variable is set.
        pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
            let endpoint = required_env_var("AZURE_ENDPOINT")?;
            let api_version = required_env_var("AZURE_API_VERSION")?;
            let (key_var, auth_scheme) = resolve_env_credential()?;
            let mut cfg = Self::new(endpoint, model);
            cfg.api_key = ApiKeyLocation::Env(key_var.to_string());
            cfg.auth_scheme = auth_scheme;
            cfg.api_version = api_version;
            Ok(cfg)
        }

        /// Select how the credential is presented on the wire.
        pub fn with_auth_scheme(mut self, auth_scheme: AuthScheme) -> Self {
            self.auth_scheme = auth_scheme;
            self
        }

        /// Config with an explicit API key.
        pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
            self.api_key = ApiKeyLocation::Inline(key.into());
            self
        }

        /// Override the API version.
        pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
            self.api_version = api_version.into();
            self
        }

        /// Request `dimensions`-sized embeddings.
        pub fn with_dimensions(mut self, dimensions: usize) -> Self {
            self.dimensions = Some(dimensions);
            self
        }
    }

    /// Build the complete HTTP embeddings request for one chunk of `texts`.
    ///
    /// Pure except for credential resolution. The URL is the
    /// deployment-scoped shape:
    /// `{endpoint}/openai/deployments/{model}/embeddings?api-version={v}`.
    pub fn build_embedding_request(
        cfg: &EmbeddingConfig,
        texts: &[String],
    ) -> Result<http::Request<Vec<u8>>, crate::embeddings::EmbeddingError> {
        use crate::embeddings::EmbeddingError;

        let body = super::build_embedding_body(texts, cfg.dimensions)?;
        let url = format!(
            "{}/openai/deployments/{}/embeddings?api-version={}",
            cfg.endpoint.trim_end_matches('/'),
            cfg.model.trim_start_matches('/'),
            cfg.api_version
        );
        let mut builder =
            http::Request::post(url).header(http::header::CONTENT_TYPE, "application/json");
        if let Some(key) = cfg
            .api_key
            .resolve()
            .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?
        {
            builder = apply_auth_header(builder, cfg.auth_scheme, key);
        }
        for (name, value) in &cfg.extra_headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        builder
            .body(body)
            .map_err(|e| EmbeddingError::ProviderError(e.to_string()))
    }

    /// Parse an embeddings response into the normalized
    /// [`crate::embeddings::EmbeddingResponse`]. Pure.
    pub fn parse_embedding_response(
        status: http::StatusCode,
        body: &str,
        documents: Vec<String>,
    ) -> Result<crate::embeddings::EmbeddingResponse, crate::embeddings::EmbeddingError> {
        super::parse_embedding_response(status, body, documents)
    }

    /// Embed `texts`, chunking to honor [`DESCRIPTOR`]'s
    /// `max_embedding_documents`; embeddings are returned in input order.
    pub async fn embed(
        cfg: &EmbeddingConfig,
        rt: &HttpRuntime,
        texts: Vec<String>,
    ) -> Result<crate::embeddings::EmbeddingResponse, crate::embeddings::EmbeddingError> {
        crate::embeddings::batching::embed_chunked(
            rt,
            texts,
            DESCRIPTOR.max_embedding_documents,
            |chunk| build_embedding_request(cfg, chunk),
            parse_embedding_response,
        )
        .await
    }

    /// Embed caller-defined batches, returning one order-aligned
    /// [`OneOrMany`](crate::OneOrMany) group per input batch plus summed
    /// usage.
    pub async fn embed_batches(
        cfg: &EmbeddingConfig,
        rt: &HttpRuntime,
        texts: Vec<Vec<String>>,
    ) -> Result<
        (
            Vec<crate::OneOrMany<crate::embeddings::Embedding>>,
            crate::completion::Usage,
        ),
        crate::embeddings::EmbeddingError,
    > {
        let (counts, flat) = crate::embeddings::batching::split_batches(texts);
        let response = embed(cfg, rt, flat).await?;
        let groups = crate::embeddings::batching::group_batches(&counts, response.embeddings)?;
        Ok((groups, response.usage))
    }

    /// Credential verification is not available for this provider.
    ///
    /// The deleted client declared `const VERIFY_PATH: &'static str = ""`, so the
    /// classic `verify()` issued a bare `GET` of the base URL — a request that
    /// checked no credential. [`DESCRIPTOR`] therefore carries no `verify_path`
    /// and this reports the fact rather than repeating the empty check.
    ///
    /// # Errors
    /// Always [`VerifyError::Unsupported`](crate::providers::verify::VerifyError::Unsupported).
    pub async fn verify(
        cfg: &Config,
        rt: &HttpRuntime,
    ) -> Result<(), crate::providers::verify::VerifyError> {
        let _ = (cfg, rt);
        Err(crate::providers::verify::VerifyError::Unsupported {
            provider: DESCRIPTOR.name,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::OneOrMany;

        fn sample_request() -> CompletionRequest {
            CompletionRequest {
                model: None,
                chat_history: OneOrMany::one(crate::message::Message::user("hello")),
                documents: Vec::new(),
                tools: Vec::new(),
                temperature: Some(0.5),
                max_tokens: Some(64),
                tool_choice: None,
                additional_params: None,
                output_schema: None,
                record_telemetry_content: false,
            }
        }

        #[test]
        fn build_request_sets_url_and_model() {
            let cfg = Config::new("https://my-resource.openai.azure.com", "gpt-4o-deploy")
                .with_api_key("secret");
            let req = build_request(&cfg, &sample_request(), false).expect("build");
            assert_eq!(
                req.uri(),
                "https://my-resource.openai.azure.com/openai/deployments/gpt-4o-deploy/chat/completions?api-version=2024-10-21"
            );
            assert_eq!(
                req.headers().get("api-key").and_then(|v| v.to_str().ok()),
                Some("secret")
            );
            let value: serde_json::Value = serde_json::from_slice(req.body()).expect("json");
            assert_eq!(value["model"], "gpt-4o-deploy");
        }

        #[test]
        fn parse_response_normalizes() {
            let body = serde_json::json!({
                "id": "chatcmpl-1",
                "model": "gpt-4o-2024",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "logprobs": null,
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
            })
            .to_string();
            let response = parse_response(http::StatusCode::OK, &body).expect("parse");
            assert_eq!(response.provider, "azure.openai");
            assert_eq!(response.usage.input_tokens, 3);
            assert_eq!(response.usage.total_tokens, 5);
        }
    }
}
