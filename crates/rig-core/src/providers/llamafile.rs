//! Llamafile API client and Rig integration
//!
//! [Llamafile](https://github.com/Mozilla-Ocho/llamafile) is a Mozilla Builders project
//! that distributes LLMs as single-file executables. When started, it exposes an
//! OpenAI-compatible API at `http://localhost:8080/v1`.
//!
//! # Example
//! ```no_run
//! use rig_core::{
//!     client::CompletionClient,
//!     completion::CompletionModel,
//!     providers::llamafile,
//! };
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a new Llamafile client (defaults to http://localhost:8080)
//! let client = llamafile::Client::from_url("http://localhost:8080")?;
//!
//! // Send a completion request with a preamble.
//! let model = client.completion_model(llamafile::LLAMA_CPP);
//! let request = model
//!     .completion_request("Hello!")
//!     .preamble("You are a helpful assistant.".to_string())
//!     .build();
//! let response = model.completion(request).await?;
//! println!("{:?}", response.choice);
//! # Ok(())
//! # }
//! ```

use crate::client::{
    self, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder, ProviderClient,
    Transport,
};
use crate::http_client::{self, HttpClientExt};
use crate::providers::openai;

// ================================================================
// Main Llamafile Client
// ================================================================
const LLAMAFILE_API_BASE_URL: &str = "http://localhost:8080";

/// The default model identifier reported by llamafile.
pub const LLAMA_CPP: &str = "LLaMA_CPP";

#[derive(Debug, Default, Clone, Copy)]
pub struct LlamafileExt;

#[derive(Debug, Default, Clone, Copy)]
pub struct LlamafileBuilder;

impl Provider for LlamafileExt {
    type Builder = LlamafileBuilder;
    const VERIFY_PATH: &'static str = "/models";

    // Llamafile clients are constructed from a bare host URL
    // (e.g. `http://localhost:8080`) while the shared OpenAI-compatible
    // endpoints are relative to `/v1`.
    fn build_uri(&self, base_url: &str, path: &str, _transport: Transport) -> String {
        let base_url = base_url.trim_end_matches('/');
        format!("{base_url}/v1/{}", path.trim_start_matches('/'))
    }
}

impl openai::completion::OpenAICompatibleProvider for LlamafileExt {
    const PROVIDER_NAME: &'static str = "llamafile";

    type StreamingUsage = openai::Usage;

    // llama.cpp-based servers can emit a whole tool call in one streaming chunk.
    const EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS: bool = true;

    type Response = openai::CompletionResponse;
}

impl openai::embedding::OpenAIEmbeddingsCompatible for LlamafileExt {
    const PROVIDER_NAME: &'static str = "llamafile";
}

impl<H> Capabilities<H> for LlamafileExt {
    type Completion = Capable<openai::completion::GenericCompletionModel<LlamafileExt, H>>;
    type Embeddings = Capable<openai::embedding::GenericEmbeddingModel<LlamafileExt, H>>;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
    type Rerank = Nothing;
}

impl DebugExt for LlamafileExt {}

impl ProviderBuilder for LlamafileBuilder {
    type Extension<H>
        = LlamafileExt
    where
        H: HttpClientExt;
    type ApiKey = Nothing;

    const BASE_URL: &'static str = LLAMAFILE_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(LlamafileExt)
    }
}

pub type Client<H = reqwest::Client> = client::Client<LlamafileExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<LlamafileBuilder, Nothing, H>;

/// Llamafile completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = reqwest::Client> =
    openai::completion::GenericCompletionModel<LlamafileExt, H>;

/// Llamafile embedding model, driven by the shared OpenAI embeddings path.
pub type EmbeddingModel<H = reqwest::Client> =
    openai::embedding::GenericEmbeddingModel<LlamafileExt, H>;

impl Client {
    /// Create a client pointing at the given llamafile base URL
    /// (e.g. `http://localhost:8080`).
    pub fn from_url(base_url: &str) -> crate::client::ProviderClientResult<Self> {
        Self::builder()
            .api_key(Nothing)
            .base_url(base_url)
            .build()
            .map_err(Into::into)
    }
}

impl ProviderClient for Client {
    type Input = Nothing;
    type Error = crate::client::ProviderClientError;

    fn from_env() -> Result<Self, Self::Error> {
        let api_base = crate::client::required_env_var("LLAMAFILE_API_BASE_URL")?;
        Self::from_url(&api_base)
    }

    fn from_val(_: Self::Input) -> Result<Self, Self::Error> {
        Self::builder().api_key(Nothing).build().map_err(Into::into)
    }
}

// ================================================================
// Tests
// ================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{EmbeddingsClient, Nothing};
    use crate::embeddings::EmbeddingModel as _;
    use crate::providers::openai::embedding::EncodingFormat;
    use crate::test_utils::RecordingHttpClient;

    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::llamafile::Client::new(Nothing).expect("Client::new() failed");
        let _client_from_builder = crate::providers::llamafile::Client::builder()
            .api_key(Nothing)
            .build()
            .expect("Client::builder() failed");
    }

    #[test]
    fn test_client_from_url() {
        let _client = crate::providers::llamafile::Client::from_url("http://localhost:8080");
    }

    #[test]
    fn test_build_uri_routes_through_v1() {
        let ext = LlamafileExt;
        assert_eq!(
            ext.build_uri(
                "http://localhost:8080",
                "/chat/completions",
                Transport::Http
            ),
            "http://localhost:8080/v1/chat/completions"
        );
        assert_eq!(
            ext.build_uri("http://localhost:8080/", "/embeddings", Transport::Http),
            "http://localhost:8080/v1/embeddings"
        );
        assert_eq!(
            ext.build_uri(
                "http://localhost:8080",
                LlamafileExt::VERIFY_PATH,
                Transport::Http
            ),
            "http://localhost:8080/v1/models"
        );
    }

    #[tokio::test]
    async fn embedding_model_preserves_v1_path_and_usage() {
        let response = r#"{
            "object": "list",
            "model": "LLaMA_CPP",
            "usage": { "prompt_tokens": 2, "total_tokens": 2 },
            "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2] }]
        }"#;
        let http_client = RecordingHttpClient::new(response);
        let client = Client::builder()
            .api_key(Nothing)
            .http_client(http_client.clone())
            .build()
            .expect("client should build");
        let model = client.embedding_model(LLAMA_CPP);

        let response = model
            .embed_texts_with_usage(["hello".to_string()])
            .await
            .expect("embedding request should succeed");

        assert_eq!(response.usage.total_tokens, 2);
        assert_eq!(
            http_client.requests()[0].uri,
            "http://localhost:8080/v1/embeddings"
        );
    }

    #[tokio::test]
    async fn embedding_model_rejects_base64_before_sending() {
        let http_client = RecordingHttpClient::new("{}");
        let client = Client::builder()
            .api_key(Nothing)
            .http_client(http_client.clone())
            .build()
            .expect("client should build");
        let model = client
            .embedding_model(LLAMA_CPP)
            .encoding_format(EncodingFormat::Base64);

        let error = model
            .embed_texts(["hello".to_string()])
            .await
            .expect_err("numeric response parser should reject base64");

        assert!(matches!(
            error,
            crate::embeddings::EmbeddingError::UnsupportedResponseEncoding {
                provider: "llamafile",
                encoding_format: "base64"
            }
        ));
        assert!(http_client.requests().is_empty());
    }
}

pub mod functions {
    //! Llamafile chat completions as config + pure functions.
    //!
    //! The data-oriented face of the Llamafile provider, mirroring
    //! [`crate::providers::openai::functions`]: a serde [`Config`], a
    //! [`DESCRIPTOR`] capability sheet, and pure
    //! [`build_request`]/[`parse_response`] free functions plus the async
    //! [`complete`]/[`open_stream`] wrappers over
    //! [`HttpRuntime`](crate::http_runtime::HttpRuntime). The request/parse
    //! mechanics are shared with the other OpenAI-compatible providers via
    //! `openai::functions`; this module instantiates them with
    //! [`LlamafileExt`](super::LlamafileExt) so Llamafile's paths, hooks, and
    //! provider name apply.

    use serde::{Deserialize, Serialize};

    use super::LlamafileExt as Ext;
    use crate::completion::{self, CompletionError, CompletionRequest};
    use crate::http_runtime::HttpRuntime;
    use crate::providers::descriptor::{ApiKeyLocation, ProviderDescriptor};
    use crate::providers::openai::functions as openai_functions;

    /// Default Llamafile API base URL.
    pub const DEFAULT_BASE_URL: &str = "http://localhost:8080";

    /// Llamafile's capability sheet.
    pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
        name: "llamafile",
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        emits_complete_single_chunk_tool_calls: true,
        composes_native_output_with_tools: true,
        max_embedding_documents: Some(1024),
    };

    /// Plain-data Llamafile provider configuration.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[non_exhaustive]
    pub struct Config {
        /// API base URL (defaults to [`DEFAULT_BASE_URL`]).
        pub base_url: String,
        /// Credential location.
        pub api_key: ApiKeyLocation,
        /// Model identifier requests are built for.
        pub model: String,
        /// Extra headers attached to every request.
        pub extra_headers: Vec<(String, String)>,
    }

    impl Config {
        /// Config for `model` with no credential (llamafile serves an unauthenticated local endpoint).
        pub fn new(model: impl Into<String>) -> Self {
            Self {
                base_url: DEFAULT_BASE_URL.to_string(),
                api_key: ApiKeyLocation::None,
                model: model.into(),
                extra_headers: Vec::new(),
            }
        }

        /// Config for `model` with an explicit API key.
        pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
            self.api_key = ApiKeyLocation::Inline(key.into());
            self
        }

        /// Override the API base URL.
        pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
            self.base_url = base_url.into();
            self
        }
    }

    /// Build the serialized chat-completions request body for `request`. Pure.
    pub fn build_request_body(
        cfg: &Config,
        request: &CompletionRequest,
        stream: bool,
    ) -> Result<Vec<u8>, CompletionError> {
        openai_functions::compatible_request_body(&Ext, &cfg.model, request, stream)
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
        openai_functions::compatible_request(
            &Ext,
            &cfg.base_url,
            &cfg.api_key,
            &cfg.extra_headers,
            &cfg.model,
            request,
            stream,
        )
    }

    /// Parse a chat-completions response body into the normalized
    /// [`completion::CompletionResponse`]. Pure.
    pub fn parse_response(
        status: http::StatusCode,
        body: &str,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        openai_functions::compatible_parse_response::<Ext>(status, body)
    }

    /// Open a streaming completion for `request`.
    pub async fn open_stream(
        cfg: &Config,
        rt: &HttpRuntime,
        request: CompletionRequest,
    ) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError> {
        let req = build_request(cfg, &request, true)?;
        openai_functions::compatible_open_stream(Ext, rt, req).await
    }

    /// Send `request` to Llamafile and return the normalized response.
    pub async fn complete(
        cfg: &Config,
        rt: &HttpRuntime,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        let req = build_request(cfg, &request, false)?;
        let (status, body) = rt.send(req).await?;
        parse_response(status, &body)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::OneOrMany;

        fn sample_request() -> CompletionRequest {
            CompletionRequest {
                model: None,
                preamble: None,
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
            let cfg = Config::new("test-model").with_api_key("secret");
            let req = build_request(&cfg, &sample_request(), false).expect("build");
            assert_eq!(req.uri(), "http://localhost:8080/chat/completions");
            let value: serde_json::Value = serde_json::from_slice(req.body()).expect("json");
            assert_eq!(value["model"], "test-model");
        }

        #[test]
        fn parse_response_normalizes() {
            let body = serde_json::json!({
                "id": "chatcmpl-1",
                "model": "test-model",
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
            assert_eq!(response.provider, "llamafile");
            assert_eq!(response.usage.input_tokens, 3);
            assert_eq!(response.usage.total_tokens, 5);
        }
    }
}
