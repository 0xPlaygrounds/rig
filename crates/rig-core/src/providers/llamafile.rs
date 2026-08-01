//! Llamafile API client and Rig integration
//!
//! [Llamafile](https://github.com/Mozilla-Ocho/llamafile) is a Mozilla Builders project
//! that distributes LLMs as single-file executables. When started, it exposes an
//! OpenAI-compatible API at `http://localhost:8080/v1`.
//!
//! # Example
//! ```no_run
//! use rig_core::providers::llamafile;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Defaults to `http://localhost:8080/v1`; override with
//! // `Config::with_base_url`, or read `LLAMAFILE_API_BASE_URL` via
//! // `Config::from_env`.
//! let cfg = llamafile::functions::Config::new(llamafile::LLAMA_CPP);
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//!
//! // Send a completion request with a preamble.
//! let request = rig_core::completion::CompletionRequest::builder("Hello!")
//!     .preamble("You are a helpful assistant.")
//!     .build();
//! let response = llamafile::functions::complete(&cfg, &rt, request).await?;
//! println!("{:?}", response.choice);
//! # Ok(())
//! # }
//! ```

/// The default model identifier reported by llamafile.
pub const LLAMA_CPP: &str = "LLaMA_CPP";

crate::providers::client::define_http_client! {
    config = functions::Config,
    default_base_url = functions::DEFAULT_BASE_URL,
    api_key_required = false,
}
crate::providers::client::impl_http_embedding_config_factory!(Client, functions::EmbeddingConfig);

impl Client {
    /// Build an unauthenticated client for a llamafile server URL.
    pub fn from_url(
        base_url: impl Into<String>,
    ) -> Result<Self, crate::providers::ClientBuildError> {
        let base_url = base_url.into();
        let base_url = base_url.trim_end_matches('/');
        let api_base_url = if base_url.ends_with("/v1") {
            base_url.to_string()
        } else {
            format!("{base_url}/v1")
        };
        Self::builder().base_url(api_base_url).build()
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
    //! [`HttpRuntime`]. The request/parse
    //! mechanics are shared with the other OpenAI-compatible providers via
    //! `openai::functions`'s stage helpers; this module owns Llamafile's own
    //! paths, dialect data, and provider name.

    use serde::{Deserialize, Serialize};

    use crate::completion::{self, CompletionError, CompletionRequest};
    use crate::http_runtime::HttpRuntime;
    use crate::providers::descriptor::ChatCompletionsDialect;
    use crate::providers::descriptor::{
        ApiKeyLocation, ConfigError, ProviderDescriptor, required_env_var,
    };
    use crate::providers::openai::completion::CompletionModelOptions;
    use crate::providers::openai::functions as openai_functions;

    /// Default Llamafile API base URL.
    ///
    /// Carries the `/v1` suffix: llamafile serves the OpenAI-compatible API
    /// under `/v1`, and this path joins `base_url` with the endpoint path
    /// verbatim. (The deleted classic client stored a bare host and appended
    /// `/v1` inside its own `build_uri`; the resulting wire URL is the same.)
    pub const DEFAULT_BASE_URL: &str = "http://localhost:8080/v1";

    /// Llamafile's Chat Completions streaming dialect.
    pub(crate) const STREAM_DIALECT: ChatCompletionsDialect =
        ChatCompletionsDialect::from_descriptor(&DESCRIPTOR);

    /// Llamafile's capability sheet.
    pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
        name: "llamafile",
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        emits_complete_single_chunk_tool_calls: true,
        composes_native_output_with_tools: true,
        max_embedding_documents: Some(1024),
        verify_path: Some("/models"),
    };

    /// Plain-data Llamafile provider configuration.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[non_exhaustive]
    pub struct Config {
        /// Reusable HTTP connection data.
        #[serde(flatten)]
        pub connection: crate::providers::HttpConnectionConfig,
        /// Model identifier requests are built for.
        pub model: String,
    }

    crate::providers::client::impl_http_connection_config!(Config);

    impl Config {
        /// Config for `model` with no credential (llamafile serves an unauthenticated local endpoint).
        pub fn new(model: impl Into<String>) -> Self {
            Self {
                connection: crate::providers::HttpConnectionConfig::new(
                    DEFAULT_BASE_URL.to_string(),
                    ApiKeyLocation::None,
                ),
                model: model.into(),
            }
        }

        /// Config for `model` built from the process environment.
        ///
        /// Reads `LLAMAFILE_API_BASE_URL` (**required**) — the same variable the
        /// deleted `llamafile::Client::from_env` read, which handed it straight to
        /// `Client::from_url`. There is no credential: llamafile serves an
        /// unauthenticated local endpoint, so `api_key` stays
        /// [`ApiKeyLocation::None`].
        ///
        /// The env value is a **bare host URL without the `/v1` suffix** (e.g.
        /// `http://localhost:8080`), exactly as the classic client expected: the
        /// classic path appended `/v1` itself in its own `build_uri`. The
        /// functions path builds URLs from `base_url` verbatim, so that same
        /// `/v1` is appended here to keep the wire URL identical to the classic
        /// client's.
        ///
        /// # Errors
        /// [`ConfigError`] when a required variable is missing or invalid.
        pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
            let mut cfg = Self::new(model);
            let base_url = required_env_var("LLAMAFILE_API_BASE_URL")?;
            cfg.base_url = format!("{}/v1", base_url.trim_end_matches('/'));
            Ok(cfg)
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
        build_body(
            &cfg.model,
            request,
            CompletionModelOptions::default(),
            stream,
        )
    }

    /// The chat-completions request path for `model`.
    pub(crate) fn completion_path(_model: &str) -> String {
        "/chat/completions".to_string()
    }

    /// Llamafile's straight-line chat-completions body assembly.
    ///
    /// No wire-level request quirks: llamafile serves the reference OpenAI
    /// dialect, so the body is the shared typed conversion serialized as-is. Its
    /// one dialect difference is on the streaming side — llama.cpp-based servers
    /// can emit a whole tool call in a single chunk
    /// (`emits_complete_single_chunk_tool_calls` on [`DESCRIPTOR`]).
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

    /// Build the complete HTTP request (URL, headers, body) for `request`.
    ///
    /// Pure except for credential resolution (`ApiKeyLocation::Env` reads the
    /// environment).
    pub fn build_request(
        cfg: &Config,
        request: &CompletionRequest,
        stream: bool,
    ) -> Result<http::Request<Vec<u8>>, CompletionError> {
        openai_functions::compatible_http_request(
            &cfg.base_url,
            &completion_path(&cfg.model),
            &cfg.api_key,
            &cfg.extra_headers,
            build_request_body(cfg, request, stream)?,
        )
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

    // ================================================================
    // Embeddings
    // ================================================================

    /// Plain-data Llamafile embeddings configuration.
    ///
    /// Llamafile serves OpenAI's `/embeddings` shape, so the request body and
    /// response parsing are OpenAI's; only the provider name (used in errors)
    /// and the unauthenticated default credential differ.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[non_exhaustive]
    pub struct EmbeddingConfig {
        /// Reusable HTTP connection data.
        #[serde(flatten)]
        pub connection: crate::providers::HttpConnectionConfig,
        /// Embedding model identifier requests are built for.
        pub model: String,
        /// Requested embedding dimensions, sent verbatim as `dimensions`.
        pub dimensions: Option<usize>,
    }

    crate::providers::client::impl_http_connection_config!(EmbeddingConfig);

    impl EmbeddingConfig {
        /// Embedding config for `model` against the default local endpoint.
        pub fn new(model: impl Into<String>) -> Self {
            Self {
                connection: crate::providers::HttpConnectionConfig::new(
                    DEFAULT_BASE_URL.to_string(),
                    ApiKeyLocation::None,
                ),
                model: model.into(),
                dimensions: None,
            }
        }

        /// Embedding config for `model` built from the process environment.
        ///
        /// Reads `LLAMAFILE_API_BASE_URL` (**required**), applying the same
        /// `/v1` suffix rule as [`Config::from_env`].
        ///
        /// # Errors
        /// [`ConfigError`] when the variable is missing or invalid.
        pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
            let mut cfg = Self::new(model);
            let base_url = required_env_var("LLAMAFILE_API_BASE_URL")?;
            cfg.base_url = format!("{}/v1", base_url.trim_end_matches('/'));
            Ok(cfg)
        }

        /// Override the API base URL.
        pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
            self.base_url = base_url.into();
            self
        }

        /// Request `dimensions`-sized embeddings.
        pub fn with_dimensions(mut self, dimensions: usize) -> Self {
            self.dimensions = Some(dimensions);
            self
        }
    }

    /// Build the complete HTTP embeddings request for one chunk of `texts`.
    pub fn build_embedding_request(
        cfg: &EmbeddingConfig,
        texts: &[String],
    ) -> Result<http::Request<Vec<u8>>, crate::embeddings::EmbeddingError> {
        use crate::embeddings::EmbeddingError;
        use http::header::{AUTHORIZATION, CONTENT_TYPE};

        let body = crate::providers::openai::embedding::build_embedding_body(
            &cfg.model,
            texts,
            cfg.dimensions
                .map(crate::providers::openai::embedding::EmbeddingDimensions::Dimensions),
            None,
            None,
        )?;
        let url = format!("{}/embeddings", cfg.base_url.trim_end_matches('/'));
        let mut builder = http::Request::post(url).header(CONTENT_TYPE, "application/json");
        if let Some(key) = cfg
            .api_key
            .resolve()
            .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?
        {
            builder = builder.header(AUTHORIZATION, format!("Bearer {key}"));
        }
        for (name, value) in &cfg.extra_headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        builder
            .body(body)
            .map_err(|e| EmbeddingError::ProviderError(e.to_string()))
    }

    /// Parse an embeddings response body. Pure.
    pub fn parse_embedding_response(
        status: http::StatusCode,
        body: &str,
        documents: Vec<String>,
    ) -> Result<crate::embeddings::EmbeddingResponse, crate::embeddings::EmbeddingError> {
        crate::providers::openai::embedding::parse_embedding_response(
            status,
            body,
            documents,
            DESCRIPTOR.name,
            true,
        )
    }

    /// Embed `texts`, chunked to [`DESCRIPTOR`]'s document limit.
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

    /// Embed pre-grouped `texts`, preserving the grouping in the result.
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
    /// Verify that `cfg`'s credential is accepted by the provider.
    ///
    /// The data-oriented replacement for the deleted `VerifyClient::verify`: the
    /// endpoint is [`DESCRIPTOR`]'s `verify_path` (`/models`, the value the
    /// deleted `Provider::VERIFY_PATH` carried) and the status mapping is the
    /// classic one — see [`crate::providers::verify`].
    ///
    /// # Errors
    /// [`VerifyError`](crate::providers::verify::VerifyError): invalid
    /// authentication on `401`/`403`, otherwise the preserved provider response
    /// or a transport failure.
    pub async fn verify(
        cfg: &Config,
        rt: &HttpRuntime,
    ) -> Result<(), crate::providers::verify::VerifyError> {
        crate::providers::verify::verify_bearer(
            &DESCRIPTOR,
            &cfg.base_url,
            &cfg.api_key,
            &cfg.extra_headers,
            rt,
        )
        .await
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::OneOrMany;

        /// Ported from the deleted classic `embedding_model_preserves_v1_path_and_usage`:
        /// llamafile embeddings must hit `/v1/embeddings` and surface usage.
        #[tokio::test]
        async fn embed_preserves_v1_path_and_usage() {
            let response = r#"{
                "object": "list",
                "model": "LLaMA_CPP",
                "usage": { "prompt_tokens": 2, "total_tokens": 2 },
                "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2] }]
            }"#;
            let http_client = crate::test_utils::RecordingHttpClient::new(response);
            let rt = HttpRuntime::recording(http_client.clone());
            let cfg = EmbeddingConfig::new(super::super::LLAMA_CPP);

            let response = embed(&cfg, &rt, vec!["hello".to_string()])
                .await
                .expect("embedding request should succeed");

            assert_eq!(response.usage.total_tokens, 2);
            assert_eq!(
                http_client.requests()[0].uri,
                "http://localhost:8080/v1/embeddings"
            );
        }

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
            let cfg = Config::new("test-model").with_api_key("secret");
            let req = build_request(&cfg, &sample_request(), false).expect("build");
            assert_eq!(req.uri(), "http://localhost:8080/v1/chat/completions");
            let value: serde_json::Value = serde_json::from_slice(req.body()).expect("json");
            assert_eq!(value["model"], "test-model");
        }

        // Retargets the deleted classic `LlamafileExt::build_uri` test: the
        // `/v1` prefix llamafile serves under must land in the wire URL, and a
        // trailing slash on the configured base URL must not double up.
        #[test]
        fn build_request_routes_through_v1() {
            assert_eq!(DEFAULT_BASE_URL, "http://localhost:8080/v1");

            let cfg = Config::new("test-model").with_base_url("http://localhost:8080/v1/");
            let req = build_request(&cfg, &sample_request(), false).expect("build");
            assert_eq!(req.uri(), "http://localhost:8080/v1/chat/completions");
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
