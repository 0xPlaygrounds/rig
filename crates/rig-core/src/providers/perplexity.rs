//! Perplexity API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig_core::{client::CompletionClient, providers::perplexity};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = perplexity::Client::new("YOUR_API_KEY")?;
//!
//! let sonar = client.completion_model(perplexity::SONAR);
//! # Ok(())
//! # }
//! ```
use crate::client::BearerAuth;
use crate::providers::descriptor::ChatCompletionsDialect;
use crate::providers::descriptor::ProviderDescriptor;
use crate::providers::openai;
use crate::{
    client::{
        self, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder, ProviderClient,
    },
    http_client::{self, HttpClientExt},
};

// ================================================================
// Main Perplexity Client
// ================================================================
const PERPLEXITY_API_BASE_URL: &str = "https://api.perplexity.ai";

#[derive(Debug, Default, Clone, Copy)]
pub struct PerplexityExt;

#[derive(Debug, Default, Clone, Copy)]
pub struct PerplexityBuilder;

type PerplexityApiKey = BearerAuth;

impl Provider for PerplexityExt {
    type Builder = PerplexityBuilder;

    // There is currently no way to verify a perplexity api key without consuming tokens
    const VERIFY_PATH: &'static str = "";
}

impl openai::completion::OpenAICompatibleProvider for PerplexityExt {
    const DESCRIPTOR: ProviderDescriptor = functions::DESCRIPTOR;
    const STREAM_DIALECT: ChatCompletionsDialect = functions::STREAM_DIALECT;

    type Response = openai::CompletionResponse;

    fn completion_path(&self, model: &str) -> String {
        functions::completion_path(model)
    }

    fn build_body(
        &self,
        model: &str,
        request: &crate::completion::CompletionRequest,
        options: crate::providers::openai::completion::CompletionModelOptions,
        stream: bool,
    ) -> Result<Vec<u8>, crate::completion::CompletionError> {
        functions::build_body(model, request, options, stream)
    }
}

impl<H> Capabilities<H> for PerplexityExt {
    type Completion = Capable<CompletionModel<H>>;
    type Transcription = Nothing;
    type Embeddings = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;

    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
    type Rerank = Nothing;
}

impl DebugExt for PerplexityExt {}

impl ProviderBuilder for PerplexityBuilder {
    type Extension<H>
        = PerplexityExt
    where
        H: HttpClientExt;
    type ApiKey = PerplexityApiKey;

    const BASE_URL: &'static str = PERPLEXITY_API_BASE_URL;

    fn build<H>(
        _builder: &crate::client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(PerplexityExt)
    }
}

pub type Client<H = reqwest::Client> = client::Client<PerplexityExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<PerplexityBuilder, PerplexityApiKey, H>;

/// Perplexity completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = reqwest::Client> =
    openai::completion::GenericCompletionModel<PerplexityExt, H>;

/// Raw completion payload, shared with the OpenAI Chat Completions path.
pub type CompletionResponse = openai::CompletionResponse;

impl ProviderClient for Client {
    type Input = String;
    type Error = crate::client::ProviderClientError;

    /// Create a new Perplexity client from the `PERPLEXITY_API_KEY` environment variable.
    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("PERPLEXITY_API_KEY")?;
        Self::new(&api_key).map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(&input).map_err(Into::into)
    }
}

// ================================================================
// Perplexity Completion API
// ================================================================

pub const SONAR_PRO: &str = "sonar_pro";
pub const SONAR: &str = "sonar";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OneOrMany;
    use crate::message::{AssistantContent, Message, UserContent};
    use crate::providers::openai::completion::CompletionModelOptions;

    /// Build the request body Perplexity would send for `chat_history`.
    fn body_for_history(chat_history: OneOrMany<Message>) -> serde_json::Value {
        body_for(crate::completion::CompletionRequest {
            model: None,
            preamble: None,
            chat_history,
            documents: vec![],
            max_tokens: None,
            temperature: None,
            tools: vec![],
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        })
    }

    /// Build the request body Perplexity would send for `request`.
    fn body_for(request: crate::completion::CompletionRequest) -> serde_json::Value {
        let bytes =
            functions::build_body(SONAR, &request, CompletionModelOptions::default(), false)
                .expect("body should build");
        serde_json::from_slice(&bytes).expect("body should be json")
    }

    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::perplexity::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::perplexity::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }

    #[test]
    fn perplexity_body_flattens_text_only_content_arrays() {
        let body = body_for_history(
            OneOrMany::many([
                Message::User {
                    content: OneOrMany::many([
                        UserContent::text("First."),
                        UserContent::text("Second."),
                    ])
                    .expect("multi-part user message"),
                },
                Message::User {
                    content: OneOrMany::many([
                        UserContent::text("Look:"),
                        UserContent::image_url("https://example.com/i.png", None, None),
                    ])
                    .expect("mixed user message"),
                },
            ])
            .expect("history"),
        );

        assert_eq!(body["messages"][0]["content"], "First.\nSecond.");
        // Mixed content stays an array for the API's multimodal handling.
        assert!(body["messages"][1]["content"].is_array());
    }

    #[test]
    fn perplexity_drops_tool_choice_instead_of_erroring() {
        // Multi-name Specific errors on tool-supporting providers; with
        // `supports_tools: false` it must be dropped before that validation.
        let body = body_for(crate::completion::CompletionRequest {
            tools: vec![crate::completion::ToolDefinition {
                name: "lookup".to_string(),
                description: String::new(),
                parameters: serde_json::json!({}),
            }],
            tool_choice: Some(crate::message::ToolChoice::Specific {
                function_names: vec!["a".to_string(), "b".to_string()],
            }),
            ..crate::completion::CompletionRequest::from_prompt("Hello!")
        });

        assert!(
            body.get("tools")
                .is_none_or(|tools| tools.as_array().is_none_or(|tools| tools.is_empty()))
        );
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn perplexity_body_strips_tool_history_and_preserves_alternation() {
        let body = body_for_history(
            OneOrMany::many([
                Message::user("Look it up."),
                Message::Assistant {
                    id: None,
                    content: OneOrMany::one(AssistantContent::tool_call(
                        "call_1",
                        "lookup",
                        serde_json::json!({}),
                    )),
                },
                Message::tool_result("call_1", "result"),
                Message::Assistant {
                    id: None,
                    content: OneOrMany::many([
                        AssistantContent::reasoning("hmm"),
                        AssistantContent::text("It is crimson."),
                    ])
                    .expect("assistant content"),
                },
                Message::user("Thanks!"),
            ])
            .expect("history"),
        );

        let messages = body["messages"].as_array().expect("messages array");
        let roles = messages
            .iter()
            .map(|m| m["role"].as_str().unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(roles, ["user", "assistant", "user"]);
        assert_eq!(messages[1]["content"], "It is crimson.");
        assert!(messages[1].get("reasoning_content").is_none());
        assert!(messages[1].get("tool_calls").is_none());
    }

    #[test]
    fn perplexity_body_drops_tools() {
        let body = body_for(crate::completion::CompletionRequest {
            tools: vec![crate::completion::ToolDefinition {
                name: "lookup".to_string(),
                description: "Lookup".to_string(),
                parameters: serde_json::json!({"type":"object","properties":{},"required":[]}),
            }],
            tool_choice: Some(crate::message::ToolChoice::Required),
            ..crate::completion::CompletionRequest::from_prompt("What's new today?")
        });

        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }
}

pub mod functions {
    //! Perplexity chat completions as config + pure functions.
    //!
    //! The data-oriented face of the Perplexity provider, mirroring
    //! [`crate::providers::openai::functions`]: a serde [`Config`], a
    //! [`DESCRIPTOR`] capability sheet, and pure
    //! [`build_request`]/[`parse_response`] free functions plus the async
    //! [`complete`]/[`open_stream`] wrappers over
    //! [`HttpRuntime`](crate::http_runtime::HttpRuntime). The request/parse
    //! mechanics are shared with the other OpenAI-compatible providers via
    //! `openai::functions`'s stage helpers; this module owns Perplexity's own
    //! dialect steps, paths, and provider name.

    use serde::{Deserialize, Serialize};

    use crate::completion::{self, CompletionError, CompletionRequest};
    use crate::http_runtime::HttpRuntime;
    use crate::providers::descriptor::ChatCompletionsDialect;
    use crate::providers::descriptor::{
        ApiKeyLocation, ConfigError, ProviderDescriptor, required_env_var,
    };
    use crate::providers::openai::completion::CompletionModelOptions;
    use crate::providers::openai::functions as openai_functions;

    /// Default Perplexity API base URL.
    pub const DEFAULT_BASE_URL: &str = "https://api.perplexity.ai";

    /// Perplexity's Chat Completions streaming dialect.
    pub(crate) const STREAM_DIALECT: ChatCompletionsDialect =
        ChatCompletionsDialect::from_descriptor(&DESCRIPTOR);

    /// Perplexity's capability sheet.
    pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
        name: "perplexity",
        supports_tools: false,
        supports_response_format: false,
        stream_include_usage: false,
        emits_complete_single_chunk_tool_calls: false,
        composes_native_output_with_tools: false,
        max_embedding_documents: None,
    };

    /// Plain-data Perplexity provider configuration.
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
        /// Config for `model` reading `PERPLEXITY_API_KEY` from the environment.
        pub fn new(model: impl Into<String>) -> Self {
            Self {
                base_url: DEFAULT_BASE_URL.to_string(),
                api_key: ApiKeyLocation::Env("PERPLEXITY_API_KEY".to_string()),
                model: model.into(),
                extra_headers: Vec::new(),
            }
        }

        /// Config for `model` built from the process environment.
        ///
        /// Reads `PERPLEXITY_API_KEY` (required) — the same variable the deleted
        /// `perplexity::Client::from_env` read. The credential is validated eagerly
        /// but stored as [`ApiKeyLocation::Env`], so the secret is read at
        /// request time rather than held inside the config.
        ///
        /// # Errors
        /// [`ConfigError`] when a required variable is missing or invalid.
        pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
            let cfg = Self::new(model);
            required_env_var("PERPLEXITY_API_KEY")?;
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

    /// The chat-completions request path for `model`.
    pub(crate) fn completion_path(_model: &str) -> String {
        "/chat/completions".to_string()
    }

    /// Perplexity's straight-line chat-completions body assembly.
    ///
    /// Perplexity historically only accepted plain `{role, content: String}`
    /// messages, and its API accepts only system/user/assistant roles with
    /// strict user/assistant alternation, so the serialized body has its
    /// tool-exchange remnants stripped and its text-only content-part arrays
    /// flattened. Tools and structured output are dropped during the typed
    /// conversion (see [`DESCRIPTOR`]), and the streaming body carries a bare
    /// `stream: true` with no `stream_options`.
    pub(crate) fn build_body(
        model: &str,
        request: &CompletionRequest,
        options: CompletionModelOptions,
        stream: bool,
    ) -> Result<Vec<u8>, CompletionError> {
        let typed =
            openai_functions::compatible_typed_request(model, request, &DESCRIPTOR, options)?;
        let mut body = openai_functions::compatible_body_value(&typed, &DESCRIPTOR, stream)?;
        // Perplexity historically only accepted plain `{role, content: String}`
        // messages, and its API accepts only system/user/assistant roles
        // with strict user/assistant alternation. Strip tool-exchange
        // remnants from shared histories and flatten text-only content-part
        // arrays; arrays with non-text parts (e.g. images on sonar models)
        // are left for the API's multimodal handling.
        if let Some(messages) = body
            .get_mut("messages")
            .and_then(serde_json::Value::as_array_mut)
        {
            crate::providers::openai::completion::sanitize_plain_text_history(
                messages,
                Some(("\n", true)),
                false,
                true,
            );
        }

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

    /// Send `request` to Perplexity and return the normalized response.
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
            assert_eq!(req.uri(), "https://api.perplexity.ai/chat/completions");
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
            assert_eq!(response.provider, "perplexity");
            assert_eq!(response.usage.input_tokens, 3);
            assert_eq!(response.usage.total_tokens, 5);
        }
    }
}
