//! Perplexity API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig_core::providers::perplexity;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = perplexity::functions::Config::from_env(perplexity::SONAR)?;
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//! let request = rig_core::completion::CompletionRequest::from_prompt("Hello!");
//! let response = perplexity::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```
use crate::providers::openai;

/// Raw completion payload, shared with the OpenAI Chat Completions path.
pub type CompletionResponse = openai::CompletionResponse;

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
        let body = body_for(
            crate::completion::CompletionRequest::builder("Hello!")
                .tools(vec![crate::completion::ToolDefinition {
                    name: "lookup".to_string(),
                    description: String::new(),
                    parameters: serde_json::json!({}),
                }])
                .tool_choice(crate::message::ToolChoice::Specific {
                    function_names: vec!["a".to_string(), "b".to_string()],
                })
                .build(),
        );

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
        let body = body_for(
            crate::completion::CompletionRequest::builder("What's new today?")
                .tools(vec![crate::completion::ToolDefinition {
                    name: "lookup".to_string(),
                    description: "Lookup".to_string(),
                    parameters: serde_json::json!({"type":"object","properties":{},"required":[]}),
                }])
                .tool_choice(crate::message::ToolChoice::Required)
                .build(),
        );

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
    //! [`HttpRuntime`]. The request/parse
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
        verify_path: None,
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
