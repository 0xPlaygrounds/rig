//! Mira API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig_core::providers::mira;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = mira::functions::Config::from_env("deepseek-r1")?;
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//! let request = rig_core::completion::CompletionRequest::from_prompt("Hello!");
//! let response = mira::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```
use crate::{
    OneOrMany,
    completion::{self, CompletionError},
    message::{self, AssistantContent, Message, UserContent},
};
use serde::{Deserialize, Serialize};
use tracing::{self};

/// Path of Mira's model-listing endpoint.
pub(crate) const LIST_MODELS_PATH: &str = "/v1/models";

/// Mira's `GET /v1/models` payload.
#[derive(Debug, Deserialize)]
pub struct ListModelsResponse {
    /// One entry per available model.
    pub data: Vec<ListModelEntry>,
}

/// One row of [`ListModelsResponse`].
#[derive(Debug, Deserialize)]
pub struct ListModelEntry {
    /// Model identifier.
    pub id: String,
}

impl From<ListModelEntry> for crate::model::Model {
    fn from(entry: ListModelEntry) -> Self {
        Self::from_id(entry.id)
    }
}

/// Parse a model-listing response body. Pure.
pub(crate) fn parse_list_models_response(
    status: http::StatusCode,
    body: &[u8],
) -> Result<crate::model::ModelList, crate::model::ModelListingError> {
    use crate::model::{Model, ModelList, ModelListingError};

    if !status.is_success() {
        return Err(ModelListingError::api_error_with_context(
            "Mira",
            LIST_MODELS_PATH,
            status.as_u16(),
            body,
        ));
    }
    let api_resp: ListModelsResponse = serde_json::from_slice(body).map_err(|error| {
        ModelListingError::parse_error_with_context("Mira", LIST_MODELS_PATH, &error, body)
    })?;
    let models = api_resp.data.into_iter().map(Model::from).collect();
    Ok(ModelList::new(models))
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct RawMessage {
    pub role: String,
    pub content: String,
}

impl TryFrom<RawMessage> for message::Message {
    type Error = CompletionError;

    fn try_from(raw: RawMessage) -> Result<Self, Self::Error> {
        match raw.role.as_str() {
            "system" => Ok(message::Message::System {
                content: raw.content,
            }),
            "user" => Ok(message::Message::User {
                content: OneOrMany::one(UserContent::Text(message::Text::new(raw.content))),
            }),
            "assistant" => Ok(message::Message::Assistant {
                id: None,
                content: OneOrMany::one(AssistantContent::Text(message::Text::new(raw.content))),
            }),
            _ => Err(CompletionError::ResponseError(format!(
                "Unsupported message role: {}",
                raw.role
            ))),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum CompletionResponse {
    Structured {
        id: String,
        object: String,
        created: u64,
        model: String,
        choices: Vec<ChatChoice>,
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
    },
    Simple(String),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatChoice {
    pub message: RawMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub index: Option<usize>,
}

impl crate::telemetry::ProviderResponseExt for CompletionResponse {
    type OutputMessage = ChatChoice;
    type Usage = Usage;

    fn get_response_id(&self) -> Option<String> {
        match self {
            Self::Structured { id, .. } => Some(id.clone()),
            Self::Simple(_) => None,
        }
    }

    fn get_response_model_name(&self) -> Option<String> {
        match self {
            Self::Structured { model, .. } => Some(model.clone()),
            Self::Simple(_) => None,
        }
    }

    fn get_output_messages(&self) -> Vec<Self::OutputMessage> {
        match self {
            Self::Structured { choices, .. } => choices
                .iter()
                .map(|choice| ChatChoice {
                    message: choice.message.clone(),
                    finish_reason: choice.finish_reason.clone(),
                    index: choice.index,
                })
                .collect(),
            Self::Simple(_) => Vec::new(),
        }
    }

    fn get_text_response(&self) -> Option<String> {
        match self {
            Self::Structured { choices, .. } => choices
                .iter()
                .find(|choice| choice.message.role == "assistant")
                .map(|choice| choice.message.content.clone()),
            Self::Simple(text) => Some(text.clone()),
        }
    }

    fn get_usage(&self) -> Option<Self::Usage> {
        match self {
            Self::Structured { usage, .. } => usage.clone(),
            Self::Simple(_) => None,
        }
    }
}

impl From<Usage> for crate::completion::Usage {
    fn from(value: Usage) -> crate::completion::Usage {
        let mut usage = crate::completion::Usage::new();
        usage.input_tokens = value.prompt_tokens as u64;
        usage.output_tokens = value.total_tokens.saturating_sub(value.prompt_tokens) as u64;
        usage.total_tokens = value.total_tokens as u64;
        usage
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let (content, usage, model, finish_reason) = match &response {
            CompletionResponse::Structured {
                choices,
                usage,
                model,
                ..
            } => {
                let choice = choices.first().ok_or_else(|| {
                    CompletionError::ResponseError("Response contained no choices".to_owned())
                })?;

                let usage = usage
                    .as_ref()
                    .map(|usage| completion::Usage::from(usage.clone()))
                    .unwrap_or_default();

                let finish_reason = choice
                    .finish_reason
                    .as_deref()
                    .map(crate::providers::openai::completion::map_finish_reason);

                // Convert RawMessage to message::Message
                let message = message::Message::try_from(choice.message.clone())?;

                let content = match message {
                    Message::Assistant { content, .. } => {
                        if content.is_empty() {
                            return Err(CompletionError::ResponseError(
                                "Response contained empty content".to_owned(),
                            ));
                        }

                        // Log warning for unsupported content types
                        for c in content.iter() {
                            if !matches!(c, AssistantContent::Text(_)) {
                                tracing::warn!(target: "rig",
                                    "Unsupported content type encountered: {:?}. The Mira provider currently only supports text content", c
                                );
                            }
                        }

                        content.iter().map(|c| {
                            match c {
                                AssistantContent::Text(text) => Ok(completion::AssistantContent::text(&text.text)),
                                other => Err(CompletionError::ResponseError(
                                    format!("Unsupported content type: {other:?}. The Mira provider currently only supports text content")
                                ))
                            }
                        }).collect::<Result<Vec<_>, _>>()?
                    }
                    Message::User { .. } => {
                        tracing::warn!(target: "rig", "Received user message in response where assistant message was expected");
                        return Err(CompletionError::ResponseError(
                            "Received user message in response where assistant message was expected".to_owned()
                        ));
                    }
                    Message::System { .. } => {
                        tracing::warn!(target: "rig", "Received system message in response where assistant message was expected");
                        return Err(CompletionError::ResponseError(
                            "Received system message in response where assistant message was expected".to_owned(),
                        ));
                    }
                };

                (content, usage, Some(model.clone()), finish_reason)
            }
            CompletionResponse::Simple(text) => (
                vec![completion::AssistantContent::text(text)],
                completion::Usage::new(),
                None,
                None,
            ),
        };

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        let mut normalized =
            completion::CompletionResponse::new(choice, usage, functions::DESCRIPTOR.name);
        if let Some(model) = model {
            normalized = normalized.with_model(model);
        }
        normalized.finish_reason = finish_reason;
        Ok(normalized)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
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
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completion_response_conversion() {
        let mira_response = CompletionResponse::Structured {
            id: "resp_123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "deepseek-r1".to_string(),
            choices: vec![ChatChoice {
                message: RawMessage {
                    role: "assistant".to_string(),
                    content: "Test response".to_string(),
                },
                finish_reason: Some("stop".to_string()),
                index: Some(0),
            }],
            usage: Some(Usage {
                prompt_tokens: 10,
                total_tokens: 20,
            }),
        };

        let completion_response: completion::CompletionResponse =
            mira_response.try_into().expect("conversion should succeed");

        assert_eq!(
            completion_response.choice.first(),
            completion::AssistantContent::text("Test response")
        );
    }
    // Proves a non-success HTTP response from `/v1/chat/completions` preserves
    // the provider's status + body through the `provider_response_*` helpers
    // (issue #1931).
    #[tokio::test]
    async fn completion_non_success_preserves_status_and_body() {
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"boom"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let rt = crate::http_runtime::HttpRuntime::recording(http_client);
        let cfg = functions::Config::new("deepseek-r1").with_api_key("test-key");
        let request = crate::completion::CompletionRequest::from_prompt("hello");

        let error = functions::complete(&cfg, &rt, request)
            .await
            .expect_err("should fail with non-success status");

        assert!(matches!(error, CompletionError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }
}

crate::providers::client::define_http_client! {
    config = functions::Config,
    default_base_url = functions::DEFAULT_BASE_URL,
    api_key_required = true,
}

pub mod functions {
    //! Mira chat completions as config + pure functions.
    //!
    //! The data-oriented face of the Mira provider, mirroring
    //! [`crate::providers::openai::functions`]: a serde [`Config`], a
    //! [`DESCRIPTOR`] capability sheet, and pure
    //! [`build_request`]/[`parse_response`] free functions plus the async
    //! [`complete`]/[`open_stream`] wrappers over
    //! [`HttpRuntime`]. The request/parse
    //! mechanics are shared with the other OpenAI-compatible providers via
    //! `openai::functions`'s stage helpers; this module owns Mira's own dialect
    //! steps, paths, and provider name.

    use serde::{Deserialize, Serialize};

    use crate::completion::{self, CompletionError, CompletionRequest};
    use crate::http_runtime::HttpRuntime;
    use crate::providers::descriptor::ChatCompletionsDialect;
    use crate::providers::descriptor::{
        ApiKeyLocation, ConfigError, ProviderDescriptor, required_env_var,
    };
    use crate::providers::openai::completion::CompletionModelOptions;
    use crate::providers::openai::functions as openai_functions;

    /// Default Mira API base URL.
    pub const DEFAULT_BASE_URL: &str = "https://api.mira.network";

    /// Mira's Chat Completions streaming dialect.
    pub(crate) const STREAM_DIALECT: ChatCompletionsDialect =
        ChatCompletionsDialect::from_descriptor(&DESCRIPTOR);

    /// Mira's capability sheet.
    pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
        name: "mira",
        supports_tools: false,
        supports_response_format: false,
        stream_include_usage: false,
        emits_complete_single_chunk_tool_calls: false,
        composes_native_output_with_tools: false,
        max_embedding_documents: None,
        verify_path: Some("/user-credits"),
    };

    /// Plain-data Mira provider configuration.
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
        /// Config for `model` reading `MIRA_API_KEY` from the environment.
        pub fn new(model: impl Into<String>) -> Self {
            Self {
                connection: crate::providers::HttpConnectionConfig::new(
                    DEFAULT_BASE_URL.to_string(),
                    ApiKeyLocation::Env("MIRA_API_KEY".to_string()),
                ),
                model: model.into(),
            }
        }

        /// Config for `model` built from the process environment.
        ///
        /// Reads `MIRA_API_KEY` (required) — the same variable the deleted
        /// `mira::Client::from_env` read. The credential is validated eagerly
        /// but stored as [`ApiKeyLocation::Env`], so the secret is read at
        /// request time rather than held inside the config.
        ///
        /// # Errors
        /// [`ConfigError`] when a required variable is missing or invalid.
        pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
            let cfg = Self::new(model);
            required_env_var("MIRA_API_KEY")?;
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
    ///
    /// The client base URL is the bare host; `list_models` builds its own v1 path.
    pub(crate) fn completion_path(_model: &str) -> String {
        "/v1/chat/completions".to_string()
    }

    /// Mira's straight-line chat-completions body assembly.
    ///
    /// Mira's gateway rejects pass-through `additional_params` and unknown
    /// parameters like `stream_options`, and only understands plain
    /// `{role, content}` string messages — so tool-exchange remnants and
    /// message names are stripped from the serialized body and content-part
    /// arrays are flattened. Tools and structured output are dropped during the
    /// typed conversion (see [`DESCRIPTOR`]).
    pub(crate) fn build_body(
        model: &str,
        request: &CompletionRequest,
        options: CompletionModelOptions,
        stream: bool,
    ) -> Result<Vec<u8>, CompletionError> {
        let mut typed =
            openai_functions::compatible_typed_request(model, request, &DESCRIPTOR, options)?;
        // Mira's gateway rejects pass-through parameters (tools are dropped
        // via `supports_tools: false` during conversion).
        if typed.additional_params.take().is_some() {
            tracing::warn!("Additional parameters are not supported by Mira and will be ignored");
        }

        let mut body = openai_functions::compatible_body_value(&typed, &DESCRIPTOR, stream)?;
        if let Some(map) = body.as_object_mut() {
            // Mira only understands plain `{role, content}` string messages;
            // strip tool-exchange remnants and message names, and flatten
            // content-part arrays.
            if let Some(messages) = map
                .get_mut("messages")
                .and_then(serde_json::Value::as_array_mut)
            {
                crate::providers::openai::completion::sanitize_plain_text_history(
                    messages,
                    Some(("\n", false)),
                    true,
                    false,
                );
            }
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
        openai_functions::compatible_parse_response::<super::CompletionResponse>(
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
    ) -> Result<crate::streaming::CompletionStream, CompletionError> {
        let req = build_request(cfg, &request, true)?;
        Ok(openai_functions::compatible_open_stream(
            rt,
            req,
            STREAM_DIALECT,
        ))
    }

    /// Send `request` to Mira and return the normalized response.
    pub async fn complete(
        cfg: &Config,
        rt: &HttpRuntime,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        let req = build_request(cfg, &request, false)?;
        let (status, body) = rt.send(req).await?;
        parse_response(status, &body)
    }

    /// Build the model-listing request for `cfg`.
    pub fn build_list_models_request(
        cfg: &Config,
    ) -> Result<http::Request<Vec<u8>>, crate::model::ModelListingError> {
        let url = format!(
            "{}{}",
            cfg.base_url.trim_end_matches('/'),
            super::LIST_MODELS_PATH
        );
        openai_functions::bearer_get(url, &cfg.api_key, &cfg.extra_headers)
    }

    /// List the models available to `cfg`'s credentials.
    ///
    /// The replacement for the deleted client's `list_models`, which returned
    /// bare `Vec<String>`; this returns the normalized
    /// [`ModelList`](crate::model::ModelList) every other provider's
    /// `list_models` returns.
    pub async fn list_models(
        cfg: &Config,
        rt: &HttpRuntime,
    ) -> Result<crate::model::ModelList, crate::model::ModelListingError> {
        let req = build_list_models_request(cfg)?;
        let (status, body) = rt.send_bytes(req).await?;
        super::parse_list_models_response(status, &body)
    }

    /// Verify that `cfg`'s credential is accepted by the provider.
    ///
    /// The data-oriented replacement for the deleted `VerifyClient::verify`: the
    /// endpoint is [`DESCRIPTOR`]'s `verify_path` (`/user-credits`, the value the
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
            assert_eq!(req.uri(), "https://api.mira.network/v1/chat/completions");
            let value: serde_json::Value = serde_json::from_slice(req.body()).expect("json");
            assert_eq!(value["model"], "test-model");
        }

        #[test]
        fn parse_response_normalizes() {
            let body = serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 3, "total_tokens": 5}
            })
            .to_string();
            let response = parse_response(http::StatusCode::OK, &body).expect("parse");
            assert_eq!(response.provider, "mira");
            assert_eq!(response.usage.input_tokens, 3);
            assert_eq!(response.usage.total_tokens, 5);
        }
    }
}
