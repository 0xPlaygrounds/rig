//! DeepSeek API integration.
//!
//! # Example
//! ```no_run
//! use rig_core::providers::deepseek;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! # let request = rig_core::completion::CompletionRequest::from_prompt("hello");
//! let cfg = deepseek::functions::Config::from_env(deepseek::DEEPSEEK_V4_FLASH)?;
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//! let response = deepseek::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```

use crate::model::{Model, ModelList, ModelListingError};
use crate::providers::openai;
use crate::telemetry::ProviderResponseExt;
use crate::{
    OneOrMany,
    completion::{self, CompletionError},
    json_utils,
};
use serde::{Deserialize, Serialize};

/// The response shape from the DeepSeek API
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub system_fingerprint: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

impl ProviderResponseExt for CompletionResponse {
    type OutputMessage = Message;
    type Usage = Usage;

    fn get_response_id(&self) -> Option<String> {
        self.id.clone()
    }

    fn get_response_model_name(&self) -> Option<String> {
        self.model.clone()
    }

    fn get_output_messages(&self) -> Vec<Self::OutputMessage> {
        self.choices
            .iter()
            .map(|choice| choice.message.clone())
            .collect()
    }

    fn get_text_response(&self) -> Option<String> {
        self.choices
            .iter()
            .find_map(|choice| match &choice.message {
                Message::Assistant { content, .. } if !content.is_empty() => Some(content.clone()),
                _ => None,
            })
    }

    fn get_usage(&self) -> Option<Self::Usage> {
        Some(self.usage.clone())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Usage {
    pub completion_tokens: u32,
    pub prompt_tokens: u32,
    pub prompt_cache_hit_tokens: u32,
    pub prompt_cache_miss_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

impl From<Usage> for crate::completion::Usage {
    fn from(value: Usage) -> crate::completion::Usage {
        crate::completion::Usage {
            input_tokens: value.prompt_tokens as u64,
            output_tokens: value.completion_tokens as u64,
            total_tokens: value.total_tokens as u64,
            cached_input_tokens: value
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens)
                .map(u64::from)
                .unwrap_or(0),
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: value
                .completion_tokens_details
                .as_ref()
                .and_then(|details| details.reasoning_tokens)
                .map(u64::from)
                .unwrap_or(0),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CompletionTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct PromptTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

/// DeepSeek's provider-native message shape, as it appears in responses.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    System {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(
            default,
            deserialize_with = "json_utils::null_or_vec",
            skip_serializing_if = "Vec::is_empty"
        )]
        tool_calls: Vec<ToolCall>,
        /// only exists on `deepseek-reasoner` model at time of addition
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
    },
    #[serde(rename = "tool")]
    ToolResult {
        tool_call_id: String,
        content: String,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
    pub id: String,
    pub index: usize,
    #[serde(default)]
    pub r#type: ToolType,
    pub function: Function,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    #[serde(with = "json_utils::stringified_json")]
    pub arguments: serde_json::Value,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;
        let finish_reason = (!choice.finish_reason.is_empty())
            .then(|| openai::completion::map_finish_reason(&choice.finish_reason));
        let content = match &choice.message {
            Message::Assistant {
                content,
                tool_calls,
                reasoning_content,
                ..
            } => {
                let mut content = if content.trim().is_empty() {
                    vec![]
                } else {
                    vec![completion::AssistantContent::text(content)]
                };

                content.extend(
                    tool_calls
                        .iter()
                        .map(|call| {
                            completion::AssistantContent::tool_call(
                                &call.id,
                                &call.function.name,
                                call.function.arguments.clone(),
                            )
                        })
                        .collect::<Vec<_>>(),
                );

                if let Some(reasoning_content) = reasoning_content {
                    content.push(completion::AssistantContent::reasoning(reasoning_content));
                }

                Ok(content)
            }
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a valid message or tool call".into(),
            )),
        }?;

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        let usage = crate::completion::Usage::from(response.usage.clone());

        let mut normalized =
            completion::CompletionResponse::new(choice, usage, functions::DESCRIPTOR.name);
        if let Some(model) = response.model.clone() {
            normalized = normalized.with_model(model);
        }
        if let Some(id) = response.id.clone() {
            normalized = normalized.with_message_id(id);
        }
        normalized.finish_reason = finish_reason;
        Ok(normalized)
    }
}

#[derive(Debug, Deserialize)]
struct ListModelsResponse {
    data: Vec<ListModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ListModelEntry {
    id: String,
    owned_by: String,
}

impl From<ListModelEntry> for Model {
    fn from(value: ListModelEntry) -> Self {
        let mut model = Model::from_id(value.id);
        model.owned_by = Some(value.owned_by);
        model
    }
}

/// Path of the model-listing endpoint, relative to the API base URL.
pub(crate) const LIST_MODELS_PATH: &str = "/models";

/// Parse a `GET /models` response into a [`ModelList`]. Pure.
///
/// Used by [`functions::list_models`].
pub(crate) fn parse_list_models_response(
    status: http::StatusCode,
    body: &[u8],
) -> Result<ModelList, ModelListingError> {
    if !status.is_success() {
        return Err(ModelListingError::api_error_with_context(
            "DeepSeek",
            LIST_MODELS_PATH,
            status.as_u16(),
            body,
        ));
    }
    let api_resp: ListModelsResponse = serde_json::from_slice(body).map_err(|error| {
        ModelListingError::parse_error_with_context("DeepSeek", LIST_MODELS_PATH, &error, body)
    })?;
    let models = api_resp.data.into_iter().map(Model::from).collect();
    Ok(ModelList::new(models))
}

// ================================================================
// DeepSeek Completion API
// ================================================================
#[deprecated(
    note = "The model names `deepseek-chat` and `deepseek-reasoner` will be deprecated on 2026/07/24. \
    For compatibility, they correspond to the non-thinking mode and thinking mode of `deepseek-v4-flash`, \
    respectively."
)]
pub const DEEPSEEK_CHAT: &str = "deepseek-chat";
#[deprecated(
    note = "The model names `deepseek-chat` and `deepseek-reasoner` will be deprecated on 2026/07/24. \
    For compatibility, they correspond to the non-thinking mode and thinking mode of `deepseek-v4-flash`, \
    respectively."
)]
pub const DEEPSEEK_REASONER: &str = "deepseek-reasoner";
pub const DEEPSEEK_V4_FLASH: &str = "deepseek-v4-flash";
pub const DEEPSEEK_V4_PRO: &str = "deepseek-v4-pro";
#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::{
        CompletionRequest as RigCompletionRequest, ToolDefinition as RigToolDefinition,
    };
    use crate::http_runtime::HttpRuntime;
    use crate::message::ToolChoice as RigToolChoice;
    use crate::providers::openai::completion::CompletionModelOptions;
    use crate::test_utils::RecordingHttpClient;

    fn finalized_body(request: crate::completion::CompletionRequest) -> serde_json::Value {
        let body = super::functions::build_body(
            "deepseek-v4-flash",
            &request,
            CompletionModelOptions::default(),
            false,
        )
        .expect("build_body should succeed");
        serde_json::from_slice(&body).expect("body should be JSON")
    }

    #[test]
    fn test_deserialize_vec_choice() {
        let data = r#"[{
            "finish_reason": "stop",
            "index": 0,
            "logprobs": null,
            "message":{"role":"assistant","content":"Hello, world!"}
            }]"#;

        let choices: Vec<Choice> = serde_json::from_str(data).unwrap();
        assert_eq!(choices.len(), 1);
        match &choices.first().unwrap().message {
            Message::Assistant { content, .. } => assert_eq!(content, "Hello, world!"),
            _ => panic!("Expected assistant message"),
        }
    }

    #[test]
    fn test_deserialize_deepseek_response() {
        let data = r#"{
            "choices":[{
                "finish_reason": "stop",
                "index": 0,
                "logprobs": null,
                "message":{"role":"assistant","content":"Hello, world!"}
            }],
            "usage": {
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 0,
                "total_tokens": 0
            }
        }"#;

        let jd = &mut serde_json::Deserializer::from_str(data);
        let result: Result<CompletionResponse, _> = serde_path_to_error::deserialize(jd);
        match result {
            Ok(response) => match &response.choices.first().unwrap().message {
                Message::Assistant { content, .. } => assert_eq!(content, "Hello, world!"),
                _ => panic!("Expected assistant message"),
            },
            Err(err) => {
                panic!("Deserialization error at {}: {}", err.path(), err);
            }
        }
    }

    #[test]
    fn deepseek_request_serializes_specific_tool_choice_as_chat_completions_object() {
        let request = RigCompletionRequest {
            tools: vec![
                RigToolDefinition {
                    name: "alpha".to_string(),
                    description: "Alpha tool".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {},
                        "required": []
                    }),
                },
                RigToolDefinition {
                    name: "beta".to_string(),
                    description: "Beta tool".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {},
                        "required": []
                    }),
                },
            ],
            tool_choice: Some(RigToolChoice::Specific {
                function_names: vec!["beta".to_string()],
            }),
            additional_params: Some(serde_json::json!({"thinking": {"type": "disabled"}})),
            ..RigCompletionRequest::from_prompt("Use a tool.")
        };

        let body = finalized_body(request);

        assert_eq!(
            body["tool_choice"],
            serde_json::json!({"type": "function", "function": {"name": "beta"}})
        );
    }

    #[test]
    fn deepseek_request_suppresses_required_tool_choice_when_thinking_is_not_disabled() {
        let request = RigCompletionRequest {
            tools: vec![RigToolDefinition {
                name: "alpha".to_string(),
                description: "Alpha tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            }],
            tool_choice: Some(RigToolChoice::Required),
            ..RigCompletionRequest::from_prompt("Use a tool.")
        };

        let body = finalized_body(request);

        assert!(
            body.as_object()
                .expect("body is object")
                .contains_key("tool_choice"),
            "suppressed tool_choice should stay present as an explicit null"
        );
        assert_eq!(body["tool_choice"], serde_json::Value::Null);
    }

    #[test]
    fn deepseek_request_flattens_message_content_to_strings() {
        let request = RigCompletionRequest::builder("Hello!")
            .preamble("You are helpful.")
            .messages(Vec::new())
            .build();

        let body = finalized_body(request);

        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "You are helpful.");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][1]["content"], "Hello!");
    }

    #[test]
    fn deepseek_finalize_joins_user_parts_with_newline_and_concats_assistant_parts() {
        let mut body = serde_json::json!({
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "first part"},
                    {"type": "text", "text": "second part"}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": " world"}
                ]}
            ]
        });

        super::functions::apply_wire_dialect(&mut body);

        assert_eq!(body["messages"][0]["content"], "first part\nsecond part");
        assert_eq!(body["messages"][1]["content"], "Hello world");
    }

    #[test]
    fn deepseek_finalize_adds_tool_call_index_to_assistant_history() {
        let mut body = serde_json::json!({
            "model": "deepseek-v4-flash",
            "messages": [{
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "subtract", "arguments": "{\"x\":2,\"y\":5}"}
                }]
            }]
        });

        super::functions::apply_wire_dialect(&mut body);

        assert_eq!(body["messages"][0]["tool_calls"][0]["index"], 0);
    }

    #[test]
    fn deepseek_response_preserves_metadata_and_reasoning_token_usage() {
        let raw: CompletionResponse = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl_123",
            "object": "chat.completion",
            "model": "deepseek-v4-flash",
            "system_fingerprint": "fp_123",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "logprobs": null,
                "message": {
                    "role": "assistant",
                    "content": "done",
                    "reasoning_content": "thinking"
                }
            }],
            "usage": {
                "completion_tokens": 8,
                "completion_tokens_details": { "reasoning_tokens": 5 },
                "prompt_tokens": 10,
                "prompt_tokens_details": { "cached_tokens": 3 },
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 10,
                "total_tokens": 18
            }
        }))
        .expect("fixture should deserialize");

        let converted = crate::completion::CompletionResponse::try_from(raw.clone())
            .expect("DeepSeek response should convert");

        assert_eq!(raw.id.as_deref(), Some("chatcmpl_123"));
        assert_eq!(raw.model.as_deref(), Some("deepseek-v4-flash"));
        assert_eq!(raw.system_fingerprint.as_deref(), Some("fp_123"));
        assert_eq!(converted.usage.input_tokens, 10);
        assert_eq!(converted.usage.cached_input_tokens, 3);
        assert_eq!(converted.usage.output_tokens, 8);
        assert_eq!(converted.usage.reasoning_tokens, 5);
    }

    #[test]
    fn test_deserialize_example_response() {
        let data = r#"
        {
            "id": "e45f6c68-9d9e-43de-beb4-4f402b850feb",
            "object": "chat.completion",
            "created": 0,
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Why don’t skeletons fight each other?  \nBecause they don’t have the guts! 😄"
                    },
                    "logprobs": null,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 32,
                "total_tokens": 45,
                "prompt_tokens_details": {
                    "cached_tokens": 0
                },
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 13
            },
            "system_fingerprint": "fp_4b6881f2c5"
        }
        "#;
        let jd = &mut serde_json::Deserializer::from_str(data);
        let result: Result<CompletionResponse, _> = serde_path_to_error::deserialize(jd);

        match result {
            Ok(response) => match &response.choices.first().unwrap().message {
                Message::Assistant { content, .. } => assert_eq!(
                    content,
                    "Why don’t skeletons fight each other?  \nBecause they don’t have the guts! 😄"
                ),
                _ => panic!("Expected assistant message"),
            },
            Err(err) => {
                panic!("Deserialization error at {}: {}", err.path(), err);
            }
        }
    }

    #[test]
    fn test_serialize_deserialize_tool_call_message() {
        let tool_call_choice_json = r#"
            {
              "finish_reason": "tool_calls",
              "index": 0,
              "logprobs": null,
              "message": {
                "content": "",
                "role": "assistant",
                "tool_calls": [
                  {
                    "function": {
                      "arguments": "{\"x\":2,\"y\":5}",
                      "name": "subtract"
                    },
                    "id": "call_0_2b4a85ee-b04a-40ad-a16b-a405caf6e65b",
                    "index": 0,
                    "type": "function"
                  }
                ]
              }
            }
        "#;

        let choice: Choice =
            serde_json::from_str(tool_call_choice_json).expect("choice should deserialize");
        match &choice.message {
            Message::Assistant { tool_calls, .. } => {
                assert_eq!(tool_calls.len(), 1);
                let call = tool_calls.first().expect("one tool call");
                assert_eq!(call.function.name, "subtract");
                assert_eq!(call.index, 0);
            }
            _ => panic!("Expected assistant message"),
        }

        let serialized = serde_json::to_value(&choice).expect("choice should serialize");
        assert_eq!(
            serialized["message"]["tool_calls"][0]["function"]["name"],
            "subtract"
        );
    }

    #[test]
    fn test_deserialize_list_models_response() {
        let data = r#"{
            "object": "list",
            "data": [
                {"id": "deepseek-chat", "object": "model", "owned_by": "deepseek"},
                {"id": "deepseek-reasoner", "object": "model", "owned_by": "deepseek"}
            ]
        }"#;

        let response: ListModelsResponse =
            serde_json::from_str(data).expect("list models response should deserialize");
        assert_eq!(response.data.len(), 2);
        assert_eq!(response.data[0].id, "deepseek-chat");
        assert_eq!(response.data[0].owned_by, "deepseek");
    }

    #[tokio::test]
    async fn test_list_models_uses_models_endpoint() {
        let response_body = r#"{
            "object": "list",
            "data": [
                {
                    "id": "deepseek-v4-flash",
                    "object": "model",
                    "owned_by": "deepseek"
                },
                {
                    "id": "deepseek-v4-pro",
                    "object": "model",
                    "owned_by": "deepseek"
                }
            ]
        }"#;

        let http_client = RecordingHttpClient::new(response_body);
        let rt = HttpRuntime::recording(http_client.clone());
        let cfg = functions::Config::new("deepseek-v4-flash").with_api_key("dummy-key");

        let models = functions::list_models(&cfg, &rt)
            .await
            .expect("list_models should succeed");

        assert_eq!(models.len(), 2);
        assert_eq!(models.data[0].id, "deepseek-v4-flash");
        assert_eq!(models.data[0].r#type, None);
        assert_eq!(models.data[0].owned_by.as_deref(), Some("deepseek"));
        let requests = http_client.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].uri, "https://api.deepseek.com/models");
    }

    #[tokio::test]
    async fn test_list_models_preserves_api_error_context() {
        let http_client = RecordingHttpClient::with_error(
            http::StatusCode::UNAUTHORIZED,
            r#"{"error":{"message":"invalid api key"}}"#,
        );
        let rt = HttpRuntime::recording(http_client);
        let cfg = functions::Config::new("deepseek-v4-flash").with_api_key("dummy-key");

        let error = functions::list_models(&cfg, &rt)
            .await
            .expect_err("list_models should fail");

        match error {
            ModelListingError::ApiError {
                status_code,
                message,
            } => {
                assert_eq!(status_code, 401);
                assert!(message.contains("provider=DeepSeek"));
                assert!(message.contains("path=/models"));
                assert!(message.contains("invalid api key"));
            }
            other => panic!("expected api error, got {other:?}"),
        }
    }
}

crate::providers::client::define_http_client! {
    config = functions::Config,
    default_base_url = functions::DEFAULT_BASE_URL,
    api_key_required = true,
}

pub mod functions {
    //! DeepSeek chat completions as config + pure functions.
    //!
    //! The data-oriented face of the DeepSeek provider, mirroring
    //! [`crate::providers::openai::functions`]: a serde [`Config`], a
    //! [`DESCRIPTOR`] capability sheet, and pure
    //! [`build_request`]/[`parse_response`] free functions plus the async
    //! [`complete`]/[`open_stream`] wrappers over
    //! [`HttpRuntime`]. The request/parse
    //! mechanics are shared with the other OpenAI-compatible providers via
    //! `openai::functions`; this module instantiates them with DeepSeek's
    //! [`DESCRIPTOR`] so DeepSeek's paths, hooks, and provider name apply.

    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    use crate::completion::{self, CompletionError, CompletionRequest};
    use crate::http_runtime::HttpRuntime;
    use crate::providers::descriptor::{
        ApiKeyLocation, ConfigError, ProviderDescriptor, required_env_var,
    };
    use crate::providers::internal::openai_chat_completions_compatible::{
        ChatCompletionsDialect, ChatCompletionsUsageDialect,
    };
    use crate::providers::openai::completion::CompletionModelOptions;
    use crate::providers::openai::functions as openai_functions;

    /// Default DeepSeek API base URL.
    pub const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";

    /// DeepSeek's Chat Completions streaming dialect: OpenAI-shaped chunks with
    /// DeepSeek's own cache hit/miss usage accounting.
    pub(crate) const STREAM_DIALECT: ChatCompletionsDialect =
        ChatCompletionsDialect::from_descriptor(&DESCRIPTOR)
            .with_usage(ChatCompletionsUsageDialect::DeepSeek);

    /// DeepSeek's capability sheet.
    pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
        name: "deepseek",
        supports_tools: true,
        supports_response_format: false,
        stream_include_usage: true,
        emits_complete_single_chunk_tool_calls: true,
        composes_native_output_with_tools: false,
        max_embedding_documents: None,
        verify_path: Some("/user/balance"),
    };

    /// Plain-data DeepSeek provider configuration.
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
        /// Config for `model` reading `DEEPSEEK_API_KEY` from the environment.
        pub fn new(model: impl Into<String>) -> Self {
            Self {
                connection: crate::providers::HttpConnectionConfig::new(
                    DEFAULT_BASE_URL.to_string(),
                    ApiKeyLocation::Env("DEEPSEEK_API_KEY".to_string()),
                ),
                model: model.into(),
            }
        }

        /// Config for `model` built from the process environment.
        ///
        /// Reads `DEEPSEEK_API_KEY` (required) — the same variable the deleted
        /// `deepseek::Client::from_env` read. The credential is validated eagerly
        /// but stored as [`ApiKeyLocation::Env`], so the secret is read at
        /// request time rather than held inside the config.
        ///
        /// # Errors
        /// [`ConfigError`] when a required variable is missing or invalid.
        pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
            let cfg = Self::new(model);
            required_env_var("DEEPSEEK_API_KEY")?;
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

    /// DeepSeek's straight-line chat-completions body assembly.
    ///
    /// Three wire-level dialect quirks: message `content` is a plain string
    /// rather than an array of content parts, assistant tool calls are echoed
    /// back with an `index` field, and forced tool choices (`required` or a
    /// specific function) are rejected unless thinking is explicitly disabled,
    /// so they are suppressed to an explicit `null`. DeepSeek also only supports
    /// `json_object` response formats (passed via `additional_params`), not the
    /// `json_schema` mapping of `output_schema` — hence
    /// `supports_response_format: false` on [`DESCRIPTOR`].
    pub(crate) fn build_body(
        model: &str,
        request: &CompletionRequest,
        options: CompletionModelOptions,
        stream: bool,
    ) -> Result<Vec<u8>, CompletionError> {
        let typed =
            openai_functions::compatible_typed_request(model, request, &DESCRIPTOR, options)?;
        let mut body = openai_functions::compatible_body_value(&typed, &DESCRIPTOR, stream)?;
        apply_wire_dialect(&mut body);
        Ok(serde_json::to_vec(&body)?)
    }

    /// Rewrite a serialized chat-completions `body` into DeepSeek's wire dialect
    /// in place: string message `content`, `index`-stamped assistant tool calls,
    /// and forced tool choices suppressed to an explicit `null`.
    pub(crate) fn apply_wire_dialect(body: &mut Value) {
        if let Some(map) = body.as_object_mut() {
            // DeepSeek takes message `content` as a plain string, not an array of
            // content parts, and echoes tool calls back with an `index` field.
            if let Some(messages) = map.get_mut("messages").and_then(Value::as_array_mut) {
                for message in messages {
                    let Some(message) = message.as_object_mut() else {
                        continue;
                    };
                    let is_assistant =
                        message.get("role").and_then(Value::as_str) == Some("assistant");

                    if let Some(content) = message.get_mut("content") {
                        let separator = if is_assistant { "" } else { "\n" };
                        crate::providers::openai::completion::flatten_text_content_parts(
                            content, separator, false,
                        );
                    } else if is_assistant && !message.contains_key("content") {
                        // Tool-call-only assistant turns must still carry an
                        // (empty) string content field.
                        message.insert("content".to_string(), Value::String(String::new()));
                    }

                    if is_assistant
                        && let Some(tool_calls) =
                            message.get_mut("tool_calls").and_then(Value::as_array_mut)
                    {
                        for tool_call in tool_calls {
                            if let Some(tool_call) = tool_call.as_object_mut() {
                                tool_call
                                    .entry("index")
                                    .or_insert_with(|| serde_json::json!(0));
                            }
                        }
                    }
                }
            }

            // DeepSeek rejects forced tool choices (`required` or a specific
            // function) unless thinking is explicitly disabled; suppress them to
            // an explicit `null` otherwise.
            let thinking_disabled = map
                .get("thinking")
                .and_then(|thinking| thinking.get("type"))
                .and_then(Value::as_str)
                .is_some_and(|mode| mode.eq_ignore_ascii_case("disabled"));
            if !thinking_disabled && let Some(tool_choice) = map.get_mut("tool_choice") {
                let forced = tool_choice.is_object() || tool_choice.as_str() == Some("required");
                if forced {
                    *tool_choice = Value::Null;
                }
            }
        }
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
    ) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError> {
        let req = build_request(cfg, &request, true)?;
        Ok(openai_functions::compatible_open_stream(
            rt,
            req,
            STREAM_DIALECT,
        ))
    }

    /// Send `request` to DeepSeek and return the normalized response.
    pub async fn complete(
        cfg: &Config,
        rt: &HttpRuntime,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        let req = build_request(cfg, &request, false)?;
        let (status, body) = rt.send(req).await?;
        parse_response(status, &body)
    }

    /// Build the `GET /models` request for [`list_models`].
    ///
    /// Pure except for credential resolution (`ApiKeyLocation::Env` reads
    /// the environment).
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
    /// Parsing goes through the pure `super::parse_list_models_response`.
    pub async fn list_models(
        cfg: &Config,
        rt: &HttpRuntime,
    ) -> Result<crate::model::ModelList, crate::model::ModelListingError> {
        let req = build_list_models_request(cfg)?;
        let (status, body) = rt.send_bytes(req).await?;
        super::parse_list_models_response(status, &body)
    }

    // Tests
    /// Verify that `cfg`'s credential is accepted by the provider.
    ///
    /// The data-oriented replacement for the deleted `VerifyClient::verify`: the
    /// endpoint is [`DESCRIPTOR`]'s `verify_path` (`/user/balance`, the value the
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
            assert_eq!(req.uri(), "https://api.deepseek.com/chat/completions");
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
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "prompt_cache_hit_tokens": 0,
                    "prompt_cache_miss_tokens": 3,
                    "total_tokens": 5
                }
            })
            .to_string();
            let response = parse_response(http::StatusCode::OK, &body).expect("parse");
            assert_eq!(response.provider, "deepseek");
            assert_eq!(response.usage.input_tokens, 3);
            assert_eq!(response.usage.total_tokens, 5);
        }
    }
}
