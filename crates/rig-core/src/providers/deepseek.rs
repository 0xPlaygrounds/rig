//! DeepSeek API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig_core::{client::CompletionClient, providers::deepseek};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = deepseek::Client::new("DEEPSEEK_API_KEY")?;
//!
//! let deepseek_chat = client.completion_model(deepseek::DEEPSEEK_V4_FLASH);
//! # Ok(())
//! # }
//! ```

use serde_json::Value;

use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, ModelLister, Nothing, Provider,
    ProviderBuilder, ProviderClient,
};
use crate::completion::GetCompletionMetadata;
use crate::http_client::{self, HttpClientExt};
use crate::model::{Model, ModelList, ModelListingError};
use crate::providers::openai;
use crate::telemetry::ProviderResponseExt;
use crate::{
    OneOrMany,
    completion::{self, CompletionError},
    json_utils,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use serde::{Deserialize, Serialize};

// ================================================================
// Main DeepSeek Client
// ================================================================
const DEEPSEEK_API_BASE_URL: &str = "https://api.deepseek.com";

#[derive(Debug, Default, Clone, Copy)]
pub struct DeepSeekExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct DeepSeekExtBuilder;

type DeepSeekApiKey = BearerAuth;

impl Provider for DeepSeekExt {
    type Builder = DeepSeekExtBuilder;
    const VERIFY_PATH: &'static str = "/user/balance";
}

impl openai::completion::OpenAICompatibleProvider for DeepSeekExt {
    const PROVIDER_NAME: &'static str = "deepseek";

    type StreamingUsage = Usage;

    const EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS: bool = true;

    // DeepSeek's API only supports `json_object` response formats (passed via
    // `additional_params`), not the `json_schema` mapping of `output_schema`.
    const SUPPORTS_RESPONSE_FORMAT: bool = false;

    type Response = CompletionResponse;

    fn finalize_request_body(&self, body: &mut Value) -> Result<(), CompletionError> {
        let Some(map) = body.as_object_mut() else {
            return Ok(());
        };

        // DeepSeek takes message `content` as a plain string, not an array of
        // content parts, and echoes tool calls back with an `index` field.
        if let Some(messages) = map.get_mut("messages").and_then(Value::as_array_mut) {
            for message in messages {
                let Some(message) = message.as_object_mut() else {
                    continue;
                };
                let is_assistant = message.get("role").and_then(Value::as_str) == Some("assistant");

                if let Some(content) = message.get_mut("content") {
                    let separator = if is_assistant { "" } else { "\n" };
                    openai::completion::flatten_text_content_parts(content, separator, false);
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

        Ok(())
    }
}

impl<H> Capabilities<H> for DeepSeekExt {
    type Completion = Capable<CompletionModel<H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Capable<DeepSeekModelLister<H>>;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
    type Rerank = Nothing;
}

impl DebugExt for DeepSeekExt {}

impl ProviderBuilder for DeepSeekExtBuilder {
    type Extension<H>
        = DeepSeekExt
    where
        H: HttpClientExt;
    type ApiKey = DeepSeekApiKey;

    const BASE_URL: &'static str = DEEPSEEK_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(DeepSeekExt)
    }
}

pub type Client<H = reqwest::Client> = client::Client<DeepSeekExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<DeepSeekExtBuilder, DeepSeekApiKey, H>;

/// DeepSeek completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = reqwest::Client> =
    openai::completion::GenericCompletionModel<DeepSeekExt, H>;

/// Final streaming response, shared with the OpenAI Chat Completions path but
/// carrying DeepSeek's own usage payload (cache hit/miss counters).
pub type StreamingCompletionResponse = openai::StreamingCompletionResponse<Usage>;

impl ProviderClient for Client {
    type Input = DeepSeekApiKey;
    type Error = crate::client::ProviderClientError;

    // If you prefer the environment variable approach:
    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("DEEPSEEK_API_KEY")?;
        let mut client_builder = Self::builder();
        client_builder.headers_mut().insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        let client_builder = client_builder.api_key(&api_key);
        client_builder.build().map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(input).map_err(Into::into)
    }
}

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

impl GetCompletionMetadata for Usage {
    fn token_usage(&self) -> crate::completion::Usage {
        crate::completion::Usage {
            input_tokens: self.prompt_tokens as u64,
            output_tokens: self.completion_tokens as u64,
            total_tokens: self.total_tokens as u64,
            cached_input_tokens: self
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens)
                .map(u64::from)
                .unwrap_or(0),
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: self
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

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;
        let terminal_metadata = openai::completion::terminal_metadata_from_finish_reason(Some(
            choice.finish_reason.as_str(),
        ));
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

        let usage = response.usage.token_usage();
        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
            message_id: None,
            terminal_metadata,
        })
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

/// [`ModelLister`] implementation for the DeepSeek API (`GET /models`).
#[derive(Clone)]
pub struct DeepSeekModelLister<H = reqwest::Client> {
    client: Client<H>,
}

impl<H> ModelLister<H> for DeepSeekModelLister<H>
where
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
    type Client = Client<H>;

    fn new(client: Self::Client) -> Self {
        Self { client }
    }

    async fn list_all(&self) -> Result<ModelList, ModelListingError> {
        let path = "/models";
        let req = self.client.get(path)?.body(http_client::NoBody)?;
        let response = self
            .client
            .send::<_, Vec<u8>>(req)
            .await
            .map_err(|error| match error {
                http_client::Error::InvalidStatusCodeWithMessage(status, message) => {
                    ModelListingError::api_error_with_context(
                        "DeepSeek",
                        path,
                        status.as_u16(),
                        message.as_bytes(),
                    )
                }
                other => ModelListingError::from(other),
            })?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let body = response.into_body().await?;
            return Err(ModelListingError::api_error_with_context(
                "DeepSeek",
                path,
                status_code,
                &body,
            ));
        }

        let body = response.into_body().await?;
        let api_resp: ListModelsResponse = serde_json::from_slice(&body).map_err(|error| {
            ModelListingError::parse_error_with_context("DeepSeek", path, &error, &body)
        })?;

        let models = api_resp.data.into_iter().map(Model::from).collect();

        Ok(ModelList::new(models))
    }
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

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::ModelListingClient;
    use crate::completion::{CompletionRequestBuilder, ToolDefinition as RigToolDefinition};
    use crate::message::ToolChoice as RigToolChoice;
    use crate::providers::openai::completion::{
        CompletionRequest as OpenAICompletionRequest, OpenAICompatibleProvider, OpenAIRequestParams,
    };
    use crate::test_utils::{MockCompletionModel, RecordingHttpClient};

    fn finalized_body(request: crate::completion::CompletionRequest) -> serde_json::Value {
        let request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
            model: "deepseek-v4-flash".to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
            supports_response_format: DeepSeekExt::SUPPORTS_RESPONSE_FORMAT,
            supports_tools: true,
        })
        .expect("request should convert");
        let mut body = serde_json::to_value(request).expect("request should serialize");
        DeepSeekExt
            .finalize_request_body(&mut body)
            .expect("finalize should succeed");
        body
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
        let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Use a tool.")
            .tool(RigToolDefinition {
                name: "alpha".to_string(),
                description: "Alpha tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            })
            .tool(RigToolDefinition {
                name: "beta".to_string(),
                description: "Beta tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            })
            .tool_choice(RigToolChoice::Specific {
                function_names: vec!["beta".to_string()],
            })
            .additional_params(serde_json::json!({"thinking": {"type": "disabled"}}))
            .build();

        let body = finalized_body(request);

        assert_eq!(
            body["tool_choice"],
            serde_json::json!({"type": "function", "function": {"name": "beta"}})
        );
    }

    #[test]
    fn deepseek_request_suppresses_required_tool_choice_when_thinking_is_not_disabled() {
        let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Use a tool.")
            .tool(RigToolDefinition {
                name: "alpha".to_string(),
                description: "Alpha tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            })
            .tool_choice(RigToolChoice::Required)
            .build();

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
        let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Hello!")
            .preamble("You are helpful.".to_string())
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

        DeepSeekExt
            .finalize_request_body(&mut body)
            .expect("finalize should succeed");

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

        DeepSeekExt
            .finalize_request_body(&mut body)
            .expect("finalize should succeed");

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
    fn test_client_initialization() {
        let _client =
            crate::providers::deepseek::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::deepseek::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
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
        let client = Client::builder()
            .api_key("dummy-key")
            .http_client(http_client.clone())
            .build()
            .expect("client should build");

        let models = client
            .list_models()
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
        let client = Client::builder()
            .api_key("dummy-key")
            .http_client(http_client)
            .build()
            .expect("client should build");

        let error = client
            .list_models()
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
