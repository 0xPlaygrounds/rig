//! DeepSeek API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::deepseek;
//!
//! let client = deepseek::Client::new("DEEPSEEK_API_KEY");
//!
//! let deepseek_chat = client.completion_model(deepseek::DEEPSEEK_CHAT);
//! ```
use crate::{
    completion::{self, CompletionError, CompletionModel, CompletionRequest},
    extractor::ExtractorBuilder,
    json_utils, message, OneOrMany,
};
use reqwest::Client as HttpClient;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

// ================================================================
// Main DeepSeek Client
// ================================================================
const DEEPSEEK_API_BASE_URL: &str = "https://api.deepseek.com";

#[derive(Clone)]
pub struct Client {
    pub base_url: String,
    http_client: HttpClient,
}

impl Client {
    // Create a new DeepSeek client from an API key.
    pub fn new(api_key: &str) -> Self {
        Self::from_url(api_key, DEEPSEEK_API_BASE_URL)
    }

    // If you prefer the environment variable approach:
    pub fn from_env() -> Self {
        let api_key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY not set");
        Self::new(&api_key)
    }

    // Handy for advanced usage, e.g. letting user override base_url or set timeouts:
    pub fn from_url(api_key: &str, base_url: &str) -> Self {
        // Possibly configure a custom HTTP client here if needed.
        Self {
            base_url: base_url.to_string(),
            http_client: reqwest::Client::builder()
                .default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        "Authorization",
                        format!("Bearer {}", api_key)
                            .parse()
                            .expect("Bearer token should parse"),
                    );
                    headers
                })
                .build()
                .expect("DeepSeek reqwest client should build"),
        }
    }

    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.post(url)
    }

    /// Creates a DeepSeek completion model with the given `model_name`.
    pub fn completion_model(&self, model_name: &str) -> DeepSeekCompletionModel {
        DeepSeekCompletionModel {
            client: self.clone(),
            model: model_name.to_string(),
        }
    }

    /// Optionally add an agent() convenience:
    pub fn agent(&self, model_name: &str) -> crate::agent::AgentBuilder<DeepSeekCompletionModel> {
        crate::agent::AgentBuilder::new(self.completion_model(model_name))
    }

    /// Create an extractor builder with the given completion model.
    pub fn extractor<T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync>(
        &self,
        model: &str,
    ) -> ExtractorBuilder<T, DeepSeekCompletionModel> {
        ExtractorBuilder::new(self.completion_model(model))
    }
}

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

impl From<ApiErrorResponse> for CompletionError {
    fn from(err: ApiErrorResponse) -> Self {
        CompletionError::ProviderError(err.message)
    }
}

/// The response shape from the DeepSeek API
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    // We'll match the JSON:
    pub choices: Vec<Choice>,
    // you may want usage or other fields
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

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
        #[serde(default, deserialize_with = "json_utils::null_or_vec")]
        tool_calls: Vec<ToolCall>,
    },
    #[serde(rename = "Tool")]
    ToolResult {
        tool_call_id: String,
        content: String,
    },
}

impl Message {
    pub fn system(content: &str) -> Self {
        Message::System {
            content: content.to_owned(),
            name: None,
        }
    }
}

impl From<message::ToolResult> for Message {
    fn from(tool_result: message::ToolResult) -> Self {
        let content = match tool_result.content.first() {
            message::ToolResultContent::Text(text) => text.text,
            message::ToolResultContent::Image(_) => String::from("[Image]"),
        };

        Message::ToolResult {
            tool_call_id: tool_result.id,
            content,
        }
    }
}

impl From<message::ToolCall> for ToolCall {
    fn from(tool_call: message::ToolCall) -> Self {
        Self {
            id: tool_call.id,
            // TODO: update index when we have it
            index: 0,
            r#type: ToolType::Function,
            function: Function {
                name: tool_call.function.name,
                arguments: tool_call.function.arguments,
            },
        }
    }
}

impl TryFrom<message::Message> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::User { content } => {
                // extract tool results
                let mut messages = vec![];

                let tool_results = content
                    .clone()
                    .into_iter()
                    .filter_map(|content| match content {
                        message::UserContent::ToolResult(tool_result) => {
                            Some(Message::from(tool_result))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();

                messages.extend(tool_results);

                // extract text results
                let text_messages = content
                    .into_iter()
                    .filter_map(|content| match content {
                        message::UserContent::Text(text) => Some(Message::User {
                            content: text.text,
                            name: None,
                        }),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                messages.extend(text_messages);

                Ok(messages)
            }
            message::Message::Assistant { content } => {
                let mut messages: Vec<Message> = vec![];

                // extract tool calls
                let tool_calls = content
                    .clone()
                    .into_iter()
                    .filter_map(|content| match content {
                        message::AssistantContent::ToolCall(tool_call) => {
                            Some(ToolCall::from(tool_call))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();

                // if we have tool calls, we add a new Assistant message with them
                if !tool_calls.is_empty() {
                    messages.push(Message::Assistant {
                        content: "".to_string(),
                        name: None,
                        tool_calls,
                    });
                }

                // extract text
                let text_content = content
                    .into_iter()
                    .filter_map(|content| match content {
                        message::AssistantContent::Text(text) => Some(Message::Assistant {
                            content: text.text,
                            name: None,
                            tool_calls: vec![],
                        }),
                        _ => None,
                    })
                    .collect::<Vec<_>>();

                messages.extend(text_content);

                Ok(messages)
            }
        }
    }
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

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: completion::ToolDefinition,
}

impl From<crate::completion::ToolDefinition> for ToolDefinition {
    fn from(tool: crate::completion::ToolDefinition) -> Self {
        Self {
            r#type: "function".into(),
            function: tool,
        }
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;
        let content = match &choice.message {
            Message::Assistant {
                content,
                tool_calls,
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
                                &call.function.name,
                                &call.function.name,
                                call.function.arguments.clone(),
                            )
                        })
                        .collect::<Vec<_>>(),
                );
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

        Ok(completion::CompletionResponse {
            choice,
            raw_response: response,
        })
    }
}

/// The struct implementing the `CompletionModel` trait
#[derive(Clone)]
pub struct DeepSeekCompletionModel {
    pub client: Client,
    pub model: String,
}

impl CompletionModel for DeepSeekCompletionModel {
    type Response = CompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<
        completion::CompletionResponse<CompletionResponse>,
        crate::completion::CompletionError,
    > {
        // Add preamble to chat history (if available)
        let mut full_history: Vec<Message> = match &completion_request.preamble {
            Some(preamble) => vec![Message::system(preamble)],
            None => vec![],
        };

        // Convert prompt to user message
        let prompt: Vec<Message> = completion_request.prompt_with_context().try_into()?;

        // Convert existing chat history
        let chat_history: Vec<Message> = completion_request
            .chat_history
            .into_iter()
            .map(|message| message.try_into())
            .collect::<Result<Vec<Vec<Message>>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        // Combine all messages into a single history
        full_history.extend(chat_history);
        full_history.extend(prompt);

        let request = if completion_request.tools.is_empty() {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
            })
        } else {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
                "tools": completion_request.tools.into_iter().map(ToolDefinition::from).collect::<Vec<_>>(),
                "tool_choice": "auto",
            })
        };

        let response = self
            .client
            .post("/chat/completions")
            .json(
                &if let Some(params) = completion_request.additional_params {
                    json_utils::merge(request, params)
                } else {
                    request
                },
            )
            .send()
            .await?;

        if response.status().is_success() {
            let t = response.text().await?;
            tracing::debug!(target: "rig", "OpenAI completion error: {}", t);

            match serde_json::from_str::<ApiResponse<CompletionResponse>>(&t)? {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
            }
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }
}

// ================================================================
// DeepSeek Completion API
// ================================================================

/// `deepseek-chat` completion model
pub const DEEPSEEK_CHAT: &str = "deepseek-chat";
/// `deepseek-reasoner` completion model
pub const DEEPSEEK_REASONER: &str = "deepseek-reasoner";

// Tests
#[cfg(test)]
mod tests {

    use super::*;

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
        let data = r#"{"choices":[{
            "finish_reason": "stop",
            "index": 0,
            "logprobs": null,
            "message":{"role":"assistant","content":"Hello, world!"}
            }]}"#;

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

        let choice: Choice = serde_json::from_str(tool_call_choice_json).unwrap();

        let expected_choice: Choice = Choice {
            finish_reason: "tool_calls".to_string(),
            index: 0,
            logprobs: None,
            message: Message::Assistant {
                content: "".to_string(),
                name: None,
                tool_calls: vec![ToolCall {
                    id: "call_0_2b4a85ee-b04a-40ad-a16b-a405caf6e65b".to_string(),
                    function: Function {
                        name: "subtract".to_string(),
                        arguments: serde_json::from_str(r#"{"x":2,"y":5}"#).unwrap(),
                    },
                    index: 0,
                    r#type: ToolType::Function,
                }],
            },
        };

        assert_eq!(choice, expected_choice);
    }
}
