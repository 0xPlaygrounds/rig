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
    json_utils,
    providers::openai::{self, Message},
    OneOrMany,
};
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use serde_json::json;

// ================================================================
// Main DeepSeek Client
// ================================================================
const DEEPSEEK_API_BASE_URL: &str = "https://api.deepseek.com";

#[derive(Clone)]
pub struct Client {
    pub base_url: String,
    pub api_key: String,
    http_client: HttpClient,
}

impl Client {
    // Create a new DeepSeek client from an API key.
    pub fn new(api_key: &str) -> Self {
        Self {
            base_url: DEEPSEEK_API_BASE_URL.to_string(),
            api_key: api_key.to_string(),
            http_client: HttpClient::new(),
        }
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
            api_key: api_key.to_string(),
            http_client: HttpClient::new(),
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
    pub choices: OneOrMany<Choice>,
    // you may want usage or other fields
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Choice {
    pub message: Message,
}

// #[derive(Clone, Debug, Serialize, Deserialize)]
// #[serde(tag = "role", rename_all = "lowercase")]
// pub enum Message {
//     System {
//         content: String,
//         name: Option<String>,
//     },
//     User {
//         content: String,
//         name: Option<String>,
//     },
//     Assistant {
//         content: String,
//         tool_calls: Option<Vec<DeepSeekToolCall>>,
//     },
//     Tool {
//         tool_call_id: String,
//         content: String,
//     },
// }

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: completion::ToolDefinition,
}

// #[derive(Debug, Deserialize)]
// pub struct DeepSeekMessage {
//     pub role: Option<String>,
//     pub content: Option<String>,
//     pub tool_calls: Option<Vec<DeepSeekToolCall>>,
// }

// #[derive(Debug, Deserialize)]
// pub struct DeepSeekToolCall {
//     pub id: String,
//     pub r#type: String,
//     pub function: DeepSeekFunction,
// }

// #[derive(Debug, Deserialize)]
// pub struct DeepSeekFunction {
//     pub name: String,
//     pub arguments: String,
// }

// #[derive(Clone, Debug, Deserialize, Serialize)]
// pub struct DeepSeekToolDefinition {
//     pub r#type: String,
//     pub function: crate::completion::ToolDefinition,
// }

impl From<crate::completion::ToolDefinition> for ToolDefinition {
    fn from(tool: crate::completion::ToolDefinition) -> Self {
        Self {
            r#type: "function".into(),
            function: tool,
        }
    }
}

// #[derive(Debug, Serialize, Deserialize, Clone)]
// pub struct Function {
//     pub name: String,
//     pub arguments: String,
// }

// impl TryFrom<message::Message> for Message {
//     type Error = message::MessageError;
//     fn try_from(message: message::Message) -> Result<Self, Self::Error> {
//         Ok(match message {
//             message::Message::User { content } => Message::User {
//                 content: content
//                     .into_iter()
//                     .map(|content| {
//                         Ok(match content {
//                             message::UserContent::Text(message::Text { text }) => text,
//                             _ => {
//                                 return Err(message::MessageError::ConversionError(
//                                     "Only text user content is supported by deepseek".to_owned(),
//                                 ))
//                             }
//                         })
//                     })
//                     .collect::<Result<Vec<_>, _>>()?
//                     .join("\n"),
//                 name: None,
//             },
//             message::Message::Assistant { content } => Message::Assistant {
//                 content: content
//                     .into_iter()
//                     .map(|content| {
//                         Ok(match content {
//                             message::AssistantContent::Text(message::Text { text }) => text,
//                             _ => {
//                                 return Err(message::MessageError::ConversionError(
//                                     "Only text assistant content is supported by deepseek"
//                                         .to_owned(),
//                                 ))
//                             }
//                         })
//                     })
//                     .collect::<Result<Vec<_>, _>>()?
//                     .join("\n"),
//             },
//         })
//     }
// }

// impl From<Message> for message::Message {
//     fn from(message: Message) -> Self {
//         match message {
//             Message::User { content, .. } => message::Message::user(content),
//             Message::Assistant { content, .. } => message::Message::assistant(content),

//             Message::Tool {
//                 tool_call_id,
//                 content,
//             } => message::Message::User {
//                 content: OneOrMany::one(message::UserContent::tool_result(
//                     tool_call_id,
//                     OneOrMany::one(content.into()),
//                 )),
//             },

//             // System messages should get stripped out when converting message's, this is just a
//             // stop gap to avoid obnoxious error handling or panic occuring.
//             Message::System { content, .. } => message::Message::user(content),
//         }
//     }
// }

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(value: CompletionResponse) -> Result<Self, Self::Error> {
        match value.choices.first() {
            Choice {
                message:
                    Message::Assistant {
                        content: Some(content),
                        ..
                    },
                ..
            } => {
                let content_str = content
                    .iter()
                    .map(|c| match c {
                        openai::AssistantContent::Text { text } => text.clone(),
                        openai::AssistantContent::Refusal { refusal } => refusal.clone(),
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                Ok(completion::CompletionResponse {
                    choice: completion::ModelChoice::Message(content_str),
                    raw_response: value,
                })
            }
            Choice {
                message:
                    Message::Assistant {
                        tool_calls: Some(tool_calls),
                        ..
                    },
                ..
            } => {
                let call = tool_calls.first();
                Ok(completion::CompletionResponse {
                    choice: completion::ModelChoice::ToolCall(
                        call.function.name.clone(),
                        "".to_string(),
                        call.function.arguments,
                    ),
                    raw_response: value,
                })
            }
            _ => Err(completion::CompletionError::ResponseError(
                "Response did not contain a valid message or tool call".into(),
            )),
        }
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

// Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_vec_choice() {
        let data = r#"[{"message":{"role":"assistant","content":"Hello, world!"}}]"#;
        let choices: OneOrMany<Choice> = serde_json::from_str(data).unwrap();
        assert_eq!(choices.len(), 1);
        match choices.first().message {
            Message::Assistant { content, .. } => assert_eq!(
                content.unwrap().first(),
                openai::AssistantContent::Text {
                    text: "Hello, world!".to_owned()
                }
            ),
            _ => panic!("Expected assistant message"),
        }
    }

    #[test]
    fn test_deserialize_deepseek_response() {
        let data = r#"{"choices":[{"message":{"role":"assistant","content":"Hello, world!"}}]}"#;
        let jd = &mut serde_json::Deserializer::from_str(data);
        let result: Result<CompletionResponse, _> = serde_path_to_error::deserialize(jd);
        match result {
            Ok(response) => match response.choices.first().message {
                Message::Assistant { content, .. } => assert_eq!(
                    content.unwrap().first(),
                    openai::AssistantContent::Text {
                        text: "Hello, world!".to_owned()
                    }
                ),
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
            Ok(response) => match response.choices.first().message {
                Message::Assistant { content, .. } => assert_eq!(
                    content.unwrap().first(),
                    openai::AssistantContent::Text {
                        text: "Why don’t skeletons fight each other?  \nBecause they don’t have the guts! 😄".to_owned()
                    }
                ),
                _ => panic!("Expected assistant message"),
            },
            Err(err) => {
                panic!("Deserialization error at {}: {}", err.path(), err);
            }
        }
    }
}
