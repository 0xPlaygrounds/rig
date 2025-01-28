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
    completion::{self, CompletionModel, CompletionRequest, CompletionResponse},
    json_utils, message, OneOrMany,
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

/// The response shape from the DeepSeek API
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepSeekResponse {
    // We'll match the JSON:
    pub choices: OneOrMany<Choice>,
    // you may want usage or other fields
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Choice {
    pub message: Message,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    System {
        content: String,
        name: Option<String>,
    },
    User {
        content: String,
        name: Option<String>,
    },
    Assistant {
        content: String,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: completion::ToolDefinition,
}

impl From<completion::ToolDefinition> for ToolDefinition {
    fn from(tool: completion::ToolDefinition) -> Self {
        Self {
            r#type: "function".into(),
            function: tool,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Function {
    pub name: String,
    pub arguments: String,
}

impl TryFrom<message::Message> for Message {
    type Error = message::MessageError;
    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        Ok(match message {
            message::Message::User { content } => Message::User {
                content: content
                    .into_iter()
                    .map(|content| {
                        Ok(match content {
                            message::UserContent::Text(message::Text { text }) => text,
                            _ => {
                                return Err(message::MessageError::ConversionError(
                                    "Only text user content is supported by deepseek".to_owned(),
                                ))
                            }
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .join("\n"),
                name: None,
            },
            message::Message::Assistant { content } => Message::Assistant {
                content: content
                    .into_iter()
                    .map(|content| {
                        Ok(match content {
                            message::AssistantContent::Text(message::Text { text }) => text,
                            _ => {
                                return Err(message::MessageError::ConversionError(
                                    "Only text assistant content is supported by deepseek"
                                        .to_owned(),
                                ))
                            }
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .join("\n"),
            },
        })
    }
}

impl From<Message> for message::Message {
    fn from(message: Message) -> Self {
        match message {
            Message::User { content, .. } => message::Message::user(content),
            Message::Assistant { content, .. } => message::Message::assistant(content),

            Message::Tool {
                tool_call_id,
                content,
            } => message::Message::User {
                content: OneOrMany::one(message::UserContent::tool_result(
                    tool_call_id,
                    OneOrMany::one(content.into()),
                )),
            },

            // System messages should get stripped out when converting message's, this is just a
            // stop gap to avoid obnoxious error handling or panic occuring.
            Message::System { content, .. } => message::Message::user(content),
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
    type Response = DeepSeekResponse;

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<DeepSeekResponse>, crate::completion::CompletionError> {
        // 1. Build the array of messages from request.chat_history + user prompt
        // if request.preamble is set, it becomes "system" or the first message.
        // So let's gather them in the style "system" + "user" + chat_history => JSON messages.

        let mut messages: Vec<Message> = vec![];
        let user_message = request.prompt_with_context();

        // If preamble is present, push a system message
        if let Some(preamble) = request.preamble {
            messages.push(Message::System {
                content: preamble,
                name: None,
            });
        }

        // If chat_history is present, we can push them.
        for msg in request.chat_history {
            messages.push(msg.try_into().map_err(|e| {
                crate::completion::CompletionError::ProviderError(format!(
                    "Message conversion error: {}",
                    e
                ))
            })?);
        }

        // Add user’s prompt
        messages.push(user_message.try_into().map_err(|e| {
            crate::completion::CompletionError::ProviderError(format!(
                "Message conversion error: {}",
                e
            ))
        })?);

        // 2. Prepare the body as DeepSeek expects
        let body = json!({
            "model": self.model,
            "messages": messages,
            "frequency_penalty": 0,
            "max_tokens": request.max_tokens.unwrap_or(2048),
            "presence_penalty": 0,
            "temperature": request.temperature.unwrap_or(1.0),
            "top_p": 1,
            "tool_choice": "none",
            "logprobs": false,
            "stream": false,
        });

        // if user set additional_params, merge them:
        let final_body = if let Some(params) = request.additional_params {
            json_utils::merge(body, params)
        } else {
            body
        };

        tracing::debug!("DeepSeek request: {}", final_body);

        // 3. Execute the HTTP call
        let url = format!("{}/chat/completions", self.client.base_url);
        let resp = self
            .client
            .http_client
            .post(url)
            .bearer_auth(&self.client.api_key)
            .json(&final_body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(crate::completion::CompletionError::ProviderError(format!(
                "DeepSeek call failed: {status} - {text}"
            )));
        }

        let text = resp.text().await?;
        tracing::debug!("DeepSeek response text: {}", text);
        let json_resp: DeepSeekResponse = serde_json::from_str(&text)?;
        // 4. Convert DeepSeekResponse -> rig’s `CompletionResponse<DeepSeekResponse>`

        // If no choices or content, return an empty message
        let content = match json_resp.choices.first().message {
            Message::User { content, .. }
            | Message::Assistant { content, .. }
            | Message::System { content, .. }
            | Message::Tool { content, .. } => content,
        };

        // For now, we just treat it as a normal text message
        let model_choice = crate::completion::ModelChoice::Message(content);

        Ok(CompletionResponse {
            choice: model_choice,
            raw_response: json_resp,
        })
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
    fn test_serialize_message() {
        let message = Message::Assistant {
            content: "Hello, world!".to_string(),
        };
        let serialized = serde_json::to_string(&message).unwrap();
        let expected = r#"{"role":"assistant","content":"Hello, world!"}"#;
        assert_eq!(serialized, expected);
    }

    #[test]
    fn test_deserialize_message() {
        let data = r#"{"role":"assistant","content":"Hello, world!"}"#;
        let message: Message = serde_json::from_str(data).unwrap();
        match message {
            Message::Assistant { content } => assert_eq!(content, "Hello, world!"),
            _ => panic!("Expected assistant message"),
        }
    }

    #[test]
    fn test_serialize_vec_choice() {
        let choices = vec![Choice {
            message: Message::Assistant {
                content: "Hello, world!".to_string(),
            },
        }];
        let serialized = serde_json::to_string(&choices).unwrap();
        let expected = r#"[{"message":{"role":"assistant","content":"Hello, world!"}}]"#;
        assert_eq!(serialized, expected);
    }

    #[test]
    fn test_deserialize_vec_choice() {
        let data = r#"[{"message":{"role":"assistant","content":"Hello, world!"}}]"#;
        let choices: Vec<Choice> = serde_json::from_str(data).unwrap();
        assert_eq!(choices.len(), 1);
        match &choices[0].message {
            Message::Assistant { content } => assert_eq!(content, "Hello, world!"),
            _ => panic!("Expected assistant message"),
        }
    }

    #[test]
    fn test_serialize_deepseek_response() {
        let response = DeepSeekResponse {
            choices: OneOrMany::one(Choice {
                message: Message::Assistant {
                    content: "Hello, world!".to_string(),
                },
            }),
        };
        let serialized = serde_json::to_string(&response).unwrap();
        let expected = r#"{"choices":[{"message":{"role":"assistant","content":"Hello, world!"}}]}"#;
        assert_eq!(serialized, expected);
    }

    #[test]
    fn test_deserialize_deepseek_response() {
        let data = r#"{"choices":[{"message":{"role":"assistant","content":"Hello, world!"}}]}"#;
        let jd = &mut serde_json::Deserializer::from_str(data);
        let result: Result<DeepSeekResponse, _> = serde_path_to_error::deserialize(jd);
        match result {
            Ok(response) => match &response.choices.first().message {
                Message::Assistant { ref content } => assert_eq!(content, "Hello, world!"),
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
        let result: Result<DeepSeekResponse, _> = serde_path_to_error::deserialize(jd);
        match result {
            Ok(response) => match &response.choices.first().message {
                Message::Assistant { content } => assert_eq!(
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
}