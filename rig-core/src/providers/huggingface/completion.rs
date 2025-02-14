use std::{convert::Infallible, str::FromStr};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    completion::{self, CompletionError, CompletionRequest},
    json_utils,
    message::{self},
    one_or_many::string_or_one_or_many,
    providers::huggingface::client::SubProvider,
    OneOrMany,
};

use super::client::Client;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ApiResponse<T> {
    Ok(T),
    Err(Value),
}

// ================================================================
// Huggingface Completion API
// ================================================================

// Conversational LLMs

/// `google/gemma-2-2b-it` completion model
pub const GEMMA_2: &str = "google/gemma-2-2b-it";
/// `meta-llama/Meta-Llama-3.1-8B-Instruct` completion model
pub const META_LLAMA_3_1: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct";
/// `microsoft/phi-4` completion model
pub const PHI_4: &str = "microsoft/phi-4";
/// `PowerInfer/SmallThinker-3B-Preview` completion model
pub const SMALLTHINKER_PREVIEW: &str = "PowerInfer/SmallThinker-3B-Preview";
/// `Qwen/Qwen2.5-7B-Instruct` completion model
pub const QWEN2_5: &str = "Qwen/Qwen2.5-7B-Instruct";
/// `Qwen/Qwen2.5-Coder-32B-Instruct` completion model
pub const QWEN2_5_CODER: &str = "Qwen/Qwen2.5-Coder-32B-Instruct";

// Conversational VLMs

/// `Qwen/Qwen2-VL-7B-Instruct` visual-language completion model
pub const QWEN2_VL: &str = "Qwen/Qwen2-VL-7B-Instruct";
/// `Qwen/QVQ-72B-Preview` visual-language completion model
pub const QWEN_QVQ_PREVIEW: &str = "Qwen/QVQ-72B-Preview";

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct Function {
    name: String,
    pub arguments: serde_json::Value,
}

impl From<Function> for message::ToolFunction {
    fn from(value: Function) -> Self {
        message::ToolFunction {
            name: value.name,
            arguments: value.arguments,
        }
    }
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
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

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct ToolCall {
    pub id: String,
    pub r#type: ToolType,
    pub function: Function,
}

impl From<ToolCall> for message::ToolCall {
    fn from(value: ToolCall) -> Self {
        message::ToolCall {
            id: value.id,
            function: value.function.into(),
        }
    }
}

impl From<message::ToolCall> for ToolCall {
    fn from(value: message::ToolCall) -> Self {
        ToolCall {
            id: value.id,
            r#type: ToolType::Function,
            function: Function {
                name: value.function.name,
                arguments: value.function.arguments,
            },
        }
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct ImageUrl {
    url: String,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContent {
    Text {
        text: String,
    },
    #[serde(rename = "image_url")]
    ImageUrl {
        image_url: ImageUrl,
    },
}

impl FromStr for UserContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(UserContent::Text {
            text: s.to_string(),
        })
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AssistantContent {
    Text { text: String },
}

impl FromStr for AssistantContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(AssistantContent::Text {
            text: s.to_string(),
        })
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SystemContent {
    Text { text: String },
}

impl FromStr for SystemContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(SystemContent::Text {
            text: s.to_string(),
        })
    }
}

impl Into<message::UserContent> for UserContent {
    fn into(self) -> message::UserContent {
        match self {
            UserContent::Text { text } => message::UserContent::text(text),
            UserContent::ImageUrl { image_url } => message::UserContent::image(
                image_url.url,
                Some(message::ContentFormat::String),
                None,
                None,
            ),
        }
    }
}

impl TryFrom<message::UserContent> for UserContent {
    type Error = message::MessageError;

    fn try_from(content: message::UserContent) -> Result<Self, Self::Error> {
        match content {
            message::UserContent::Text(text) => Ok(UserContent::Text { text: text.text }),
            message::UserContent::Image(message::Image { data, format, .. }) => match format {
                Some(format) => match format {
                    message::ContentFormat::String => Ok(UserContent::ImageUrl {
                        image_url: ImageUrl { url: data },
                    }),
                    _ => Err(message::MessageError::ConversionError(
                        "Huggingface only supports images as urls".into(),
                    )),
                },
                None => Err(message::MessageError::ConversionError(
                    "Huggingface only supports images as urls".into(),
                )),
            },
            _ => Err(message::MessageError::ConversionError(
                "Huggingface only supports text and images".into(),
            )),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    System {
        #[serde(deserialize_with = "string_or_one_or_many")]
        content: OneOrMany<SystemContent>,
    },
    User {
        #[serde(deserialize_with = "string_or_one_or_many")]
        content: OneOrMany<UserContent>,
    },
    Assistant {
        #[serde(default, deserialize_with = "json_utils::string_or_vec")]
        content: Vec<AssistantContent>,
        #[serde(default, deserialize_with = "json_utils::null_or_vec")]
        tool_calls: Vec<ToolCall>,
    },
    #[serde(rename = "Tool")]
    ToolResult {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        arguments: Option<serde_json::Value>,
        #[serde(deserialize_with = "string_or_one_or_many")]
        content: OneOrMany<String>,
    },
}

impl Message {
    pub fn system(content: &str) -> Self {
        Message::System {
            content: OneOrMany::one(SystemContent::Text {
                text: content.to_string(),
            }),
        }
    }
}

impl TryFrom<message::Message> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Vec<Message>, Self::Error> {
        match message {
            message::Message::User { content } => {
                let (tool_results, other_content): (Vec<_>, Vec<_>) = content
                    .into_iter()
                    .partition(|content| matches!(content, message::UserContent::ToolResult(_)));

                if !tool_results.is_empty() {
                    tool_results
                        .into_iter()
                        .map(|content| match content {
                            message::UserContent::ToolResult(message::ToolResult {
                                id,
                                content,
                            }) => Ok::<_, message::MessageError>(Message::ToolResult {
                                name: id,
                                arguments: None,
                                content: content.try_map(|content| match content {
                                    message::ToolResultContent::Text(message::Text { text }) => {
                                        Ok(text.into())
                                    }
                                    _ => Err(message::MessageError::ConversionError(
                                        "Tool result content does not support non-text".into(),
                                    )),
                                })?,
                            }),
                            _ => unreachable!(),
                        })
                        .collect::<Result<Vec<_>, _>>()
                } else {
                    let other_content = OneOrMany::many(other_content).expect(
                        "There must be other content here if there were no tool result content",
                    );

                    Ok(vec![Message::User {
                        content: other_content.try_map(|content| match content {
                            message::UserContent::Text(text) => {
                                Ok(UserContent::Text { text: text.text })
                            }
                            _ => Err(message::MessageError::ConversionError(
                                "Huggingface does not support non-text".into(),
                            )),
                        })?,
                    }])
                }
            }
            message::Message::Assistant { content } => {
                let (text_content, tool_calls) = content.into_iter().fold(
                    (Vec::new(), Vec::new()),
                    |(mut texts, mut tools), content| {
                        match content {
                            message::AssistantContent::Text(text) => texts.push(text),
                            message::AssistantContent::ToolCall(tool_call) => tools.push(tool_call),
                        }
                        (texts, tools)
                    },
                );

                // `OneOrMany` ensures at least one `AssistantContent::Text` or `ToolCall` exists,
                //  so either `content` or `tool_calls` will have some content.
                Ok(vec![Message::Assistant {
                    content: text_content
                        .into_iter()
                        .map(|content| AssistantContent::Text { text: content.text })
                        .collect::<Vec<_>>(),
                    tool_calls: tool_calls
                        .into_iter()
                        .map(|tool_call| tool_call.into())
                        .collect::<Vec<_>>(),
                }])
            }
        }
    }
}

impl TryFrom<Message> for message::Message {
    type Error = message::MessageError;

    fn try_from(message: Message) -> Result<Self, Self::Error> {
        Ok(match message {
            Message::User { content, .. } => message::Message::User {
                content: content.map(|content| content.into()),
            },
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut content = content
                    .into_iter()
                    .map(|content| match content {
                        AssistantContent::Text { text } => message::AssistantContent::text(text),
                    })
                    .collect::<Vec<_>>();

                content.extend(
                    tool_calls
                        .into_iter()
                        .map(|tool_call| Ok(message::AssistantContent::ToolCall(tool_call.into())))
                        .collect::<Result<Vec<_>, _>>()?,
                );

                message::Message::Assistant {
                    content: OneOrMany::many(content).map_err(|_| {
                        message::MessageError::ConversionError(
                            "Neither `content` nor `tool_calls` was provided to the Message"
                                .to_owned(),
                        )
                    })?,
                }
            }

            Message::ToolResult { name, content, .. } => message::Message::User {
                content: OneOrMany::one(message::UserContent::tool_result(
                    name,
                    content.map(|content| message::ToolResultContent::text(content)),
                )),
            },

            // System messages should get stripped out when converting message's, this is just a
            // stop gap to avoid obnoxious error handling or panic occurring.
            Message::System { content, .. } => message::Message::User {
                content: content.map(|c| match c {
                    SystemContent::Text { text } => message::UserContent::text(text),
                }),
            },
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub finish_reason: String,
    pub index: usize,
    pub logprobs: serde_json::Value,
    pub message: Message,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Usage {
    pub completion_tokens: i32,
    pub prompt_tokens: i32,
    pub total_tokens: i32,
}

#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub created: i32,
    pub id: String,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub system_fingerprint: String,
    pub usage: Usage,
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
                let mut content = content
                    .iter()
                    .map(|c| match c {
                        AssistantContent::Text { text } => message::AssistantContent::text(text),
                    })
                    .collect::<Vec<_>>();

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

#[derive(Clone)]
pub struct CompletionModel {
    pub(crate) client: Client,
    /// Name of the model (e.g: google/gemma-2-2b-it)
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }

    pub(crate) fn create_request_body(
        &self,
        completion_request: &CompletionRequest,
    ) -> Result<serde_json::Value, CompletionError> {
        let mut full_history: Vec<Message> = match &completion_request.preamble {
            Some(preamble) => vec![Message::system(preamble)],
            None => vec![],
        };

        let prompt: Vec<Message> = completion_request.prompt_with_context().try_into()?;

        let chat_history: Vec<Message> = completion_request
            .chat_history
            .clone()
            .into_iter()
            .map(|message| message.try_into())
            .collect::<Result<Vec<Vec<Message>>, _>>()?
            .into_iter()
            .flatten()
            .collect();

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
                "tools": completion_request.tools.clone().into_iter().map(ToolDefinition::from).collect::<Vec<_>>(),
                "tool_choice": "auto",
            })
        };
        Ok(request)
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let request = self.create_request_body(&completion_request)?;


        let path = match self.client.sub_provider {
            SubProvider::HFInference => format!("/{}/v1/chat/completions", self.model),
            _ => "/v1/chat/completions".to_string()
        };

        let request = if let Some(ref params) = completion_request.additional_params {
            json_utils::merge(request, params.clone())
        } else {
            request
        };

        let response = self
            .client
            .post(&path)
            .json(
                &request,
            )
            .send()
            .await?;

        if response.status().is_success() {
            let t = response.text().await?;
            tracing::debug!(target: "rig", "Huggingface completion error: {}", t);

            match serde_json::from_str::<ApiResponse<CompletionResponse>>(&t)? {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "Huggingface completion token usage: {:?}",
                        format!("{:?}", response.usage)
                    );
                    response.try_into()
                }
                ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.to_string())),
            }
        } else {
            Err(CompletionError::ProviderError(format!(
                "{}: {}",
                response.status(),
                response.text().await?
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_path_to_error::deserialize;

    #[test]
    fn test_deserialize_message() {
        let assistant_message_json = r#"
        {
            "role": "assistant",
            "content": "\n\nHello there, how may I assist you today?"
        }
        "#;

        let assistant_message_json2 = r#"
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "\n\nHello there, how may I assist you today?"
                }
            ],
            "tool_calls": null
        }
        "#;

        let assistant_message_json3 = r#"
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_h89ipqYUjEpCPI6SxspMnoUU",
                    "type": "function",
                    "function": {
                        "name": "subtract",
                        "arguments": "{\"x\": 2, \"y\": 5}"
                    }
                }
            ],
            "content": null,
            "refusal": null
        }
        "#;

        let user_message_json = r#"
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    }
                }
            ]
        }
        "#;

        let assistant_message: Message = {
            let jd = &mut serde_json::Deserializer::from_str(assistant_message_json);
            deserialize(jd).unwrap_or_else(|err| {
                panic!(
                    "Deserialization error at {} ({}:{}): {}",
                    err.path(),
                    err.inner().line(),
                    err.inner().column(),
                    err
                );
            })
        };

        let assistant_message2: Message = {
            let jd = &mut serde_json::Deserializer::from_str(assistant_message_json2);
            deserialize(jd).unwrap_or_else(|err| {
                panic!(
                    "Deserialization error at {} ({}:{}): {}",
                    err.path(),
                    err.inner().line(),
                    err.inner().column(),
                    err
                );
            })
        };

        let assistant_message3: Message = {
            let jd: &mut serde_json::Deserializer<serde_json::de::StrRead<'_>> =
                &mut serde_json::Deserializer::from_str(assistant_message_json3);
            deserialize(jd).unwrap_or_else(|err| {
                panic!(
                    "Deserialization error at {} ({}:{}): {}",
                    err.path(),
                    err.inner().line(),
                    err.inner().column(),
                    err
                );
            })
        };

        let _user_message: Message = {
            let jd = &mut serde_json::Deserializer::from_str(user_message_json);
            deserialize(jd).unwrap_or_else(|err| {
                panic!(
                    "Deserialization error at {} ({}:{}): {}",
                    err.path(),
                    err.inner().line(),
                    err.inner().column(),
                    err
                );
            })
        };

        match assistant_message {
            Message::Assistant { content, .. } => {
                assert_eq!(
                    content[0],
                    AssistantContent::Text {
                        text: "\n\nHello there, how may I assist you today?".to_string()
                    }
                );
            }
            _ => panic!("Expected assistant message"),
        }

        match assistant_message2 {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                assert_eq!(
                    content[0],
                    AssistantContent::Text {
                        text: "\n\nHello there, how may I assist you today?".to_string()
                    }
                );

                assert_eq!(tool_calls, vec![]);
            }
            _ => panic!("Expected assistant message"),
        }

        match assistant_message3 {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                assert!(content.is_empty());
                assert_eq!(
                    tool_calls[0],
                    ToolCall {
                        id: "call_h89ipqYUjEpCPI6SxspMnoUU".to_string(),
                        r#type: ToolType::Function,
                        function: Function {
                            name: "subtract".to_string(),
                            arguments: serde_json::json!({"x": 2, "y": 5}),
                        },
                    }
                );
            }
            _ => panic!("Expected assistant message"),
        }

        // match user_message {
        //     Message::User { content, .. } => {
        //         let (first, second) = {
        //             let mut iter = content.into_iter();
        //             (iter.next().unwrap(), iter.next().unwrap())
        //         };
        //         assert_eq!(first, "What's in this image?".to_string());
        //         assert_eq!(second, UserContent::Image { image_url: ImageUrl { url: "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg".to_string(), detail: ImageDetail::default() } });
        //     }
        //     _ => panic!("Expected user message"),
        // }
    }

    #[test]
    fn test_message_to_message_conversion() {
        let user_message = message::Message::User {
            content: OneOrMany::one(message::UserContent::text("Hello")),
        };

        let assistant_message = message::Message::Assistant {
            content: OneOrMany::one(message::AssistantContent::text("Hi there!")),
        };

        let converted_user_message: Vec<Message> = user_message.clone().try_into().unwrap();
        let converted_assistant_message: Vec<Message> =
            assistant_message.clone().try_into().unwrap();

        match converted_user_message[0].clone() {
            Message::User { content, .. } => {
                assert_eq!(
                    content.first(),
                    UserContent::Text {
                        text: "Hello".to_string()
                    }
                );
            }
            _ => panic!("Expected user message"),
        }

        match converted_assistant_message[0].clone() {
            Message::Assistant { content, .. } => {
                assert_eq!(
                    content[0],
                    AssistantContent::Text {
                        text: "Hi there!".to_string()
                    }
                );
            }
            _ => panic!("Expected assistant message"),
        }

        let original_user_message: message::Message =
            converted_user_message[0].clone().try_into().unwrap();
        let original_assistant_message: message::Message =
            converted_assistant_message[0].clone().try_into().unwrap();

        assert_eq!(original_user_message, user_message);
        assert_eq!(original_assistant_message, assistant_message);
    }

    #[test]
    fn test_message_from_message_conversion() {
        let user_message = Message::User {
            content: OneOrMany::one(UserContent::Text {
                text: "Hello".to_string(),
            }),
        };

        let assistant_message = Message::Assistant {
            content: vec![AssistantContent::Text {
                text: "Hi there!".to_string(),
            }],
            tool_calls: vec![],
        };

        let converted_user_message: message::Message = user_message.clone().try_into().unwrap();
        let converted_assistant_message: message::Message =
            assistant_message.clone().try_into().unwrap();

        match converted_user_message.clone() {
            message::Message::User { content } => {
                assert_eq!(content.first(), message::UserContent::text("Hello"));
            }
            _ => panic!("Expected user message"),
        }

        match converted_assistant_message.clone() {
            message::Message::Assistant { content } => {
                assert_eq!(
                    content.first(),
                    message::AssistantContent::text("Hi there!")
                );
            }
            _ => panic!("Expected assistant message"),
        }

        let original_user_message: Vec<Message> = converted_user_message.try_into().unwrap();
        let original_assistant_message: Vec<Message> =
            converted_assistant_message.try_into().unwrap();

        assert_eq!(original_user_message[0], user_message);
        assert_eq!(original_assistant_message[0], assistant_message);
    }
}
