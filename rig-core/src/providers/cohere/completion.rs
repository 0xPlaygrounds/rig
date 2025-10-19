use crate::{
    OneOrMany,
    completion::{self, CompletionError, GetTokenUsage},
    http_client::{self, HttpClientExt},
    json_utils,
    message::{self, Reasoning, ToolChoice},
    telemetry::SpanCombinator,
};
use std::collections::HashMap;

use super::client::Client;
use crate::completion::CompletionRequest;
use crate::providers::cohere::streaming::StreamingCompletionResponse;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::{Instrument, info_span};

#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub finish_reason: FinishReason,
    message: Message,
    #[serde(default)]
    pub usage: Option<Usage>,
}

impl CompletionResponse {
    /// Return that parts of the response for assistant messages w/o dealing with the other variants
    pub fn message(&self) -> (Vec<AssistantContent>, Vec<Citation>, Vec<ToolCall>) {
        let Message::Assistant {
            content,
            citations,
            tool_calls,
            ..
        } = self.message.clone()
        else {
            unreachable!("Completion responses will only return an assistant message")
        };

        (content, citations, tool_calls)
    }
}

impl crate::telemetry::ProviderResponseExt for CompletionResponse {
    type OutputMessage = Message;
    type Usage = Usage;

    fn get_response_id(&self) -> Option<String> {
        Some(self.id.clone())
    }

    fn get_response_model_name(&self) -> Option<String> {
        None
    }

    fn get_output_messages(&self) -> Vec<Self::OutputMessage> {
        vec![self.message.clone()]
    }

    fn get_text_response(&self) -> Option<String> {
        let Message::Assistant { ref content, .. } = self.message else {
            return None;
        };

        let res = content
            .iter()
            .filter_map(|x| {
                if let AssistantContent::Text { text } = x {
                    Some(text.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<String>>()
            .join("\n");

        if res.is_empty() { None } else { Some(res) }
    }

    fn get_usage(&self) -> Option<Self::Usage> {
        self.usage.clone()
    }
}

#[derive(Debug, Deserialize, PartialEq, Eq, Clone, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinishReason {
    MaxTokens,
    StopSequence,
    Complete,
    Error,
    ToolCall,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct Usage {
    #[serde(default)]
    pub billed_units: Option<BilledUnits>,
    #[serde(default)]
    pub tokens: Option<Tokens>,
}

impl GetTokenUsage for Usage {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();

        if let Some(ref billed_units) = self.billed_units {
            usage.input_tokens = billed_units.input_tokens.unwrap_or_default() as u64;
            usage.output_tokens = billed_units.output_tokens.unwrap_or_default() as u64;
            usage.total_tokens = usage.input_tokens + usage.output_tokens;
        }

        Some(usage)
    }
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct BilledUnits {
    #[serde(default)]
    pub output_tokens: Option<f64>,
    #[serde(default)]
    pub classifications: Option<f64>,
    #[serde(default)]
    pub search_units: Option<f64>,
    #[serde(default)]
    pub input_tokens: Option<f64>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct Tokens {
    #[serde(default)]
    pub input_tokens: Option<f64>,
    #[serde(default)]
    pub output_tokens: Option<f64>,
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let (content, _, tool_calls) = response.message();

        let model_response = if !tool_calls.is_empty() {
            OneOrMany::many(
                tool_calls
                    .into_iter()
                    .filter_map(|tool_call| {
                        let ToolCallFunction { name, arguments } = tool_call.function?;
                        let id = tool_call.id.unwrap_or_else(|| name.clone());

                        Some(completion::AssistantContent::tool_call(id, name, arguments))
                    })
                    .collect::<Vec<_>>(),
            )
            .expect("We have atleast 1 tool call in this if block")
        } else {
            OneOrMany::many(content.into_iter().map(|content| match content {
                AssistantContent::Text { text } => completion::AssistantContent::text(text),
                AssistantContent::Thinking { thinking } => {
                    completion::AssistantContent::Reasoning(Reasoning {
                        id: None,
                        reasoning: vec![thinking],
                        signature: None,
                    })
                }
            }))
            .map_err(|_| {
                CompletionError::ResponseError(
                    "Response contained no message or tool call (empty)".to_owned(),
                )
            })?
        };

        let usage = response
            .usage
            .as_ref()
            .and_then(|usage| usage.tokens.as_ref())
            .map(|tokens| {
                let input_tokens = tokens.input_tokens.unwrap_or(0.0);
                let output_tokens = tokens.output_tokens.unwrap_or(0.0);

                completion::Usage {
                    input_tokens: input_tokens as u64,
                    output_tokens: output_tokens as u64,
                    total_tokens: (input_tokens + output_tokens) as u64,
                }
            })
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice: OneOrMany::many(model_response).expect("There is atleast one content"),
            usage,
            raw_response: response,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Document {
    pub id: String,
    pub data: HashMap<String, serde_json::Value>,
}

impl From<completion::Document> for Document {
    fn from(document: completion::Document) -> Self {
        let mut data: HashMap<String, serde_json::Value> = HashMap::new();

        // We use `.into()` here explicitly since the `document.additional_props` type will likely
        //  evolve into `serde_json::Value` in the future.
        document
            .additional_props
            .into_iter()
            .for_each(|(key, value)| {
                data.insert(key, value.into());
            });

        data.insert("text".to_string(), document.text.into());

        Self {
            id: document.id,
            data,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ToolCall {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub r#type: Option<ToolType>,
    #[serde(default)]
    pub function: Option<ToolCallFunction>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ToolCallFunction {
    pub name: String,
    #[serde(with = "json_utils::stringified_json")]
    pub arguments: serde_json::Value,
}

#[derive(Clone, Default, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Tool {
    pub r#type: ToolType,
    pub function: Function,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

impl From<completion::ToolDefinition> for Tool {
    fn from(tool: completion::ToolDefinition) -> Self {
        Self {
            r#type: ToolType::default(),
            function: Function {
                name: tool.name,
                description: Some(tool.description),
                parameters: tool.parameters,
            },
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    User {
        content: OneOrMany<UserContent>,
    },

    Assistant {
        #[serde(default)]
        content: Vec<AssistantContent>,
        #[serde(default)]
        citations: Vec<Citation>,
        #[serde(default)]
        tool_calls: Vec<ToolCall>,
        #[serde(default)]
        tool_plan: Option<String>,
    },

    Tool {
        content: OneOrMany<ToolResultContent>,
        tool_call_id: String,
    },

    System {
        content: String,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContent {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AssistantContent {
    Text { text: String },
    Thinking { thinking: String },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub enum ToolResultContent {
    Text { text: String },
    Document { document: Document },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Citation {
    #[serde(default)]
    pub start: Option<u32>,
    #[serde(default)]
    pub end: Option<u32>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(rename = "type")]
    pub citation_type: Option<CitationType>,
    #[serde(default)]
    pub sources: Vec<Source>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Source {
    Document {
        id: Option<String>,
        document: Option<serde_json::Map<String, serde_json::Value>>,
    },
    Tool {
        id: Option<String>,
        tool_output: Option<serde_json::Map<String, serde_json::Value>>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CitationType {
    TextContent,
    Plan,
}

impl TryFrom<message::Message> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        Ok(match message {
            message::Message::User { content } => content
                .into_iter()
                .map(|content| match content {
                    message::UserContent::Text(message::Text { text }) => Ok(Message::User {
                        content: OneOrMany::one(UserContent::Text { text }),
                    }),
                    message::UserContent::ToolResult(message::ToolResult {
                        id, content, ..
                    }) => Ok(Message::Tool {
                        tool_call_id: id,
                        content: content.try_map(|content| match content {
                            message::ToolResultContent::Text(text) => {
                                Ok(ToolResultContent::Text { text: text.text })
                            }
                            _ => Err(message::MessageError::ConversionError(
                                "Only text tool result content is supported by Cohere".to_owned(),
                            )),
                        })?,
                    }),
                    _ => Err(message::MessageError::ConversionError(
                        "Only text content is supported by Cohere".to_owned(),
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?,
            message::Message::Assistant { content, .. } => {
                let mut text_content = vec![];
                let mut tool_calls = vec![];
                content.into_iter().for_each(|content| match content {
                    message::AssistantContent::Text(message::Text { text }) => {
                        text_content.push(AssistantContent::Text { text });
                    }
                    message::AssistantContent::ToolCall(message::ToolCall {
                        id,
                        function:
                            message::ToolFunction {
                                name, arguments, ..
                            },
                        ..
                    }) => {
                        tool_calls.push(ToolCall {
                            id: Some(id),
                            r#type: Some(ToolType::Function),
                            function: Some(ToolCallFunction {
                                name,
                                arguments: serde_json::to_value(arguments).unwrap_or_default(),
                            }),
                        });
                    }
                    message::AssistantContent::Reasoning(Reasoning { reasoning, .. }) => {
                        let thinking = reasoning.join("\n");
                        text_content.push(AssistantContent::Thinking { thinking });
                    }
                });

                vec![Message::Assistant {
                    content: text_content,
                    citations: vec![],
                    tool_calls,
                    tool_plan: None,
                }]
            }
        })
    }
}

impl TryFrom<Message> for message::Message {
    type Error = message::MessageError;

    fn try_from(message: Message) -> Result<Self, Self::Error> {
        match message {
            Message::User { content } => Ok(message::Message::User {
                content: content.map(|content| match content {
                    UserContent::Text { text } => {
                        message::UserContent::Text(message::Text { text })
                    }
                    UserContent::ImageUrl { image_url } => {
                        message::UserContent::image_url(image_url.url, None, None)
                    }
                }),
            }),
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut content = content
                    .into_iter()
                    .map(|content| match content {
                        AssistantContent::Text { text } => message::AssistantContent::text(text),
                        AssistantContent::Thinking { thinking } => {
                            message::AssistantContent::Reasoning(Reasoning {
                                id: None,
                                reasoning: vec![thinking],
                                signature: None,
                            })
                        }
                    })
                    .collect::<Vec<_>>();

                content.extend(tool_calls.into_iter().filter_map(|tool_call| {
                    let ToolCallFunction { name, arguments } = tool_call.function?;

                    Some(message::AssistantContent::tool_call(
                        tool_call.id.unwrap_or_else(|| name.clone()),
                        name,
                        arguments,
                    ))
                }));

                let content = OneOrMany::many(content).map_err(|_| {
                    message::MessageError::ConversionError(
                        "Expected either text content or tool calls".to_string(),
                    )
                })?;

                Ok(message::Message::Assistant { id: None, content })
            }
            Message::Tool {
                content,
                tool_call_id,
            } => {
                let content = content.try_map(|content| {
                    Ok(match content {
                        ToolResultContent::Text { text } => message::ToolResultContent::text(text),
                        ToolResultContent::Document { document } => {
                            message::ToolResultContent::text(
                                serde_json::to_string(&document.data).map_err(|e| {
                                    message::MessageError::ConversionError(
                                        format!("Failed to convert tool result document content into text: {e}"),
                                    )
                                })?,
                            )
                        }
                    })
                })?;

                Ok(message::Message::User {
                    content: OneOrMany::one(message::UserContent::tool_result(
                        tool_call_id,
                        content,
                    )),
                })
            }
            Message::System { content } => Ok(message::Message::user(content)),
        }
    }
}

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    pub(crate) client: Client<T>,
    pub model: String,
}

impl<T> CompletionModel<T>
where
    T: HttpClientExt,
{
    pub fn new(client: Client<T>, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }

    pub(crate) fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<Value, CompletionError> {
        // Build up the order of messages (context, chat_history)
        let mut partial_history = vec![];
        if let Some(docs) = completion_request.normalized_documents() {
            partial_history.push(docs);
        }
        partial_history.extend(completion_request.chat_history);

        // Initialize full history with preamble (or empty if non-existent)
        let mut full_history: Vec<Message> = completion_request
            .preamble
            .map_or_else(Vec::new, |preamble| {
                vec![Message::System { content: preamble }]
            });

        // Convert and extend the rest of the history
        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Vec<Message>>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );

        let request = json!({
            "model": self.model,
            "messages": full_history,
            "documents": completion_request.documents,
            "temperature": completion_request.temperature,
            "tools": completion_request.tools.into_iter().map(Tool::from).collect::<Vec<_>>(),
            "tool_choice": if let Some(tool_choice) = completion_request.tool_choice && !matches!(tool_choice, ToolChoice::Auto) { tool_choice } else {
                return Err(CompletionError::RequestError("\"auto\" is not an allowed tool_choice value in the Cohere API".into()))
            },
        });

        if let Some(ref params) = completion_request.additional_params {
            Ok(json_utils::merge(request.clone(), params.clone()))
        } else {
            Ok(request)
        }
    }
}

impl completion::CompletionModel for CompletionModel<reqwest::Client> {
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let request = self.create_completion_request(completion_request)?;

        let llm_span = if tracing::Span::current().is_disabled() {
            info_span!(
            target: "rig::completions",
            "chat",
            gen_ai.operation.name = "chat",
            gen_ai.provider.name = "cohere",
            gen_ai.request.model = self.model,
            gen_ai.response.id = tracing::field::Empty,
            gen_ai.response.model = self.model,
            gen_ai.usage.output_tokens = tracing::field::Empty,
            gen_ai.usage.input_tokens = tracing::field::Empty,
            gen_ai.input.messages = serde_json::to_string(request.get("messages").expect("Converting request messages to JSON should not fail!")).unwrap(),
            gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        tracing::debug!(
            "Cohere request: {}",
            serde_json::to_string_pretty(&request)?
        );

        async {
            let response = self
                .client
                .client()
                .post("/v2/chat")
                .json(&request)
                .send()
                .await
                .map_err(|e| http_client::Error::Instance(e.into()))?;

            if response.status().is_success() {
                let text_response = response.text().await.map_err(|e| {
                    CompletionError::HttpError(http_client::Error::Instance(e.into()))
                })?;
                tracing::debug!("Cohere completion request: {}", text_response);

                let json_response: CompletionResponse = serde_json::from_str(&text_response)?;
                let span = tracing::Span::current();
                span.record_token_usage(&json_response.usage);
                span.record_model_output(&json_response.message);
                span.record_response_metadata(&json_response);
                tracing::debug!("Cohere completion response: {}", text_response);
                let completion: completion::CompletionResponse<CompletionResponse> =
                    json_response.try_into()?;
                Ok(completion)
            } else {
                Err(CompletionError::ProviderError(
                    response.text().await.map_err(|e| {
                        CompletionError::HttpError(http_client::Error::Instance(e.into()))
                    })?,
                ))
            }
        }
        .instrument(llm_span)
        .await
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        CompletionModel::stream(self, request).await
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_path_to_error::deserialize;

    #[test]
    fn test_deserialize_completion_response() {
        let json_data = r#"
        {
            "id": "abc123",
            "message": {
                "role": "assistant",
                "tool_plan": "I will use the subtract tool to find the difference between 2 and 5.",
                "tool_calls": [
                        {
                            "id": "subtract_sm6ps6fb6y9f",
                            "type": "function",
                            "function": {
                                "name": "subtract",
                                "arguments": "{\"x\":5,\"y\":2}"
                            }
                        }
                    ]
                },
                "finish_reason": "TOOL_CALL",
                "usage": {
                "billed_units": {
                    "input_tokens": 78,
                    "output_tokens": 27
                },
                "tokens": {
                    "input_tokens": 1028,
                    "output_tokens": 63
                }
            }
        }
        "#;

        let mut deserializer = serde_json::Deserializer::from_str(json_data);
        let result: Result<CompletionResponse, _> = deserialize(&mut deserializer);

        let response = result.unwrap();
        let (_, citations, tool_calls) = response.message();
        let CompletionResponse {
            id,
            finish_reason,
            usage,
            ..
        } = response;

        assert_eq!(id, "abc123");
        assert_eq!(finish_reason, FinishReason::ToolCall);

        let Usage {
            billed_units,
            tokens,
        } = usage.unwrap();
        let BilledUnits {
            input_tokens: billed_input_tokens,
            output_tokens: billed_output_tokens,
            ..
        } = billed_units.unwrap();
        let Tokens {
            input_tokens,
            output_tokens,
        } = tokens.unwrap();

        assert_eq!(billed_input_tokens.unwrap(), 78.0);
        assert_eq!(billed_output_tokens.unwrap(), 27.0);
        assert_eq!(input_tokens.unwrap(), 1028.0);
        assert_eq!(output_tokens.unwrap(), 63.0);

        assert!(citations.is_empty());
        assert_eq!(tool_calls.len(), 1);

        let ToolCallFunction { name, arguments } = tool_calls[0].function.clone().unwrap();

        assert_eq!(name, "subtract");
        assert_eq!(arguments, serde_json::json!({"x": 5, "y": 2}));
    }

    #[test]
    fn test_convert_completion_message_to_message_and_back() {
        let completion_message = completion::Message::User {
            content: OneOrMany::one(completion::message::UserContent::Text(
                completion::message::Text {
                    text: "Hello, world!".to_string(),
                },
            )),
        };

        let messages: Vec<Message> = completion_message.clone().try_into().unwrap();
        let _converted_back: Vec<completion::Message> = messages
            .into_iter()
            .map(|msg| msg.try_into().unwrap())
            .collect::<Vec<_>>();
    }

    #[test]
    fn test_convert_message_to_completion_message_and_back() {
        let message = Message::User {
            content: OneOrMany::one(UserContent::Text {
                text: "Hello, world!".to_string(),
            }),
        };

        let completion_message: completion::Message = message.clone().try_into().unwrap();
        let _converted_back: Vec<Message> = completion_message.try_into().unwrap();
    }
}
