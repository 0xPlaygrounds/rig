use std::collections::HashMap;

use crate::{
    completion::{self, CompletionError},
    json_utils, message, OneOrMany,
};

use super::client::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub finish_reason: FinishReason,
    message: Message,
    #[serde(default)]
    pub usage: Option<Usage>,
}

impl CompletionResponse {
    /// Return that parts of the response for assistant messages w/o dealing with the other variants
    pub fn message(&self) -> (Vec<Content>, Vec<Citation>, Vec<ToolCall>) {
        let Message::Assistant {
            content,
            citations,
            tool_calls,
            ..
        } = self.message.clone()
        else {
            unreachable!("Completion responses will only return an assistant message")
        };

        return (content, citations, tool_calls);
    }
}

#[derive(Debug, Deserialize, PartialEq, Eq, Clone)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinishReason {
    MaxTokens,
    StopSequence,
    Complete,
    Error,
    ToolCall,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Usage {
    #[serde(default)]
    pub billed_units: Option<BilledUnits>,
    #[serde(default)]
    pub tokens: Option<Tokens>,
}

#[derive(Debug, Deserialize, Clone)]
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

#[derive(Debug, Deserialize, Clone)]
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
                Content::Text { text } => completion::AssistantContent::text(text),
            }))
            .map_err(|_| {
                CompletionError::ResponseError(
                    "Response contained no message or tool call (empty)".to_owned(),
                )
            })?
        };

        Ok(completion::CompletionResponse {
            choice: OneOrMany::many(model_response).expect("There is atleast one content"),
            raw_response: response,
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct Document {
    pub id: String,
    pub data: HashMap<String, serde_json::Value>,
}

impl From<completion::Document> for Document {
    fn from(document: completion::Document) -> Self {
        let mut data: HashMap<String, serde_json::Value> = HashMap::new();

        // We use `.into()` here explictely since the `document.additional_props` type will likely
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

#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    pub text: String,
    pub generation_id: String,
}

#[derive(Debug, Deserialize)]
pub struct SearchResult {
    pub search_query: SearchQuery,
    pub connector: Connector,
    pub document_ids: Vec<String>,
    #[serde(default)]
    pub error_message: Option<String>,
    #[serde(default)]
    pub continue_on_failure: bool,
}

#[derive(Debug, Deserialize)]
pub struct Connector {
    pub id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolCall {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub r#type: Option<ToolType>,
    #[serde(default)]
    pub function: Option<ToolCallFunction>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolCallFunction {
    pub name: String,
    #[serde(with = "json_utils::stringified_json")]
    pub arguments: serde_json::Value,
}

#[derive(Clone, Default, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Tool {
    pub r#type: ToolType,
    pub function: Function,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
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

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    User {
        content: OneOrMany<Content>,
    },

    Assistant {
        #[serde(default)]
        content: Vec<Content>,
        #[serde(default)]
        citations: Vec<Citation>,
        #[serde(default)]
        tool_calls: Vec<ToolCall>,
        #[serde(default)]
        tool_plan: Option<String>,
    },

    Tool {
        tool_results: Vec<ToolResult>,
    },

    System {
        content: String,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Content {
    Text { text: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CitationType {
    TextContent,
    Plan,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolResult {
    pub call: ToolCall,
    pub outputs: Vec<serde_json::Value>,
}

impl TryFrom<message::Message> for Message {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        Ok(match message {
            message::Message::User { content } => Message::User {
                content: content.try_map(|content| match content {
                    message::UserContent::Text(message::Text { text }) => {
                        Ok(Content::Text { text })
                    }
                    _ => Err(message::MessageError::ConversionError(
                        "Only text content is supported by Cohere".to_owned(),
                    )),
                })?,
            },
            _ => Err(message::MessageError::ConversionError(
                "Only user messages are supported by Cohere".to_owned(),
            ))?,
        })
    }
}

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let prompt = completion_request.prompt_with_context();

        let mut messages: Vec<message::Message> =
            if let Some(preamble) = completion_request.preamble {
                vec![preamble.into()]
            } else {
                vec![]
            };

        messages.extend(completion_request.chat_history);
        messages.push(prompt);

        let messages: Vec<Message> = messages
            .into_iter()
            .map(Message::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        let request = json!({
            "model": self.model,
            "messages": messages,
            "documents": completion_request.documents,
            "temperature": completion_request.temperature,
            "tools": completion_request.tools.into_iter().map(Tool::from).collect::<Vec<_>>(),
        });

        tracing::debug!("Cohere request: {}", serde_json::to_string_pretty(&request)?);

        let response = self
            .client
            .post("/v2/chat")
            .json(
                &if let Some(ref params) = completion_request.additional_params {
                    json_utils::merge(request.clone(), params.clone())
                } else {
                    request.clone()
                },
            )
            .send()
            .await?;

        if response.status().is_success() {
            let text_response = response.text().await?;
            tracing::debug!("Cohere response text: {}", text_response);

            let json_response: CompletionResponse = serde_json::from_str(&text_response)?;
            let completion: completion::CompletionResponse<CompletionResponse> =
                json_response.try_into()?;
            Ok(completion)
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
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
            "id": "d007e969-af58-4da4-beb1-c30454c3b75f",
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

        assert_eq!(id, "7874d629-11e5-4f13-8a25-32ecfe04aee3");
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
        assert_eq!(billed_output_tokens.unwrap(), 25.0);
        assert_eq!(input_tokens.unwrap(), 1028.0);
        assert_eq!(output_tokens.unwrap(), 61.0);

        assert!(citations.is_empty());
        assert_eq!(tool_calls.len(), 1);

        let ToolCallFunction { name, arguments } = tool_calls[0].function.clone().unwrap();

        assert_eq!(name, "subtract");
        assert_eq!(arguments, serde_json::json!({"x": 2, "y": 5}));
    }
}
