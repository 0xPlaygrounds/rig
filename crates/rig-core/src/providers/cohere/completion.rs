use crate::{
    OneOrMany,
    completion::{self, CompletionError, GetTokenUsage},
    http_client::HttpClientExt,
    json_utils,
    message::{self, Reasoning, ToolChoice},
    telemetry::{CompletionOperation, CompletionSpanBuilder, SpanCombinator},
};
use std::collections::HashMap;

use super::client::Client;
use crate::completion::CompletionRequest;
use crate::providers::cohere::streaming::StreamingCompletionResponse;
use serde::{Deserialize, Serialize};
use tracing::{Instrument, Level, enabled};

#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub finish_reason: FinishReason,
    message: Message,
    #[serde(default)]
    pub usage: Option<Usage>,
}

type AssistantMessageParts = (Vec<AssistantContent>, Vec<Citation>, Vec<ToolCall>);

impl CompletionResponse {
    /// Return that parts of the response for assistant messages w/o dealing with the other variants
    pub fn message(&self) -> Result<AssistantMessageParts, CompletionError> {
        let Message::Assistant {
            content,
            citations,
            tool_calls,
            ..
        } = self.message.clone()
        else {
            return Err(CompletionError::ResponseError(
                "completion response did not contain an assistant message".into(),
            ));
        };

        Ok((content, citations, tool_calls))
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
    #[serde(default)]
    pub cached_tokens: Option<f64>,
}

impl GetTokenUsage for Usage {
    fn token_usage(&self) -> crate::completion::Usage {
        let mut usage = crate::completion::Usage::new();

        if let Some(ref tokens) = self.tokens {
            usage.input_tokens = tokens.input_tokens.unwrap_or_default() as u64;
            usage.output_tokens = tokens.output_tokens.unwrap_or_default() as u64;
            usage.total_tokens = usage.input_tokens + usage.output_tokens;
            usage.cached_input_tokens = self.cached_tokens.unwrap_or_default() as u64;
        }

        usage
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
        let (content, _, tool_calls) = response.message()?;

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
            .map_err(|_| {
                CompletionError::ResponseError(
                    "response contained tool call metadata without any callable tool content"
                        .to_owned(),
                )
            })?
        } else {
            OneOrMany::many(content.into_iter().map(|content| match content {
                AssistantContent::Text { text } => completion::AssistantContent::text(text),
                AssistantContent::Thinking { thinking } => {
                    completion::AssistantContent::Reasoning(Reasoning::new(&thinking))
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
            .map(GetTokenUsage::token_usage)
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice: model_response,
            usage,
            raw_response: response,
            message_id: None,
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
#[serde(tag = "type", rename_all = "lowercase")]
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
                    message::UserContent::Text(message::Text { text, .. }) => Ok(Message::User {
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
                            message::ToolResultContent::Json { value } => {
                                Ok(ToolResultContent::Text {
                                    text: value.to_string(),
                                })
                            }
                            message::ToolResultContent::Image(_) => {
                                Err(message::MessageError::ConversionError(
                                    "Only text tool result content is supported by Cohere"
                                        .to_owned(),
                                ))
                            }
                        })?,
                    }),
                    _ => Err(message::MessageError::ConversionError(
                        "Only text content is supported by Cohere".to_owned(),
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?,
            message::Message::System { content } => {
                vec![Message::System { content }]
            }
            message::Message::Assistant { content, .. } => {
                let mut text_content = vec![];
                let mut tool_calls = vec![];

                for content in content.into_iter() {
                    match content {
                        message::AssistantContent::Text(message::Text { text, .. }) => {
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
                        message::AssistantContent::Reasoning(reasoning) => {
                            let thinking = reasoning.display_text();
                            text_content.push(AssistantContent::Thinking { thinking });
                        }
                        message::AssistantContent::Image(_) => {
                            return Err(message::MessageError::ConversionError(
                                "Cohere currently doesn't support images.".to_owned(),
                            ));
                        }
                    }
                }

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
                        message::UserContent::Text(message::Text::new(text))
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
                            message::AssistantContent::Reasoning(Reasoning::new(&thinking))
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
                            message::ToolResultContent::json(
                                serde_json::to_value(document.data).map_err(|e| {
                                    message::MessageError::ConversionError(
                                        format!("Failed to convert tool result document content into JSON: {e}"),
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

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CohereToolChoice {
    Required,
    None,
}

impl TryFrom<ToolChoice> for CohereToolChoice {
    type Error = CompletionError;

    fn try_from(tool_choice: ToolChoice) -> Result<Self, Self::Error> {
        match tool_choice {
            ToolChoice::Required => Ok(Self::Required),
            ToolChoice::None => Ok(Self::None),
            ToolChoice::Auto => Err(CompletionError::RequestError(
                "\"auto\" is not an allowed tool_choice value in the Cohere API; \
                 omit tool_choice to let the model decide"
                    .into(),
            )),
            ToolChoice::Specific { .. } => Err(CompletionError::RequestError(
                "the Cohere API cannot be forced to call specific tools by name; \
                 use ToolChoice::Required and restrict the tools you pass instead"
                    .into(),
            )),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct CohereCompletionRequest {
    pub(super) model: String,
    pub messages: Vec<Message>,
    documents: Vec<Document>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<CohereToolChoice>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

impl TryFrom<(&str, CompletionRequest)> for CohereCompletionRequest {
    type Error = CompletionError;

    fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        let documents = req
            .documents
            .iter()
            .cloned()
            .map(Document::from)
            .collect::<Vec<_>>();
        if req.output_schema.is_some() {
            tracing::warn!("Structured outputs currently not supported for Cohere");
        }

        let model = req.model.clone().unwrap_or_else(|| model.to_string());
        let mut partial_history = vec![];
        partial_history.extend(req.chat_history);

        let mut full_history: Vec<Message> = req.preamble.map_or_else(Vec::new, |preamble| {
            vec![Message::System { content: preamble }]
        });

        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Vec<Message>>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );

        let tool_choice = req
            .tool_choice
            .map(CohereToolChoice::try_from)
            .transpose()?;

        Ok(Self {
            model: model.to_string(),
            messages: full_history,
            documents,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            tools: req.tools.into_iter().map(Tool::from).collect::<Vec<_>>(),
            tool_choice,
            additional_params: req.additional_params,
        })
    }
}

impl<T> CompletionModel<T>
where
    T: HttpClientExt,
{
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;
    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model.into())
    }

    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let system_instructions = completion_request.preamble.clone();
        let record_telemetry_content = completion_request.record_telemetry_content;
        let request = CohereCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

        let llm_span =
            CompletionSpanBuilder::new("cohere", &request.model, CompletionOperation::Chat)
                .system_instructions(system_instructions.as_deref(), record_telemetry_content)
                .build();

        if enabled!(Level::TRACE) {
            tracing::trace!(
                "Cohere completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let req_body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/v2/chat")?
            .body(req_body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        async {
            let response = self.client.send::<_, bytes::Bytes>(req).await?;

            let status = response.status();
            let body = response.into_body().into_future().await?.to_owned();

            if status.is_success() {
                let json_response: CompletionResponse = serde_json::from_slice(&body)?;
                let span = tracing::Span::current();
                span.record_token_usage(&json_response.usage);
                span.record_response_metadata(&json_response);

                if enabled!(Level::TRACE) {
                    tracing::trace!(
                        target: "rig::completions",
                        "Cohere completion response: {}",
                        serde_json::to_string_pretty(&json_response)?
                    );
                }

                let completion: completion::CompletionResponse<CompletionResponse> =
                    json_response.try_into()?;
                Ok(completion)
            } else {
                Err(CompletionError::from_http_response(
                    status,
                    String::from_utf8_lossy(&body),
                ))
            }
        }
        .instrument(llm_span)
        .await
    }

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
        let (_, citations, tool_calls) = response.message().expect("assistant message");
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
            ..
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
                completion::message::Text::new("Hello, world!".to_string()),
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

    #[test]
    fn usage_is_mapped_from_tokens_and_carries_cached_input() {
        let usage: Usage = serde_json::from_str(
            r#"{
                "billed_units": {"input_tokens": 135, "output_tokens": 24},
                "cached_tokens": 112,
                "tokens": {"input_tokens": 1610, "output_tokens": 56}
            }"#,
        )
        .expect("usage should deserialize");

        let mapped = usage.token_usage();
        assert_eq!(mapped.input_tokens, 1610);
        assert_eq!(mapped.output_tokens, 56);
        assert_eq!(mapped.total_tokens, 1666);
        assert_eq!(mapped.cached_input_tokens, 112);
    }

    #[test]
    fn response_usage_matches_the_canonical_mapping() {
        let response: CompletionResponse = serde_json::from_str(
            r#"{
                "id": "abc123",
                "finish_reason": "COMPLETE",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                "usage": {
                    "billed_units": {"input_tokens": 135, "output_tokens": 24},
                    "cached_tokens": 112,
                    "tokens": {"input_tokens": 1610, "output_tokens": 56}
                }
            }"#,
        )
        .expect("response should deserialize");

        let expected = response
            .usage
            .as_ref()
            .expect("usage should be present")
            .token_usage();
        let converted: completion::CompletionResponse<CompletionResponse> =
            response.try_into().expect("response should convert");

        assert_eq!(converted.usage, expected);
        assert_eq!(converted.usage.input_tokens, 1610);
        assert_eq!(converted.usage.cached_input_tokens, 112);
    }

    #[test]
    fn usage_without_token_counts_maps_to_zero() {
        let usage: Usage = serde_json::from_str("{}").expect("usage should deserialize");
        assert_eq!(usage.token_usage(), crate::completion::Usage::new());

        let cached_only: Usage =
            serde_json::from_str(r#"{"cached_tokens": 512}"#).expect("usage should deserialize");
        assert_eq!(cached_only.token_usage(), crate::completion::Usage::new());
    }

    #[test]
    fn tool_result_content_is_type_tagged() {
        let text = serde_json::to_value(ToolResultContent::Text {
            text: "-3".to_owned(),
        })
        .expect("tool result text content should serialize");
        assert_eq!(text, serde_json::json!({"type": "text", "text": "-3"}));

        let document = serde_json::to_value(ToolResultContent::Document {
            document: Document {
                id: "doc_1".to_owned(),
                data: HashMap::from([("text".to_owned(), "-3".into())]),
            },
        })
        .expect("tool result document content should serialize");
        assert_eq!(
            document,
            serde_json::json!({
                "type": "document",
                "document": {"id": "doc_1", "data": {"text": "-3"}}
            })
        );

        let roundtrip: ToolResultContent =
            serde_json::from_value(text).expect("tool result content should deserialize");
        assert_eq!(
            roundtrip,
            ToolResultContent::Text {
                text: "-3".to_owned()
            }
        );
    }

    #[test]
    fn cohere_builder_request_serializes_documents_in_cohere_shape() {
        let request = crate::completion::CompletionRequestBuilder::new(
            crate::test_utils::MockCompletionModel::default(),
            "What is glarb-glarb?",
        )
        .document(crate::completion::request::Document {
            id: "doc_1".to_string(),
            text: "Definition of glarb-glarb: an ancient tool.".to_string(),
            additional_props: HashMap::from([("source".to_string(), "field-notes".to_string())]),
        })
        .build();

        let request = CohereCompletionRequest::try_from(("command-a-03-2025", request))
            .expect("request conversion should succeed");

        assert_eq!(request.documents.len(), 1);
        assert_eq!(request.documents[0].id, "doc_1");

        let documents = serde_json::to_value(&request.documents)
            .expect("documents should serialize")
            .as_array()
            .cloned()
            .expect("documents should serialize as an array");
        assert_eq!(
            documents[0],
            serde_json::json!({
                "id": "doc_1",
                "data": {
                    "text": "Definition of glarb-glarb: an ancient tool.",
                    "source": "field-notes"
                }
            })
        );
    }

    #[test]
    fn tool_choice_serializes_as_a_bare_cohere_string() {
        assert_eq!(
            serde_json::to_value(CohereToolChoice::Required).expect("serialize"),
            serde_json::json!("REQUIRED")
        );
        assert_eq!(
            serde_json::to_value(CohereToolChoice::None).expect("serialize"),
            serde_json::json!("NONE")
        );

        assert_eq!(
            CohereToolChoice::try_from(ToolChoice::Required).expect("required is supported"),
            CohereToolChoice::Required
        );
        assert_eq!(
            CohereToolChoice::try_from(ToolChoice::None).expect("none is supported"),
            CohereToolChoice::None
        );
    }

    #[test]
    fn unsupported_tool_choices_are_rejected_before_the_request_is_sent() {
        for unsupported in [
            ToolChoice::Auto,
            ToolChoice::Specific {
                function_names: vec!["subtract".to_string()],
            },
        ] {
            let error = CohereToolChoice::try_from(unsupported.clone())
                .expect_err("Cohere has no encoding for this tool choice");
            assert!(
                matches!(error, CompletionError::RequestError(_)),
                "expected a request error for {unsupported:?}, got {error:?}"
            );
        }
    }

    #[test]
    fn max_tokens_is_forwarded_and_omitted_when_unset() {
        let capped = crate::completion::CompletionRequestBuilder::new(
            crate::test_utils::MockCompletionModel::default(),
            "hello",
        )
        .max_tokens(64)
        .build();
        let capped = CohereCompletionRequest::try_from(("command-a-03-2025", capped))
            .expect("request conversion should succeed");
        let body = serde_json::to_value(&capped).expect("request should serialize");
        assert_eq!(body["max_tokens"], serde_json::json!(64));

        let uncapped = crate::completion::CompletionRequestBuilder::new(
            crate::test_utils::MockCompletionModel::default(),
            "hello",
        )
        .build();
        let uncapped = CohereCompletionRequest::try_from(("command-a-03-2025", uncapped))
            .expect("request conversion should succeed");
        let body = serde_json::to_value(&uncapped).expect("request should serialize");
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn tool_choice_is_omitted_when_unset() {
        let request = crate::completion::CompletionRequestBuilder::new(
            crate::test_utils::MockCompletionModel::default(),
            "hello",
        )
        .build();

        let request = CohereCompletionRequest::try_from(("command-a-03-2025", request))
            .expect("request conversion should succeed");
        let body = serde_json::to_value(&request).expect("request should serialize");

        assert!(body.get("tool_choice").is_none());
    }

    #[tokio::test]
    async fn completion_non_success_preserves_status_and_body() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel as _;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"boom"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = crate::providers::cohere::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model(crate::providers::cohere::COMMAND_A_03_2025);
        let request = model.completion_request("hello").build();

        let error = model
            .completion(request)
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
