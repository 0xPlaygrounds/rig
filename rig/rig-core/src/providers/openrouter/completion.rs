use super::{
    client::{ApiErrorResponse, ApiResponse, Client, Usage},
    streaming::StreamingCompletionResponse,
};
use crate::message;
use crate::telemetry::SpanCombinator;
use crate::{
    OneOrMany,
    completion::{self, CompletionError, CompletionRequest},
    http_client::HttpClientExt,
    json_utils,
    one_or_many::string_or_one_or_many,
    providers::openai,
};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{Instrument, Level, enabled, info_span};

// ================================================================
// OpenRouter Completion API
// ================================================================

/// The `qwen/qwq-32b` model. Find more models at <https://openrouter.ai/models>.
pub const QWEN_QWQ_32B: &str = "qwen/qwq-32b";
/// The `anthropic/claude-3.7-sonnet` model. Find more models at <https://openrouter.ai/models>.
pub const CLAUDE_3_7_SONNET: &str = "anthropic/claude-3.7-sonnet";
/// The `perplexity/sonar-pro` model. Find more models at <https://openrouter.ai/models>.
pub const PERPLEXITY_SONAR_PRO: &str = "perplexity/sonar-pro";
/// The `google/gemini-2.0-flash-001` model. Find more models at <https://openrouter.ai/models>.
pub const GEMINI_FLASH_2_0: &str = "google/gemini-2.0-flash-001";

/// A openrouter completion object.
///
/// For more information, see this link: <https://docs.openrouter.xyz/reference/create_chat_completion_v1_chat_completions_post>
#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub system_fingerprint: Option<String>,
    pub usage: Option<Usage>,
}

impl From<ApiErrorResponse> for CompletionError {
    fn from(err: ApiErrorResponse) -> Self {
        CompletionError::ProviderError(err.message)
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
                reasoning,
                reasoning_details,
                ..
            } => {
                let mut content = content
                    .iter()
                    .map(|c| match c {
                        openai::AssistantContent::Text { text } => {
                            completion::AssistantContent::text(text)
                        }
                        openai::AssistantContent::Refusal { refusal } => {
                            completion::AssistantContent::text(refusal)
                        }
                    })
                    .collect::<Vec<_>>();

                content.extend(tool_calls.iter().map(|call| {
                    completion::AssistantContent::tool_call(
                        &call.id,
                        &call.function.name,
                        call.function.arguments.clone(),
                    )
                }));

                let mut grouped_reasoning: HashMap<
                    Option<String>,
                    Vec<(usize, usize, message::ReasoningContent)>,
                > = HashMap::new();
                let mut reasoning_order: Vec<Option<String>> = Vec::new();
                for (position, detail) in reasoning_details.iter().enumerate() {
                    let (reasoning_id, sort_index, parsed_content) = match detail {
                        ReasoningDetails::Summary {
                            id, index, summary, ..
                        } => (
                            id.clone(),
                            *index,
                            Some(message::ReasoningContent::Summary(summary.clone())),
                        ),
                        ReasoningDetails::Encrypted {
                            id, index, data, ..
                        } => (
                            id.clone(),
                            *index,
                            Some(message::ReasoningContent::Encrypted(data.clone())),
                        ),
                        ReasoningDetails::Text {
                            id,
                            index,
                            text,
                            signature,
                            ..
                        } => (
                            id.clone(),
                            *index,
                            text.as_ref().map(|text| message::ReasoningContent::Text {
                                text: text.clone(),
                                signature: signature.clone(),
                            }),
                        ),
                    };

                    let Some(parsed_content) = parsed_content else {
                        continue;
                    };
                    let sort_index = sort_index.unwrap_or(position);

                    if !grouped_reasoning.contains_key(&reasoning_id) {
                        reasoning_order.push(reasoning_id.clone());
                    }
                    grouped_reasoning.entry(reasoning_id).or_default().push((
                        sort_index,
                        position,
                        parsed_content,
                    ));
                }

                if grouped_reasoning.is_empty() {
                    if let Some(reasoning) = reasoning {
                        content.push(completion::AssistantContent::reasoning(reasoning));
                    }
                } else {
                    for reasoning_id in reasoning_order {
                        let Some(mut blocks) = grouped_reasoning.remove(&reasoning_id) else {
                            continue;
                        };
                        blocks.sort_by_key(|(index, position, _)| (*index, *position));
                        content.push(completion::AssistantContent::Reasoning(
                            message::Reasoning {
                                id: reasoning_id,
                                content: blocks
                                    .into_iter()
                                    .map(|(_, _, content)| content)
                                    .collect::<Vec<_>>(),
                            },
                        ));
                    }
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

        let usage = response
            .usage
            .as_ref()
            .map(|usage| completion::Usage {
                input_tokens: usage.prompt_tokens as u64,
                output_tokens: (usage.total_tokens - usage.prompt_tokens) as u64,
                total_tokens: usage.total_tokens as u64,
                cached_input_tokens: 0,
            })
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Choice {
    pub index: usize,
    pub native_finish_reason: Option<String>,
    pub message: Message,
    pub finish_reason: Option<String>,
}

/// OpenRouter message.
///
/// Almost identical to OpenAI's Message, but supports more parameters
/// for some providers like `reasoning`.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    #[serde(alias = "developer")]
    System {
        #[serde(deserialize_with = "string_or_one_or_many")]
        content: OneOrMany<openai::SystemContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        #[serde(deserialize_with = "string_or_one_or_many")]
        content: OneOrMany<openai::UserContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        #[serde(default, deserialize_with = "json_utils::string_or_vec")]
        content: Vec<openai::AssistantContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        refusal: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        audio: Option<openai::AudioAssistant>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(
            default,
            deserialize_with = "json_utils::null_or_vec",
            skip_serializing_if = "Vec::is_empty"
        )]
        tool_calls: Vec<openai::ToolCall>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        reasoning_details: Vec<ReasoningDetails>,
    },
    #[serde(rename = "tool")]
    ToolResult {
        tool_call_id: String,
        content: String,
    },
}

impl Message {
    pub fn system(content: &str) -> Self {
        Message::System {
            content: OneOrMany::one(content.to_owned().into()),
            name: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningDetails {
    #[serde(rename = "reasoning.summary")]
    Summary {
        id: Option<String>,
        format: Option<String>,
        index: Option<usize>,
        summary: String,
    },
    #[serde(rename = "reasoning.encrypted")]
    Encrypted {
        id: Option<String>,
        format: Option<String>,
        index: Option<usize>,
        data: String,
    },
    #[serde(rename = "reasoning.text")]
    Text {
        id: Option<String>,
        format: Option<String>,
        index: Option<usize>,
        text: Option<String>,
        signature: Option<String>,
    },
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
#[serde(untagged)]
enum ToolCallAdditionalParams {
    ReasoningDetails(ReasoningDetails),
    Minimal {
        id: Option<String>,
        format: Option<String>,
    },
}

impl From<openai::Message> for Message {
    fn from(value: openai::Message) -> Self {
        match value {
            openai::Message::System { content, name } => Self::System { content, name },
            openai::Message::User { content, name } => Self::User { content, name },
            openai::Message::Assistant {
                content,
                refusal,
                audio,
                name,
                tool_calls,
            } => Self::Assistant {
                content,
                refusal,
                audio,
                name,
                tool_calls,
                reasoning: None,
                reasoning_details: Vec::new(),
            },
            openai::Message::ToolResult {
                tool_call_id,
                content,
            } => Self::ToolResult {
                tool_call_id,
                content: content.as_text(),
            },
        }
    }
}

impl TryFrom<OneOrMany<message::AssistantContent>> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(value: OneOrMany<message::AssistantContent>) -> Result<Self, Self::Error> {
        let mut text_content = Vec::new();
        let mut tool_calls = Vec::new();
        let mut reasoning = None;
        let mut reasoning_details = Vec::new();

        for content in value.into_iter() {
            match content {
                message::AssistantContent::Text(text) => text_content.push(text),
                message::AssistantContent::ToolCall(tool_call) => {
                    // We usually want to provide back the reasoning to OpenRouter since some
                    // providers require it.
                    // 1. Full reasoning details passed back the user
                    // 2. The signature, an id and a format if present
                    // 3. The signature and the call_id if present
                    if let Some(additional_params) = &tool_call.additional_params
                        && let Ok(additional_params) =
                            serde_json::from_value::<ToolCallAdditionalParams>(
                                additional_params.clone(),
                            )
                    {
                        match additional_params {
                            ToolCallAdditionalParams::ReasoningDetails(full) => {
                                reasoning_details.push(full);
                            }
                            ToolCallAdditionalParams::Minimal { id, format } => {
                                let id = id.or_else(|| tool_call.call_id.clone());
                                if let Some(signature) = &tool_call.signature
                                    && let Some(id) = id
                                {
                                    reasoning_details.push(ReasoningDetails::Encrypted {
                                        id: Some(id),
                                        format,
                                        index: None,
                                        data: signature.clone(),
                                    })
                                }
                            }
                        }
                    } else if let Some(signature) = &tool_call.signature {
                        reasoning_details.push(ReasoningDetails::Encrypted {
                            id: tool_call.call_id.clone(),
                            format: None,
                            index: None,
                            data: signature.clone(),
                        });
                    }
                    tool_calls.push(tool_call.into())
                }
                message::AssistantContent::Reasoning(r) => {
                    let mut emitted_reasoning_detail = false;
                    for reasoning_block in &r.content {
                        let index = Some(reasoning_details.len());
                        match reasoning_block {
                            message::ReasoningContent::Text { text, signature } => {
                                reasoning_details.push(ReasoningDetails::Text {
                                    id: r.id.clone(),
                                    format: None,
                                    index,
                                    text: Some(text.clone()),
                                    signature: signature.clone(),
                                });
                                emitted_reasoning_detail = true;
                            }
                            message::ReasoningContent::Summary(summary) => {
                                reasoning_details.push(ReasoningDetails::Summary {
                                    id: r.id.clone(),
                                    format: None,
                                    index,
                                    summary: summary.clone(),
                                });
                                emitted_reasoning_detail = true;
                            }
                            message::ReasoningContent::Encrypted(data)
                            | message::ReasoningContent::Redacted { data } => {
                                reasoning_details.push(ReasoningDetails::Encrypted {
                                    id: r.id.clone(),
                                    format: None,
                                    index,
                                    data: data.clone(),
                                });
                                emitted_reasoning_detail = true;
                            }
                        }
                    }

                    if !emitted_reasoning_detail {
                        let display = r.display_text();
                        if !display.is_empty() {
                            reasoning = Some(display);
                        }
                    }
                }
                message::AssistantContent::Image(_) => {
                    return Err(Self::Error::ConversionError(
                        "OpenRouter currently doesn't support images.".into(),
                    ));
                }
            }
        }

        // `OneOrMany` ensures at least one `AssistantContent::Text` or `ToolCall` exists,
        //  so either `content` or `tool_calls` will have some content.
        Ok(vec![Message::Assistant {
            content: text_content
                .into_iter()
                .map(|content| content.text.into())
                .collect::<Vec<_>>(),
            refusal: None,
            audio: None,
            name: None,
            tool_calls,
            reasoning,
            reasoning_details,
        }])
    }
}

// We re-use most of the openai implementation when we can and we re-implement
// only the part that differentate for openrouter (like reasoning support).
impl TryFrom<message::Message> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::User { content } => {
                let messages: Vec<openai::Message> = content.try_into()?;
                Ok(messages.into_iter().map(Message::from).collect::<Vec<_>>())
            }
            message::Message::Assistant { content, .. } => content.try_into(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged, rename_all = "snake_case")]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Function(Vec<ToolChoiceFunctionKind>),
}

impl TryFrom<crate::message::ToolChoice> for ToolChoice {
    type Error = CompletionError;

    fn try_from(value: crate::message::ToolChoice) -> Result<Self, Self::Error> {
        let res = match value {
            crate::message::ToolChoice::None => Self::None,
            crate::message::ToolChoice::Auto => Self::Auto,
            crate::message::ToolChoice::Required => Self::Required,
            crate::message::ToolChoice::Specific { function_names } => {
                let vec: Vec<ToolChoiceFunctionKind> = function_names
                    .into_iter()
                    .map(|name| ToolChoiceFunctionKind::Function { name })
                    .collect();

                Self::Function(vec)
            }
        };

        Ok(res)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "function")]
pub enum ToolChoiceFunctionKind {
    Function { name: String },
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct OpenrouterCompletionRequest {
    model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<crate::providers::openai::completion::ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<crate::providers::openai::completion::ToolChoice>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

/// Parameters for building an OpenRouter CompletionRequest
pub struct OpenRouterRequestParams<'a> {
    pub model: &'a str,
    pub request: CompletionRequest,
    pub strict_tools: bool,
}

impl TryFrom<OpenRouterRequestParams<'_>> for OpenrouterCompletionRequest {
    type Error = CompletionError;

    fn try_from(params: OpenRouterRequestParams) -> Result<Self, Self::Error> {
        let OpenRouterRequestParams {
            model,
            request: req,
            strict_tools,
        } = params;

        let mut full_history: Vec<Message> = match &req.preamble {
            Some(preamble) => vec![Message::system(preamble)],
            None => vec![],
        };
        if let Some(docs) = req.normalized_documents() {
            let docs: Vec<Message> = docs.try_into()?;
            full_history.extend(docs);
        }

        let chat_history: Vec<Message> = req
            .chat_history
            .clone()
            .into_iter()
            .map(|message| message.try_into())
            .collect::<Result<Vec<Vec<Message>>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        full_history.extend(chat_history);

        let tool_choice = req
            .tool_choice
            .clone()
            .map(crate::providers::openai::completion::ToolChoice::try_from)
            .transpose()?;

        let tools: Vec<crate::providers::openai::completion::ToolDefinition> = req
            .tools
            .clone()
            .into_iter()
            .map(|tool| {
                let def = crate::providers::openai::completion::ToolDefinition::from(tool);
                if strict_tools { def.with_strict() } else { def }
            })
            .collect();

        Ok(Self {
            model: model.to_string(),
            messages: full_history,
            temperature: req.temperature,
            tools,
            tool_choice,
            additional_params: req.additional_params,
        })
    }
}

impl TryFrom<(&str, CompletionRequest)> for OpenrouterCompletionRequest {
    type Error = CompletionError;

    fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        OpenrouterCompletionRequest::try_from(OpenRouterRequestParams {
            model,
            request: req,
            strict_tools: false,
        })
    }
}

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    pub(crate) client: Client<T>,
    pub model: String,
    /// Enable strict mode for tool schemas.
    /// When enabled, tool schemas are sanitized to meet OpenAI's strict mode requirements.
    pub strict_tools: bool,
}

impl<T> CompletionModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            strict_tools: false,
        }
    }

    /// Enable strict mode for tool schemas.
    ///
    /// When enabled, tool schemas are automatically sanitized to meet OpenAI's strict mode requirements:
    /// - `additionalProperties: false` is added to all objects
    /// - All properties are marked as required
    /// - `strict: true` is set on each function definition
    ///
    /// Note: Not all models on OpenRouter support strict mode. This works best with OpenAI models.
    pub fn with_strict_tools(mut self) -> Self {
        self.strict_tools = true;
        self
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let preamble = completion_request.preamble.clone();
        let request = OpenrouterCompletionRequest::try_from(OpenRouterRequestParams {
            model: self.model.as_ref(),
            request: completion_request,
            strict_tools: self.strict_tools,
        })?;

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "OpenRouter completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "openrouter",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/chat/completions")?
            .body(body)
            .map_err(|x| CompletionError::HttpError(x.into()))?;

        async move {
            let response = self.client.send::<_, Bytes>(req).await?;
            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if status.is_success() {
                match serde_json::from_slice::<ApiResponse<CompletionResponse>>(&response_body)? {
                    ApiResponse::Ok(response) => {
                        let span = tracing::Span::current();
                        span.record_token_usage(&response.usage);
                        span.record("gen_ai.response.id", &response.id);
                        span.record("gen_ai.response.model_name", &response.model);

                        tracing::debug!(target: "rig::completions",
                            "OpenRouter response: {response:?}");
                        response.try_into()
                    }
                    ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
                }
            } else {
                Err(CompletionError::ProviderError(
                    String::from_utf8_lossy(&response_body).to_string(),
                ))
            }
        }
        .instrument(span)
        .await
    }

    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        CompletionModel::stream(self, completion_request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_completion_response_deserialization_gemini_flash() {
        // Real response from OpenRouter with google/gemini-2.5-flash
        let json = json!({
            "id": "gen-AAAAAAAAAA-AAAAAAAAAAAAAAAAAAAA",
            "provider": "Google",
            "model": "google/gemini-2.5-flash",
            "object": "chat.completion",
            "created": 1765971703u64,
            "choices": [{
                "logprobs": null,
                "finish_reason": "stop",
                "native_finish_reason": "STOP",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "CONTENT",
                    "refusal": null,
                    "reasoning": null
                }
            }],
            "usage": {
                "prompt_tokens": 669,
                "completion_tokens": 5,
                "total_tokens": 674
            }
        });

        let response: CompletionResponse = serde_json::from_value(json).unwrap();
        assert_eq!(response.id, "gen-AAAAAAAAAA-AAAAAAAAAAAAAAAAAAAA");
        assert_eq!(response.model, "google/gemini-2.5-flash");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].finish_reason, Some("stop".to_string()));
    }

    #[test]
    fn test_message_assistant_without_reasoning_details() {
        // Verify that missing reasoning_details field doesn't cause deserialization failure
        let json = json!({
            "role": "assistant",
            "content": "Hello world",
            "refusal": null,
            "reasoning": null
        });

        let message: Message = serde_json::from_value(json).unwrap();
        match message {
            Message::Assistant {
                content,
                reasoning_details,
                ..
            } => {
                assert_eq!(content.len(), 1);
                assert!(reasoning_details.is_empty());
            }
            _ => panic!("Expected Assistant message"),
        }
    }

    #[test]
    fn test_completion_response_with_reasoning_details_maps_to_typed_reasoning() {
        let json = json!({
            "id": "resp_123",
            "object": "chat.completion",
            "created": 1,
            "model": "openrouter/test-model",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "reasoning": null,
                    "reasoning_details": [
                        {"type":"reasoning.summary","id":"rs_1","summary":"s1"},
                        {"type":"reasoning.text","id":"rs_1","text":"t1","signature":"sig_1"},
                        {"type":"reasoning.encrypted","id":"rs_1","data":"enc_1"}
                    ]
                }
            }]
        });

        let response: CompletionResponse = serde_json::from_value(json).unwrap();
        let converted: completion::CompletionResponse<CompletionResponse> =
            response.try_into().unwrap();
        let items: Vec<completion::AssistantContent> = converted.choice.into_iter().collect();

        assert!(items.iter().any(|item| matches!(
            item,
            completion::AssistantContent::Reasoning(message::Reasoning { id: Some(id), content })
                if id == "rs_1" && content.len() == 3
        )));
    }

    #[test]
    fn test_assistant_reasoning_emits_openrouter_reasoning_details() {
        let reasoning = message::Reasoning {
            id: Some("rs_2".to_string()),
            content: vec![
                message::ReasoningContent::Text {
                    text: "step".to_string(),
                    signature: Some("sig_step".to_string()),
                },
                message::ReasoningContent::Summary("summary".to_string()),
                message::ReasoningContent::Encrypted("enc_blob".to_string()),
            ],
        };

        let messages = Vec::<Message>::try_from(OneOrMany::one(
            message::AssistantContent::Reasoning(reasoning),
        ))
        .unwrap();
        let Message::Assistant {
            reasoning,
            reasoning_details,
            ..
        } = messages.first().expect("assistant message")
        else {
            panic!("Expected assistant message");
        };

        assert!(reasoning.is_none());
        assert_eq!(reasoning_details.len(), 3);
        assert!(matches!(
            reasoning_details.first(),
            Some(ReasoningDetails::Text {
                id: Some(id),
                text: Some(text),
                signature: Some(signature),
                ..
            }) if id == "rs_2" && text == "step" && signature == "sig_step"
        ));
    }

    #[test]
    fn test_assistant_redacted_reasoning_emits_encrypted_detail_not_text() {
        let reasoning = message::Reasoning {
            id: Some("rs_redacted".to_string()),
            content: vec![message::ReasoningContent::Redacted {
                data: "opaque-redacted-data".to_string(),
            }],
        };

        let messages = Vec::<Message>::try_from(OneOrMany::one(
            message::AssistantContent::Reasoning(reasoning),
        ))
        .unwrap();

        let Message::Assistant {
            reasoning_details,
            reasoning,
            ..
        } = messages.first().expect("assistant message")
        else {
            panic!("Expected assistant message");
        };

        assert!(reasoning.is_none());
        assert_eq!(reasoning_details.len(), 1);
        assert!(matches!(
            reasoning_details.first(),
            Some(ReasoningDetails::Encrypted {
                id: Some(id),
                data,
                ..
            }) if id == "rs_redacted" && data == "opaque-redacted-data"
        ));
    }

    #[test]
    fn test_completion_response_reasoning_details_respects_index_ordering() {
        let json = json!({
            "id": "resp_ordering",
            "object": "chat.completion",
            "created": 1,
            "model": "openrouter/test-model",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "reasoning": null,
                    "reasoning_details": [
                        {"type":"reasoning.summary","id":"rs_order","index":1,"summary":"second"},
                        {"type":"reasoning.summary","id":"rs_order","index":0,"summary":"first"}
                    ]
                }
            }]
        });

        let response: CompletionResponse = serde_json::from_value(json).unwrap();
        let converted: completion::CompletionResponse<CompletionResponse> =
            response.try_into().unwrap();
        let items: Vec<completion::AssistantContent> = converted.choice.into_iter().collect();
        let reasoning_blocks: Vec<_> = items
            .into_iter()
            .filter_map(|item| match item {
                completion::AssistantContent::Reasoning(reasoning) => Some(reasoning),
                _ => None,
            })
            .collect();

        assert_eq!(reasoning_blocks.len(), 1);
        assert_eq!(reasoning_blocks[0].id.as_deref(), Some("rs_order"));
        assert_eq!(
            reasoning_blocks[0].content,
            vec![
                message::ReasoningContent::Summary("first".to_string()),
                message::ReasoningContent::Summary("second".to_string()),
            ]
        );
    }

    #[test]
    fn test_completion_response_reasoning_details_with_multiple_ids_stay_separate() {
        let json = json!({
            "id": "resp_multi_id",
            "object": "chat.completion",
            "created": 1,
            "model": "openrouter/test-model",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "reasoning": null,
                    "reasoning_details": [
                        {"type":"reasoning.summary","id":"rs_a","summary":"a1"},
                        {"type":"reasoning.summary","id":"rs_b","summary":"b1"},
                        {"type":"reasoning.summary","id":"rs_a","summary":"a2"}
                    ]
                }
            }]
        });

        let response: CompletionResponse = serde_json::from_value(json).unwrap();
        let converted: completion::CompletionResponse<CompletionResponse> =
            response.try_into().unwrap();
        let items: Vec<completion::AssistantContent> = converted.choice.into_iter().collect();
        let reasoning_blocks: Vec<_> = items
            .into_iter()
            .filter_map(|item| match item {
                completion::AssistantContent::Reasoning(reasoning) => Some(reasoning),
                _ => None,
            })
            .collect();

        assert_eq!(reasoning_blocks.len(), 2);
        assert_eq!(reasoning_blocks[0].id.as_deref(), Some("rs_a"));
        assert_eq!(
            reasoning_blocks[0].content,
            vec![
                message::ReasoningContent::Summary("a1".to_string()),
                message::ReasoningContent::Summary("a2".to_string()),
            ]
        );
        assert_eq!(reasoning_blocks[1].id.as_deref(), Some("rs_b"));
        assert_eq!(
            reasoning_blocks[1].content,
            vec![message::ReasoningContent::Summary("b1".to_string())]
        );
    }
}
