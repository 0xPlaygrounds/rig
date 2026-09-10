use crate::{
    completion::{self, CompletionError},
    http_client::HttpClientExt,
    json_utils,
    message::{self, Reasoning, ToolChoice},
    providers::internal::{completion_send::send_completion, envelope::DirectPayload},
    telemetry::{CompletionOperation, CompletionSpanBuilder, SpanCombinator},
};
use std::collections::HashMap;

use super::client::Client;
use crate::completion::CompletionRequest;
use serde::{Deserialize, Serialize};
use tracing::Instrument;

/// Stable descriptor name recorded on normalized responses, streams, and
/// telemetry spans for this provider.
pub(crate) const PROVIDER_NAME: &str = "cohere";

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
    type Usage = Usage;

    fn response_id(&self) -> Option<&str> {
        Some(self.id.as_str())
    }

    fn response_model_name(&self) -> Option<&str> {
        None
    }

    fn text_response(&self) -> Option<String> {
        let Message::Assistant { ref content, .. } = self.message else {
            return None;
        };

        let res = content
            .iter()
            .filter_map(|x| {
                if let AssistantContent::Text { text } = x {
                    Some(text.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<String>>()
            .join("\n");

        if res.is_empty() { None } else { Some(res) }
    }

    fn usage(&self) -> Option<Self::Usage> {
        self.usage
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
    /// A reason outside the set Cohere documents today, kept verbatim in
    /// Cohere's own spelling rather than failing deserialization.
    #[serde(untagged)]
    Other(String),
}

/// Map Cohere's `finish_reason` onto rig's normalized vocabulary.
///
/// `ERROR` — and anything Cohere adds later — is carried through as
/// [`completion::FinishReason::Other`] in Cohere's own wire spelling instead of
/// being flattened into a natural stop.
pub(crate) fn map_finish_reason(reason: &FinishReason) -> completion::FinishReason {
    match reason {
        FinishReason::Complete | FinishReason::StopSequence => completion::FinishReason::Stop,
        FinishReason::MaxTokens => completion::FinishReason::Length,
        FinishReason::ToolCall => completion::FinishReason::ToolCalls,
        FinishReason::Error => completion::FinishReason::Other("ERROR".to_owned()),
        FinishReason::Other(other) => completion::FinishReason::Other(other.clone()),
    }
}

#[derive(Copy, Debug, Deserialize, Clone, Serialize)]
pub struct Usage {
    #[serde(default)]
    pub billed_units: Option<BilledUnits>,
    #[serde(default)]
    pub tokens: Option<Tokens>,
    /// Subset of `tokens.input_tokens`; excluded from `billed_units.input_tokens`.
    #[serde(default)]
    pub cached_tokens: Option<f64>,
}

/// `tokens` is the total-usage counter; `billed_units` excludes cached input
/// and system overhead, silently undercounting.
impl From<&Usage> for crate::completion::Usage {
    fn from(usage: &Usage) -> crate::completion::Usage {
        let mut normalized = crate::completion::Usage::new();

        if let Some(ref tokens) = usage.tokens {
            normalized.input_tokens = tokens.input_tokens.unwrap_or_default() as u64;
            normalized.output_tokens = tokens.output_tokens.unwrap_or_default() as u64;
            normalized.total_tokens = normalized.input_tokens + normalized.output_tokens;
            // `cached_input_tokens` is a subset of `input_tokens`, so it's only
            // reported when Cohere also reports `input_tokens`.
            normalized.cached_input_tokens = usage.cached_tokens.unwrap_or_default() as u64;
        }

        normalized
    }
}

impl From<Usage> for crate::completion::Usage {
    fn from(usage: Usage) -> crate::completion::Usage {
        crate::completion::Usage::from(&usage)
    }
}

#[derive(Copy, Debug, Deserialize, Clone, Serialize)]
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

#[derive(Copy, Debug, Deserialize, Clone, Serialize)]
pub struct Tokens {
    #[serde(default)]
    pub input_tokens: Option<f64>,
    #[serde(default)]
    pub output_tokens: Option<f64>,
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let (content, _, tool_calls) = response.message()?;

        let model_response = if !tool_calls.is_empty() {
            crate::message::require_non_empty(
                tool_calls
                    .into_iter()
                    .filter_map(|tool_call| {
                        let ToolCallFunction { name, arguments } = tool_call.function?;
                        // The wire's id when present, or empty so the
                        // conversion mints — never the tool name: a name-as-id
                        // is fake provenance and collides two same-tool calls
                        // in one turn.
                        let id = tool_call.id.unwrap_or_default();

                        Some(completion::AssistantContent::tool_call(id, name, arguments))
                    })
                    .collect::<Vec<_>>(),
                || {
                    CompletionError::ResponseError(
                        "response contained tool call metadata without any callable tool content"
                            .to_owned(),
                    )
                },
            )?
        } else {
            crate::message::require_non_empty_response(
                content
                    .into_iter()
                    .map(|content| match content {
                        AssistantContent::Text { text } => completion::AssistantContent::text(text),
                        AssistantContent::Thinking { thinking } => {
                            completion::AssistantContent::Reasoning(Reasoning::new(&thinking))
                        }
                    })
                    .collect::<Vec<_>>(),
            )?
        };

        let usage = response
            .usage
            .as_ref()
            .map(completion::Usage::from)
            .unwrap_or_default();

        Ok(
            // Cohere's `/v2/chat` payload reports no model identifier, so the
            // normalized `model` stays unset.
            completion::CompletionResponse::new(model_response, usage, PROVIDER_NAME)
                .with_optional_response_id(Some(response.id.as_str()).filter(|id| !id.is_empty()))
                .with_finish_reason(map_finish_reason(&response.finish_reason)),
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Document {
    pub id: String,
    /// Document metadata plus its `text`.
    ///
    /// Serialized in sorted key order: `HashMap` iteration order is randomized
    /// per instance, and documents sit inside the `messages` block that Cohere's
    /// prompt cache keys on. An unsorted map therefore gave every request
    /// carrying a document a different prefix, so the cache could never hit —
    /// see [`crate::json_utils::serialize_map_sorted`]. Rig already sorts the
    /// same metadata deliberately when rendering a document into prompt text
    /// (`crate::completion::Document`'s `Display`); this makes the native
    /// document block agree with it.
    #[serde(serialize_with = "crate::json_utils::serialize_map_sorted")]
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
        content: Vec<UserContent>,
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
        content: Vec<ToolResultContent>,
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
                        content: vec![UserContent::Text { text }],
                    }),
                    message::UserContent::ToolResult(tool_result) => Ok(Message::Tool {
                        tool_call_id: tool_result.wire_call_id().to_owned(),
                        content: tool_result
                            .content
                            .into_iter()
                            .map(|content| match content {
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
                            })
                            .collect::<Result<Vec<_>, _>>()?,
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
                            provider,
                            function:
                                message::ToolFunction {
                                    name, arguments, ..
                                },
                            ..
                        }) => {
                            tool_calls.push(ToolCall {
                                id: Some(match provider {
                                    Some(provider) => provider.call_id,
                                    None => id.into_string(),
                                }),
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
                content: content
                    .into_iter()
                    .map(|content| match content {
                        UserContent::Text { text } => {
                            message::UserContent::Text(message::Text::new(text))
                        }
                        UserContent::ImageUrl { image_url } => {
                            message::UserContent::image_url(image_url.url, None, None)
                        }
                    })
                    .collect(),
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

                    // Empty when the wire issued no id, so the conversion
                    // mints — never the tool name (fake provenance; collides
                    // two same-tool calls in one turn).
                    Some(message::AssistantContent::tool_call(
                        tool_call.id.unwrap_or_default(),
                        name,
                        arguments,
                    ))
                }));

                let content = crate::message::require_non_empty(content, || {
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
                let content = content.into_iter().map(|content| {
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
                }).collect::<Result<Vec<_>, _>>()?;

                Ok(message::Message::User {
                    // Cohere tool messages carry no tool name; this
                    // conversion is lossy for name-keyed wires.
                    content: vec![message::UserContent::tool_result_from_wire(
                        tool_call_id,
                        "",
                        content,
                    )],
                })
            }
            Message::System { content } => Ok(message::Message::user(content)),
        }
    }
}

#[derive(Clone)]
pub struct CompletionModel<T = crate::http_client::BoxedHttpClient> {
    pub(crate) client: Client<T>,
    pub model: String,
}

/// Cohere's `tool_choice` is a bare string; only `REQUIRED`/`NONE` are valid.
/// `Auto` errors below rather than silently mapping to the omitted-field
/// behavior that would actually let the model decide.
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

        let mut full_history: Vec<Message> = Vec::new();

        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Vec<Message>>, _>>()?
                .into_iter()
                .flatten(),
        );

        let tool_choice = req
            .tool_choice
            .map(CohereToolChoice::try_from)
            .transpose()?;

        // Count tools supplied through the provider escape hatch as well as
        // typed tools so REQUIRED remains usable with Cohere-specific schemas.
        let has_tools = !req.tools.is_empty()
            || req
                .additional_params
                .as_ref()
                .and_then(|params| params.get("tools"))
                .and_then(serde_json::Value::as_array)
                .is_some_and(|tools| !tools.is_empty());
        if matches!(tool_choice, Some(CohereToolChoice::Required)) && !has_tools {
            return Err(CompletionError::RequestError(
                "Cohere requires at least one tool when tool_choice is REQUIRED".into(),
            ));
        }

        Ok(Self {
            model,
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

impl<T> CompletionModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    /// Execute a completion and return Cohere's own wire response.
    ///
    /// This is the escape hatch for Cohere-specific fields rig does not
    /// normalize (citations, tool plans). It shares the request builder,
    /// transport, telemetry, and error handling with
    /// [`CompletionModel::completion`](completion::CompletionModel::completion),
    /// which calls it and then applies the provider-local mapping — one network
    /// request either way.
    pub async fn raw_completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        let system_instructions = completion_request.system_instructions().map(str::to_owned);
        let record_telemetry_content = completion_request.record_telemetry_content;
        let request = CohereCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

        let llm_span =
            CompletionSpanBuilder::new(PROVIDER_NAME, &request.model, CompletionOperation::Chat)
                .system_instructions(system_instructions.as_deref(), record_telemetry_content)
                .build();

        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "Cohere completion request",
            &request,
        );

        let req_body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/v2/chat")?
            .body(req_body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        // Left unboxed so `provider_response_status`/`_body` can read the
        // status and body straight off the transport error.
        send_completion::<_, DirectPayload<CompletionResponse>, _>(
            &self.client,
            req,
            "Cohere completion",
            // Cohere reports no request-id response header (its `x-debug-trace-id`
            // is a debug trace handle, not a documented request id); the
            // normalized id is None by design.
            None,
            |json_response| {
                let span = tracing::Span::current();
                let usage = json_response
                    .usage
                    .as_ref()
                    .map(completion::Usage::from)
                    .unwrap_or_default();
                span.record_token_usage(&usage);
                span.record_response_metadata(json_response);
            },
        )
        .instrument(llm_span)
        .await
        .map(|(payload, _)| payload)
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        // Capture before `try_into` consumes the raw value.
        let raw = self.raw_completion(completion_request).await?;
        let captured = serde_json::to_value(&raw)?;
        let response: completion::CompletionResponse = raw.try_into()?;
        Ok(response.with_raw(captured))
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError> {
        CompletionModel::stream(self, request).await
    }
}
#[cfg(test)]
mod tests;
