//! Cohere chat request conversion and typed responses, usage, citations, and tools.
//!
//! ```
//! use rig_core::providers::cohere::completion::FinishReason;
//! let reason: FinishReason = serde_json::from_str("\"COMPLETE\"")?;
//! assert_eq!(reason, FinishReason::Complete);
//! # Ok::<(), serde_json::Error>(())
//! ```

use crate::error::EncodeError;
use crate::error::ProviderError;
use crate::{
    completion, json_utils,
    message::{self, ToolChoice},
};
use std::collections::HashMap;

use crate::completion::CompletionRequest;
use serde::{Deserialize, Serialize};

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

impl CompletionResponse {
    /// Clone assistant content, citations, and tool calls. Returns a response
    /// error when the message is not an assistant message.
    pub fn message(
        &self,
    ) -> Result<(Vec<AssistantContent>, Vec<Citation>, Vec<ToolCall>), ProviderError> {
        let Message::Assistant {
            content,
            citations,
            tool_calls,
            ..
        } = self.message.clone()
        else {
            return Err(ProviderError::Response(
                "completion response did not contain an assistant message".into(),
            ));
        };

        Ok((content, citations, tool_calls))
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

/// Normalize the terminal reason, preserving `ERROR` and unknown values as
/// [`completion::FinishReason::Other`] rather than treating them as natural stops.
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

/// Normalize total token counters, not billed units, which exclude cached input
/// and system overhead. A total requires both input and output counts.
impl From<&Usage> for crate::completion::Usage {
    fn from(usage: &Usage) -> crate::completion::Usage {
        let tokens = usage.tokens.as_ref();
        let input_tokens = tokens.and_then(|t| t.input_tokens).map(|n| n as u64);
        let output_tokens = tokens.and_then(|t| t.output_tokens).map(|n| n as u64);
        crate::completion::Usage {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens
                .zip(output_tokens)
                .map(|(input, output)| input + output),
            // `cached_input_tokens` is a subset of `input_tokens`, so it's only
            // reported when Cohere also reports `tokens`.
            cached_input_tokens: tokens.and(usage.cached_tokens).map(|n| n as u64),
            ..Default::default()
        }
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Document {
    pub id: String,
    /// Document text and metadata, serialized in sorted key order to keep
    /// prompt-cache prefixes stable across map instances.
    #[serde(serialize_with = "crate::json_utils::serialize_map_sorted")]
    pub data: HashMap<String, serde_json::Value>,
}

impl From<completion::Document> for Document {
    fn from(document: completion::Document) -> Self {
        let mut data: HashMap<String, serde_json::Value> = HashMap::new();

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
                        tool_call_id: tool_result.wire_call_id().into_owned(),
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
                                    None => id.wire_hint().into_owned(),
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
    type Error = EncodeError;

    fn try_from(tool_choice: ToolChoice) -> Result<Self, Self::Error> {
        match tool_choice {
            ToolChoice::Required => Ok(Self::Required),
            ToolChoice::None => Ok(Self::None),
            ToolChoice::Auto => Err(EncodeError::request(
                "\"auto\" is not an allowed tool_choice value in the Cohere API; \
                 omit tool_choice to let the model decide",
            )),
            ToolChoice::Specific { .. } => Err(EncodeError::request(
                "the Cohere API cannot be forced to call specific tools by name; \
                 use ToolChoice::Required and restrict the tools you pass instead",
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
    type Error = EncodeError;

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

        let tool_ids =
            crate::providers::internal::tool_call_ids::ToolCallIds::new(&partial_history)
                .map_err(EncodeError::request)?;
        for (position, message) in partial_history.into_iter().enumerate() {
            let mut messages = Vec::<Message>::try_from(message)?;
            let slots: Vec<&mut String> = messages
                .iter_mut()
                .flat_map(|message| match message {
                    Message::Assistant { tool_calls, .. } => tool_calls
                        .iter_mut()
                        .filter_map(|call| call.id.as_mut())
                        .collect(),
                    Message::Tool { tool_call_id, .. } => vec![tool_call_id],
                    _ => Vec::new(),
                })
                .collect();
            tool_ids
                .apply(position, slots)
                .map_err(EncodeError::request)?;
            full_history.extend(messages);
        }

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
            return Err(EncodeError::request(
                "Cohere requires at least one tool when tool_choice is REQUIRED",
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

#[cfg(test)]
mod tests;
