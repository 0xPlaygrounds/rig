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

/// The whole `/v2/chat` reply. Only `message` and `finish_reason` build the
/// turn; the usage stays as Cohere sent it, read by [`usage_of`].
#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    #[serde(default)]
    pub id: String,
    pub finish_reason: FinishReason,
    /// The assistant message, as Cohere sent it.
    pub message: serde_json::Value,
    #[serde(default)]
    pub usage: Option<serde_json::Value>,
}

impl CompletionResponse {
    /// Clone assistant content, citations, and tool calls. Returns a response
    /// error when the message is not an assistant message.
    pub fn message(
        &self,
    ) -> Result<(Vec<AssistantContent>, Vec<Citation>, Vec<ToolCall>), ProviderError> {
        let Ok(Message::Assistant {
            content,
            citations,
            tool_calls,
            ..
        }) = Message::deserialize(&self.message)
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
    ErrorToxic,
    ErrorLimit,
    UserCancel,
    Timeout,
    ToolCall,
    /// A reason outside the set Cohere documents today, kept verbatim in
    /// Cohere's own spelling rather than failing deserialization.
    #[serde(untagged)]
    Other(String),
}

/// Normalize the terminal reason. Every documented failure (`ERROR`,
/// `ERROR_TOXIC`, `ERROR_LIMIT`, `USER_CANCEL`, `TIMEOUT`) and any unknown
/// value is [`completion::FinishReason::Other`] in Cohere's spelling, which
/// fails the turn.
pub(crate) fn map_finish_reason(reason: &FinishReason) -> completion::FinishReason {
    let failed = |reason: &str| completion::FinishReason::Other(reason.to_owned());
    match reason {
        FinishReason::Complete | FinishReason::StopSequence => completion::FinishReason::Stop,
        FinishReason::MaxTokens => completion::FinishReason::Length,
        FinishReason::ToolCall => completion::FinishReason::ToolCalls,
        FinishReason::Error => failed("ERROR"),
        FinishReason::ErrorToxic => failed("ERROR_TOXIC"),
        FinishReason::ErrorLimit => failed("ERROR_LIMIT"),
        FinishReason::UserCancel => failed("USER_CANCEL"),
        FinishReason::Timeout => failed("TIMEOUT"),
        FinishReason::Other(other) => failed(other),
    }
}

/// The normalized usage of a Cohere usage object, read leniently: a counter
/// that is absent or not a number is unreported. Totals count tokens, not
/// billed units, which exclude cached input and system overhead; a total
/// needs both counts, and cached input is reported only beside `tokens`.
pub(crate) fn usage_of(usage: &serde_json::Value) -> crate::completion::Usage {
    let count = |pointer: &str| {
        usage
            .pointer(pointer)
            .and_then(serde_json::Value::as_f64)
            .filter(|count| *count >= 0.0)
            .map(|count| count as u64)
    };
    let input_tokens = count("/tokens/input_tokens");
    let output_tokens = count("/tokens/output_tokens");
    crate::completion::Usage {
        input_tokens,
        output_tokens,
        total_tokens: input_tokens
            .zip(output_tokens)
            .map(|(input, output)| input + output),
        cached_input_tokens: count("/cached_tokens")
            .filter(|_| usage.get("tokens").is_some_and(|tokens| !tokens.is_null())),
        ..Default::default()
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

/// [`usage_of`] the usage this typed view reads.
impl From<&Usage> for crate::completion::Usage {
    fn from(usage: &Usage) -> crate::completion::Usage {
        usage_of(&serde_json::to_value(usage).unwrap_or_default())
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
                name: tool.name.into(),
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

    /// An assistant message as the wire carries it: Cohere's own message,
    /// or one rebuilt from a turn's blocks. Request-only.
    #[serde(untagged, skip_deserializing)]
    Native(serde_json::Value),
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
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
            message::Message::User { content } => user_messages(content)?,
            message::Message::System { content } => {
                vec![Message::System { content }]
            }
            message::Message::Assistant(turn) => assistant_message(turn).into_iter().collect(),
        })
    }
}

/// A user message as Cohere takes it: each run of text and images one user
/// message, and each tool result a tool message, in order.
fn user_messages(
    content: Vec<message::UserContent>,
) -> Result<Vec<Message>, message::MessageError> {
    let mut messages = Vec::new();
    let mut pending = Vec::new();
    for part in content {
        match part {
            message::UserContent::Text(message::Text { text, .. }) => {
                pending.push(UserContent::Text { text });
            }
            message::UserContent::Image(image) => pending.push(UserContent::ImageUrl {
                image_url: ImageUrl {
                    url: image_url(image)?,
                },
            }),
            message::UserContent::ToolResult(tool_result) => {
                if !pending.is_empty() {
                    messages.push(Message::User {
                        content: std::mem::take(&mut pending),
                    });
                }
                messages.push(Message::Tool {
                    tool_call_id: tool_result.call.wire().into_owned(),
                    content: tool_result
                        .content
                        .into_iter()
                        .map(|content| match content {
                            message::ToolResultContent::Text(text) => {
                                ToolResultContent::Text { text: text.text }
                            }
                            message::ToolResultContent::Json { value } => ToolResultContent::Text {
                                text: value.to_string(),
                            },
                            // The adapter moves a result's images to a user
                            // message, since Cohere reads none here.
                            message::ToolResultContent::Image(_) => ToolResultContent::Text {
                                text: crate::completion::history::TOOL_IMAGE_OMITTED.to_owned(),
                            },
                        })
                        .collect(),
                });
            }
            message::UserContent::Audio(_)
            | message::UserContent::Video(_)
            | message::UserContent::Document(_) => {
                return Err(message::MessageError::ConversionError(
                    "Cohere takes text and images in user messages".to_owned(),
                ));
            }
        }
    }
    if !pending.is_empty() {
        messages.push(Message::User { content: pending });
    }
    Ok(messages)
}

/// An image as the URL Cohere reads: its URL, or a data URL of its base64
/// data.
fn image_url(image: message::Image) -> Result<String, message::MessageError> {
    use message::{DocumentSourceKind, MimeType};
    match image.data {
        DocumentSourceKind::Url(url) => Ok(url),
        DocumentSourceKind::Base64(data) => {
            let media_type = image.media_type.ok_or_else(|| {
                message::MessageError::ConversionError(
                    "a base64 image needs a media type to build its data URL".to_owned(),
                )
            })?;
            Ok(format!("data:{};base64,{data}", media_type.to_mime_type()))
        }
        DocumentSourceKind::Raw(_)
        | DocumentSourceKind::FileId(_)
        | DocumentSourceKind::String(_)
        | DocumentSourceKind::Unknown => Err(message::MessageError::ConversionError(
            "Cohere reads an image by URL or as base64 data".to_owned(),
        )),
    }
}

/// One assistant turn as Cohere takes it, rebuilt from its blocks: each
/// text, thinking and unknown content item in order (as it came while
/// current), the tool plan, and each call with its canonical name and
/// arguments. The provider's own message is never sent, and a turn with no
/// content and no calls sends nothing, as Cohere refuses an empty message.
#[deny(clippy::wildcard_enum_match_arm)]
fn assistant_message(turn: message::AssistantMessage) -> Option<Message> {
    use crate::providers::internal::rebuild::{Piece, Rebuilt, call_item};
    let rebuilt = Rebuilt::of(&turn);
    let content: Vec<serde_json::Value> = rebuilt
        .pieces
        .iter()
        .filter_map(|piece| match piece {
            Piece::Text {
                part: Some(part), ..
            }
            | Piece::Reasoning {
                part: Some(part), ..
            } => Some(part.clone()),
            Piece::Text { text, part: None } => {
                (!text.trim().is_empty()).then(|| serde_json::json!({"type": "text", "text": text}))
            }
            Piece::Reasoning {
                text,
                field: None,
                part: None,
            } => (!text.trim().is_empty())
                .then(|| serde_json::json!({"type": "thinking", "thinking": text})),
            Piece::Reasoning { field: Some(_), .. } => None,
            Piece::Opaque(item) => Some(item.clone()),
        })
        .collect();
    let tool_calls: Vec<serde_json::Value> = rebuilt
        .calls
        .iter()
        .map(|(call, item)| {
            let id = call.id.wire().into_owned();
            call_item(call, item.clone(), Some(id), true)
        })
        .collect();
    if content.is_empty() && tool_calls.is_empty() {
        return None;
    }
    let mut message = serde_json::json!({
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    });
    for (field, text) in rebuilt.reasoning(None) {
        message[field] = text.into();
    }
    Some(Message::Native(message))
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
        let full_history = crate::providers::internal::wire_ids::WireIds::convert(
            req.chat_history,
            Vec::<Message>::try_from,
            |message| match message {
                Message::Assistant { tool_calls, .. } => tool_calls
                    .iter_mut()
                    .filter_map(|call| call.id.as_mut())
                    .collect(),
                Message::Native(item) => item
                    .get_mut("tool_calls")
                    .and_then(serde_json::Value::as_array_mut)
                    .into_iter()
                    .flatten()
                    .filter_map(|call| {
                        match call
                            .as_object_mut()?
                            .entry("id")
                            .or_insert_with(|| "".into())
                        {
                            serde_json::Value::String(id) => Some(id),
                            _ => None,
                        }
                    })
                    .collect(),
                Message::Tool { tool_call_id, .. } => vec![tool_call_id],
                Message::User { .. } | Message::System { .. } => Vec::new(),
            },
        )?;

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
