//! Anthropic Messages payloads, conversion, citations, and prompt-cache configuration.
//!
//! ```
//! use rig_core::providers::anthropic::completion::CacheControl;
//!
//! let cache = CacheControl::ephemeral_1h();
//! ```

use crate::completion::CompletionRequest;
use crate::error::EncodeError;
use crate::json_utils::string_or_vec;
use crate::{
    completion,
    message::{self, DocumentMediaType, DocumentSourceKind, MessageError, MimeType},
};
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, str::FromStr};

/// `claude-fable-5-1` completion model
pub const CLAUDE_FABLE_5_1: &str = "claude-fable-5-1";
/// `claude-fable-5` completion model
pub const CLAUDE_FABLE_5: &str = "claude-fable-5";
/// `claude-opus-5` completion model
pub const CLAUDE_OPUS_5: &str = "claude-opus-5";
/// `claude-sonnet-5` completion model
pub const CLAUDE_SONNET_5: &str = "claude-sonnet-5";
/// `claude-opus-4-6` completion model
pub const CLAUDE_OPUS_4_6: &str = "claude-opus-4-6";
/// `claude-opus-4-7` completion model
pub const CLAUDE_OPUS_4_7: &str = "claude-opus-4-7";
/// `claude-opus-4-8` completion model
pub const CLAUDE_OPUS_4_8: &str = "claude-opus-4-8";
/// `claude-sonnet-4-6` completion model
pub const CLAUDE_SONNET_4_6: &str = "claude-sonnet-4-6";
/// `claude-haiku-4-5` completion model
pub const CLAUDE_HAIKU_4_5: &str = "claude-haiku-4-5";

pub const ANTHROPIC_VERSION_2023_01_01: &str = "2023-01-01";
pub const ANTHROPIC_VERSION_2023_06_01: &str = "2023-06-01";
pub const ANTHROPIC_VERSION_LATEST: &str = ANTHROPIC_VERSION_2023_06_01;
pub(crate) const ANTHROPIC_RAW_CONTENT_KEY: &str = "anthropic_content";

#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    pub content: Vec<Content>,
    pub id: String,
    pub model: String,
    pub role: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
    /// Transport request id from the `request-id` response header, attached by
    /// the request driver. Absent when the provider reports no header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
}

/// Normalize a Messages `stop_reason`, preserving unrecognized values verbatim.
pub(crate) fn map_finish_reason(stop_reason: &str) -> completion::FinishReason {
    match stop_reason {
        // `stop_sequence` is a natural termination too: the model completed its
        // turn by emitting one of the caller's stop sequences.
        "end_turn" | "stop_sequence" => completion::FinishReason::Stop,
        "max_tokens" => completion::FinishReason::Length,
        "tool_use" => completion::FinishReason::ToolCalls,
        // Anthropic's classifier-driven refusal; the closest normalized reason
        // is content filtering.
        "refusal" => completion::FinishReason::ContentFilter,
        other => completion::FinishReason::Other(other.to_owned()),
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
    /// Per-TTL breakdown of `cache_creation_input_tokens`. Absent when the
    /// provider does not report it; the aggregate above is always authoritative.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation: Option<CacheCreation>,
    pub output_tokens: u64,
    /// Breakdown of `output_tokens`. Absent when the provider does not report
    /// it (a turn with extended thinking disabled).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
}

/// Breakdown of `usage.output_tokens`, including thinking tokens already counted
/// in that total. Deserialization ignores unrecognized fields.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct OutputTokensDetails {
    /// Output tokens spent on extended thinking this turn.
    #[serde(default)]
    pub thinking_tokens: u64,
}

/// Cache-write tokens by TTL (`usage.cache_creation`).
/// Deserialization ignores unrecognized fields.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct CacheCreation {
    /// Tokens written to the 5-minute cache on this turn.
    #[serde(default)]
    pub ephemeral_5m_input_tokens: u64,
    /// Tokens written to the 1-hour cache on this turn.
    #[serde(default)]
    pub ephemeral_1h_input_tokens: u64,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Input tokens: {}\nCache read input tokens: {}\nCache creation input tokens: {}\nOutput tokens: {}",
            self.input_tokens,
            self.cache_read_input_tokens
                .map_or_else(|| "n/a".to_string(), |token| token.to_string()),
            self.cache_creation_input_tokens
                .map_or_else(|| "n/a".to_string(), |token| token.to_string()),
            self.output_tokens
        )
    }
}

/// Normalize usage, summing input, output, cache-read, and cache-write tokens.
/// Thinking tokens are already included in output and are not added again.
/// Without an input count, leave the total absent.
pub(super) fn anthropic_usage_totals(
    input_tokens: Option<u64>,
    output_tokens: u64,
    cache_read: Option<u64>,
    cache_creation: Option<u64>,
    output_tokens_details: Option<OutputTokensDetails>,
) -> crate::completion::Usage {
    crate::completion::Usage {
        input_tokens,
        output_tokens: Some(output_tokens),
        cached_input_tokens: cache_read,
        cache_creation_input_tokens: cache_creation,
        reasoning_tokens: output_tokens_details.map(|details| details.thinking_tokens),
        total_tokens: input_tokens.map(|input| {
            input + cache_read.unwrap_or(0) + cache_creation.unwrap_or(0) + output_tokens
        }),
        tool_use_prompt_tokens: None,
    }
}

impl From<&Usage> for crate::completion::Usage {
    fn from(value: &Usage) -> crate::completion::Usage {
        anthropic_usage_totals(
            Some(value.input_tokens),
            value.output_tokens,
            value.cache_read_input_tokens,
            value.cache_creation_input_tokens,
            value.output_tokens_details,
        )
    }
}

impl From<Usage> for crate::completion::Usage {
    fn from(value: Usage) -> crate::completion::Usage {
        (&value).into()
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
    /// Whether Anthropic must constrain tool arguments to `input_schema`.
    #[serde(default, skip_serializing_if = "crate::json_utils::is_false")]
    pub strict: bool,
    /// Cache breakpoint marker. Set on the last tool in the array to cache
    /// the tools layer independently of the system prompt. Anthropic accepts
    /// up to 4 `cache_control` markers per request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Cache-breakpoint lifetime: five minutes by default, or one hour.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Default)]
pub enum CacheTtl {
    /// 5-minute TTL (default).
    #[default]
    #[serde(rename = "5m")]
    FiveMinutes,
    /// 1-hour TTL.
    #[serde(rename = "1h")]
    OneHour,
}

/// Cache control directive for Anthropic prompt caching.
///
/// Serialises to `{"type":"ephemeral"}` (default TTL) or
/// `{"type":"ephemeral","ttl":"1h"}` (extended TTL).
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CacheControl {
    Ephemeral {
        /// Optional TTL. Defaults to `"5m"` when omitted.
        #[serde(skip_serializing_if = "Option::is_none")]
        ttl: Option<CacheTtl>,
    },
}

impl CacheControl {
    /// Create a cache control with the default 5-minute TTL.
    pub fn ephemeral() -> Self {
        Self::Ephemeral { ttl: None }
    }

    /// Create a cache control with a 1-hour TTL.
    pub fn ephemeral_1h() -> Self {
        Self::Ephemeral {
            ttl: Some(CacheTtl::OneHour),
        }
    }
}

/// System message content block with optional cache control
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SystemContent {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    #[serde(deserialize_with = "string_or_vec")]
    pub content: Vec<Content>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    Text {
        text: String,
        /// Citations returned by Claude pointing back into the source documents.
        /// Empty (and skipped during serialization) on request-side blocks.
        #[serde(
            default,
            deserialize_with = "null_as_empty_vec",
            skip_serializing_if = "Vec::is_empty"
        )]
        citations: Vec<Citation>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ServerToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: serde_json::Value,
    },
    WebSearchToolResult {
        tool_use_id: String,
        content: serde_json::Value,
    },
    /// The result of an Anthropic-hosted code execution tool call.
    CodeExecutionToolResult {
        tool_use_id: String,
        content: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(deserialize_with = "string_or_vec")]
        content: Vec<ToolResultContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Document {
        source: DocumentSource,
        /// Optional document title, passed to the model but not citable.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        /// Optional document context (e.g. metadata), passed to the model but
        /// not citable. Useful for storing additional information about the
        /// document that should not appear in citation `cited_text`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context: Option<String>,
        /// Configuration for enabling citations on this document. When `enabled`
        /// is true, Claude returns citation metadata on response text blocks
        /// pointing back into this document's content.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        citations: Option<CitationsConfig>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    RedactedThinking {
        data: String,
    },
}

impl FromStr for Content {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Content::from(s.to_owned()))
    }
}

/// Enable [citation metadata](https://docs.anthropic.com/en/docs/build-with-claude/citations)
/// on response text referencing this document.
/// Enable citations on all or none of a request's documents; mixed settings fail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CitationsConfig {
    /// Whether citation tracking is enabled for this document.
    pub enabled: bool,
}

/// A citation pointing to source text using a source-specific locator.
/// Known tags require valid payloads. Unknown tags retain their raw JSON.
/// See the [wire format](https://docs.anthropic.com/en/docs/build-with-claude/citations).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Citation {
    /// A citation locating a character span in a plain text document.
    CharLocation(CharLocationCitation),
    /// A citation locating a page range in a PDF document.
    PageLocation(PageLocationCitation),
    /// A citation locating a block range in a custom-content document.
    ContentBlockLocation(ContentBlockLocationCitation),
    /// A citation locating a block range in a user-provided search result.
    SearchResultLocation(SearchResultLocationCitation),
    /// A citation emitted by Anthropic's server-side web search tool.
    WebSearchResultLocation(WebSearchResultLocationCitation),
    /// A forward-compatible raw citation payload for citation types this crate
    /// does not yet model.
    Unknown(serde_json::Value),
}

/// Payload of a [`Citation::CharLocation`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CharLocationCitation {
    /// The exact text being cited. Not counted toward output tokens.
    pub cited_text: String,
    /// 0-indexed position of the source document in the request's document list.
    pub document_index: usize,
    /// Optional title of the source document, echoed back from the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_title: Option<String>,
    /// 0-indexed character offset where the cited span begins.
    pub start_char_index: usize,
    /// Character offset where the cited span ends (exclusive).
    pub end_char_index: usize,
}

/// Payload of a [`Citation::PageLocation`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageLocationCitation {
    /// The exact text being cited. Not counted toward output tokens.
    pub cited_text: String,
    /// 0-indexed position of the source document in the request's document list.
    pub document_index: usize,
    /// Optional title of the source document, echoed back from the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_title: Option<String>,
    /// 1-indexed page number where the cited span begins.
    pub start_page_number: u32,
    /// 1-indexed page number where the cited span ends (exclusive).
    pub end_page_number: u32,
}

/// Payload of a [`Citation::ContentBlockLocation`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentBlockLocationCitation {
    /// The exact text being cited. Not counted toward output tokens.
    pub cited_text: String,
    /// 0-indexed position of the source document in the request's document list.
    pub document_index: usize,
    /// Optional title of the source document, echoed back from the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_title: Option<String>,
    /// 0-indexed content block index where the cited span begins.
    pub start_block_index: usize,
    /// Content block index where the cited span ends (exclusive).
    pub end_block_index: usize,
}

/// Payload of a [`Citation::SearchResultLocation`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchResultLocationCitation {
    /// The exact text being cited. Not counted toward output tokens.
    pub cited_text: String,
    /// Source URL or identifier from the original search result.
    pub source: String,
    /// Title from the original search result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// 0-indexed position of the cited search result across all search
    /// result blocks in the request.
    pub search_result_index: usize,
    /// 0-indexed content block index where the cited span begins.
    pub start_block_index: usize,
    /// Content block index where the cited span ends (exclusive).
    pub end_block_index: usize,
}

/// Payload of a [`Citation::WebSearchResultLocation`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSearchResultLocationCitation {
    /// The exact text being cited. Not counted toward output tokens.
    pub cited_text: String,
    /// URL of the cited source.
    pub url: String,
    /// Source title, serialized as `null` when absent.
    pub title: Option<String>,
    /// Encrypted reference that must be preserved for multi-turn
    /// conversations.
    pub encrypted_index: String,
}

impl Serialize for Citation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        /// Serialize the per-variant DTO and insert the wire `type` tag.
        fn tagged<S, T>(serializer: S, tag: &str, fields: &T) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
            T: Serialize,
        {
            let mut value = serde_json::to_value(fields).map_err(serde::ser::Error::custom)?;
            if let serde_json::Value::Object(obj) = &mut value {
                obj.insert("type".into(), serde_json::json!(tag));
            }
            value.serialize(serializer)
        }

        match self {
            Citation::CharLocation(fields) => tagged(serializer, "char_location", fields),
            Citation::PageLocation(fields) => tagged(serializer, "page_location", fields),
            Citation::ContentBlockLocation(fields) => {
                tagged(serializer, "content_block_location", fields)
            }
            Citation::SearchResultLocation(fields) => {
                tagged(serializer, "search_result_location", fields)
            }
            Citation::WebSearchResultLocation(fields) => {
                tagged(serializer, "web_search_result_location", fields)
            }
            Citation::Unknown(raw) => raw.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for Citation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        /// Decode the payload of an already tag-matched citation. A modeled tag
        /// carrying a defective payload is an error, never a silent
        /// [`Citation::Unknown`].
        fn payload<T, E>(value: serde_json::Value) -> Result<T, E>
        where
            T: serde::de::DeserializeOwned,
            E: serde::de::Error,
        {
            serde_json::from_value(value).map_err(E::custom)
        }

        // Explicit dispatch prevents malformed known citations from falling back to Unknown.
        let value = serde_json::Value::deserialize(deserializer)?;
        let Some(citation_type) = value.get("type").and_then(serde_json::Value::as_str) else {
            return Ok(Citation::Unknown(value));
        };

        match citation_type {
            "char_location" => Ok(Citation::CharLocation(payload(value)?)),
            "page_location" => Ok(Citation::PageLocation(payload(value)?)),
            "content_block_location" => Ok(Citation::ContentBlockLocation(payload(value)?)),
            "search_result_location" => Ok(Citation::SearchResultLocation(payload(value)?)),
            "web_search_result_location" => Ok(Citation::WebSearchResultLocation(payload(value)?)),
            _ => Ok(Citation::Unknown(value)),
        }
    }
}

/// Deserialize a vector, treating explicit JSON `null` as empty.
/// Anthropic can send null citations on text-block start events.
fn null_as_empty_vec<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    Ok(Option::<Vec<T>>::deserialize(deserializer)?.unwrap_or_default())
}

/// Extract Anthropic-specific document fields (`title`, `context`, `citations`)
/// from the generic [`message::Document::additional_params`] JSON blob.
///
/// Return absent fields when parameters are missing. Ignore non-string title
/// and context values; reject a present, invalid [`CitationsConfig`].
fn extract_anthropic_doc_params(
    additional_params: Option<message::AdditionalParams>,
) -> Result<(Option<String>, Option<String>, Option<CitationsConfig>), MessageError> {
    let Some(value) = additional_params else {
        return Ok((None, None, None));
    };
    let title = value
        .get("title")
        .and_then(|v| v.as_str())
        .map(String::from);
    let context = value
        .get("context")
        .and_then(|v| v.as_str())
        .map(String::from);
    let citations = value
        .get("citations")
        .cloned()
        .map(serde_json::from_value::<CitationsConfig>)
        .transpose()
        .map_err(|e| {
            MessageError::ConversionError(format!(
                "Document `additional_params.citations` is not a valid CitationsConfig: {e}",
            ))
        })?;
    Ok((title, context, citations))
}

/// Extract Anthropic citations attached to a generic [`message::Text`] block.
///
/// Citations are returned by Claude on assistant text blocks when the request
/// enabled them via [`CitationsConfig`]. Internally they are stored as JSON in
/// [`message::Text::additional_params`] so they survive conversion through the
/// generic [`message::AssistantContent`] surface.
///
/// Returns `Ok(vec![])` when no citations are attached. Unknown citation types
/// are preserved as [`Citation::Unknown`]. Returns an error if the `citations`
/// field is malformed or if a known citation type has an invalid shape.
///
/// ```no_run
/// use rig_core::completion::message::{self, AssistantContent};
/// use rig_core::providers::anthropic::completion::anthropic_citations;
///
/// fn print_citations(content: &AssistantContent) {
///     if let AssistantContent::Text(text) = content
///         && let Ok(citations) = anthropic_citations(text)
///         && !citations.is_empty()
///     {
///         println!("{citations:?}");
///     }
/// }
/// # let _ = message::Text::new("");
/// ```
pub fn anthropic_citations(text: &message::Text) -> Result<Vec<Citation>, serde_json::Error> {
    match text
        .additional_params
        .as_ref()
        .and_then(|v| v.get("citations"))
    {
        Some(c) => <Vec<Citation> as serde::Deserialize>::deserialize(c),
        None => Ok(Vec::new()),
    }
}

fn extract_anthropic_text_citations(text: &message::Text) -> Result<Vec<Citation>, MessageError> {
    anthropic_citations(text).map_err(|err| {
        MessageError::ConversionError(format!(
            "Text `additional_params.citations` is not valid Anthropic citations: {err}"
        ))
    })
}

fn anthropic_text_content_from_message_text(text: message::Text) -> Result<Content, MessageError> {
    if let Some(raw_content) = extract_anthropic_raw_content(&text)? {
        if !text.text.is_empty() {
            return Err(MessageError::ConversionError(format!(
                "Text `{ANTHROPIC_RAW_CONTENT_KEY}` metadata cannot be combined with non-empty text"
            )));
        }

        return Ok(raw_content);
    }

    let citations = extract_anthropic_text_citations(&text)?;
    Ok(Content::Text {
        text: text.text,
        citations,
        cache_control: None,
    })
}

fn extract_anthropic_raw_content(text: &message::Text) -> Result<Option<Content>, MessageError> {
    let Some(raw_content) = text
        .additional_params
        .as_ref()
        .and_then(|value| value.get(ANTHROPIC_RAW_CONTENT_KEY))
    else {
        return Ok(None);
    };

    let content = <Content as serde::Deserialize>::deserialize(raw_content).map_err(|err| {
        MessageError::ConversionError(format!(
            "Text `{ANTHROPIC_RAW_CONTENT_KEY}` metadata is not valid Anthropic content: {err}"
        ))
    })?;

    match content {
        Content::ServerToolUse { .. }
        | Content::WebSearchToolResult { .. }
        | Content::CodeExecutionToolResult { .. } => Ok(Some(content)),
        _ => Err(MessageError::ConversionError(format!(
            "Text `{ANTHROPIC_RAW_CONTENT_KEY}` metadata only supports Anthropic server_tool_use, web_search_tool_result, and code_execution_tool_result blocks"
        ))),
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContent {
    Text { text: String },
    Image { source: ImageSource },
}

impl FromStr for ToolResultContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(ToolResultContent::Text { text: s.to_owned() })
    }
}

/// The source of an image content block.
///
/// Anthropic supports two source types for images:
/// - `Base64`: Base64-encoded image data with media type
/// - `Url`: URL reference to an image
///
/// See: <https://docs.anthropic.com/en/api/messages>
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    #[serde(rename = "base64")]
    Base64 {
        data: String,
        media_type: ImageFormat,
    },
    #[serde(rename = "url")]
    Url { url: String },
}

/// The source of a document content block.
///
/// Anthropic supports multiple source types for documents:
/// - `Base64`: Base64-encoded document data (used for PDFs)
/// - `Text`: Plain text document data
/// - `Url`: URL reference to a document
/// - `File`: Provider-side uploaded file reference from the Files API
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    Base64 {
        data: String,
        media_type: DocumentFormat,
    },
    Text {
        data: String,
        media_type: PlainTextMediaType,
    },
    Url {
        url: String,
    },
    File {
        file_id: String,
    },
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat {
    #[serde(rename = "image/jpeg")]
    JPEG,
    #[serde(rename = "image/png")]
    PNG,
    #[serde(rename = "image/gif")]
    GIF,
    #[serde(rename = "image/webp")]
    WEBP,
}

/// The media type for base64-encoded documents.
///
/// Used with the `DocumentSource::Base64` variant. Currently only PDF is supported
/// for base64-encoded document sources.
///
/// See: <https://docs.anthropic.com/en/docs/build-with-claude/pdf-support>
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DocumentFormat {
    #[serde(rename = "application/pdf")]
    PDF,
}

/// The media type for plain text document sources.
///
/// Used with the `DocumentSource::Text` variant.
///
/// See: <https://docs.anthropic.com/en/api/messages>
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub enum PlainTextMediaType {
    #[serde(rename = "text/plain")]
    Plain,
}

impl From<String> for Content {
    fn from(text: String) -> Self {
        Content::Text {
            text,
            citations: Vec::new(),
            cache_control: None,
        }
    }
}

impl From<String> for ToolResultContent {
    fn from(text: String) -> Self {
        ToolResultContent::Text { text }
    }
}

impl TryFrom<message::ImageMediaType> for ImageFormat {
    type Error = MessageError;

    fn try_from(media_type: message::ImageMediaType) -> Result<Self, Self::Error> {
        Ok(match media_type {
            message::ImageMediaType::JPEG => ImageFormat::JPEG,
            message::ImageMediaType::PNG => ImageFormat::PNG,
            message::ImageMediaType::GIF => ImageFormat::GIF,
            message::ImageMediaType::WEBP => ImageFormat::WEBP,
            _ => {
                return Err(MessageError::ConversionError(format!(
                    "Unsupported image media type: {media_type:?}"
                )));
            }
        })
    }
}

/// Preserve object arguments or decode an object from a JSON string.
/// Replace other values with `{}` because Messages requires object tool input.
fn coerce_tool_input(input: serde_json::Value) -> serde_json::Value {
    match input {
        v @ serde_json::Value::Object(_) => v,
        serde_json::Value::String(s) => match serde_json::from_str::<serde_json::Value>(&s) {
            Ok(serde_json::Value::Object(m)) => serde_json::Value::Object(m),
            _ => serde_json::json!({}),
        },
        _ => serde_json::json!({}),
    }
}

fn anthropic_content_from_assistant_content(
    content: message::AssistantContent,
) -> Result<Vec<Content>, MessageError> {
    match content {
        message::AssistantContent::Text(text) => {
            // Anthropic rejects empty text; only supported raw hosted-tool metadata
            // can give an otherwise empty block replayable content.
            if text.text.is_empty() && extract_anthropic_raw_content(&text)?.is_none() {
                return Ok(Vec::new());
            }
            Ok(vec![anthropic_text_content_from_message_text(text)?])
        }
        message::AssistantContent::Image(_) => Err(MessageError::ConversionError(
            "Anthropic currently doesn't support images.".to_string(),
        )),
        message::AssistantContent::ToolCall(tool_call) => Ok(vec![Content::ToolUse {
            // The wire requires a non-empty id: the provider-issued one when it
            // exists, else rig's minted handle.
            id: tool_call.wire_call_id().into_owned(),
            name: tool_call.function.name,
            input: coerce_tool_input(tool_call.function.arguments),
        }]),
        message::AssistantContent::Reasoning(reasoning) => {
            let mut converted = Vec::new();
            for block in reasoning.content {
                match block {
                    message::ReasoningContent::Text { text, signature } => {
                        converted.push(Content::Thinking {
                            thinking: text,
                            signature,
                        });
                    }
                    message::ReasoningContent::Summary(summary) => {
                        converted.push(Content::Thinking {
                            thinking: summary,
                            signature: None,
                        });
                    }
                    message::ReasoningContent::Redacted { data }
                    | message::ReasoningContent::Encrypted(data) => {
                        converted.push(Content::RedactedThinking { data });
                    }
                }
            }

            if converted.is_empty() {
                return Err(MessageError::ConversionError(
                    "Cannot convert empty reasoning content to Anthropic format".to_string(),
                ));
            }

            Ok(converted)
        }
    }
}

impl TryFrom<message::Message> for Message {
    type Error = MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        Ok(match message {
            message::Message::User { content } => Message {
                role: Role::User,
                content: content.into_iter().map(|content| match content {
                    message::UserContent::Text(message::Text { text, .. }) => {
                        Ok(Content::from(text))
                    }
                    message::UserContent::ToolResult(tool_result) => Ok(Content::ToolResult {
                        tool_use_id: tool_result.wire_call_id().into_owned(),
                        content: tool_result.content.into_iter().map(|content| match content {
                            message::ToolResultContent::Text(message::Text { text, .. }) => {
                                Ok(ToolResultContent::Text { text })
                            }
                            message::ToolResultContent::Json { value } => {
                                Ok(ToolResultContent::Text {
                                    text: value.to_string(),
                                })
                            }
                            message::ToolResultContent::Image(image) => {
                                let DocumentSourceKind::Base64(data) = image.data else {
                                    return Err(MessageError::ConversionError(
                                        "Only base64 strings can be used with the Anthropic API"
                                            .to_string(),
                                    ));
                                };
                                let media_type =
                                    image.media_type.ok_or(MessageError::ConversionError(
                                        "Image media type is required".to_owned(),
                                    ))?;
                                Ok(ToolResultContent::Image {
                                    source: ImageSource::Base64 {
                                        data,
                                        media_type: media_type.try_into()?,
                                    },
                                })
                            }
                        }).collect::<Result<Vec<_>, _>>()?,
                        is_error: None,
                        cache_control: None,
                    }),
                    message::UserContent::Image(message::Image {
                        data, media_type, ..
                    }) => {
                        let source = match data {
                            DocumentSourceKind::Base64(data) => {
                                let media_type =
                                    media_type.ok_or(MessageError::ConversionError(
                                        "Image media type is required for Claude API".to_string(),
                                    ))?;
                                ImageSource::Base64 {
                                    data,
                                    media_type: ImageFormat::try_from(media_type)?,
                                }
                            }
                            DocumentSourceKind::Url(url) => ImageSource::Url { url },
                            DocumentSourceKind::Unknown => {
                                return Err(MessageError::ConversionError(
                                    "Image content has no body".into(),
                                ));
                            }
                            doc => {
                                return Err(MessageError::ConversionError(format!(
                                    "Unsupported document type: {doc:?}"
                                )));
                            }
                        };

                        Ok(Content::Image {
                            source,
                            cache_control: None,
                        })
                    }
                    message::UserContent::Document(message::Document {
                        data,
                        media_type,
                        additional_params,
                    }) => {
                        let (title, context, citations) =
                            extract_anthropic_doc_params(additional_params)?;

                        if let DocumentSourceKind::FileId(file_id) = data {
                            return Ok(Content::Document {
                                source: DocumentSource::File { file_id },
                                title,
                                context,
                                citations,
                                cache_control: None,
                            });
                        }

                        let media_type = match media_type {
                            Some(media_type) => media_type,
                            // Anthropic's URL document source has no media-type field and is
                            // defined specifically for PDFs, so the source itself is sufficient.
                            None if matches!(&data, DocumentSourceKind::Url(_)) => {
                                DocumentMediaType::PDF
                            }
                            None => {
                                return Err(MessageError::ConversionError(
                                    "Document media type is required".to_string(),
                                ));
                            }
                        };

                        let source = match media_type {
                            DocumentMediaType::PDF => match data {
                                DocumentSourceKind::Base64(data)
                                | DocumentSourceKind::String(data) => DocumentSource::Base64 {
                                    data,
                                    media_type: DocumentFormat::PDF,
                                },
                                DocumentSourceKind::Url(url) => DocumentSource::Url { url },
                                _ => {
                                    return Err(MessageError::ConversionError(
                                        "Only base64 encoded data or URLs are supported for PDF documents".into(),
                                    ));
                                }
                            },
                            DocumentMediaType::TXT => {
                                let (DocumentSourceKind::String(data)
                                | DocumentSourceKind::Base64(data)) = data
                                else {
                                    return Err(MessageError::ConversionError(
                                        "Only string or base64 data is supported for plain text documents".into(),
                                    ));
                                };
                                DocumentSource::Text {
                                    data,
                                    media_type: PlainTextMediaType::Plain,
                                }
                            }
                            other => {
                                return Err(MessageError::ConversionError(format!(
                                    "Anthropic only supports PDF and plain text documents, got: {}",
                                    other.to_mime_type()
                                )));
                            }
                        };

                        Ok(Content::Document {
                            source,
                            title,
                            context,
                            citations,
                            cache_control: None,
                        })
                    }
                    message::UserContent::Audio { .. } => Err(MessageError::ConversionError(
                        "Audio is not supported in Anthropic".to_owned(),
                    )),
                    message::UserContent::Video { .. } => Err(MessageError::ConversionError(
                        "Video is not supported in Anthropic".to_owned(),
                    )),
                }).collect::<Result<Vec<_>, _>>()?,
            },

            message::Message::System { content } => Message {
                role: Role::System,
                content: vec![Content::from(content)],
            },

            message::Message::Assistant { content, .. } => {
                let converted_content = content.into_iter().try_fold(
                    Vec::new(),
                    |mut accumulated, assistant_content| {
                        accumulated
                            .extend(anthropic_content_from_assistant_content(assistant_content)?);
                        Ok::<Vec<Content>, MessageError>(accumulated)
                    },
                )?;

                Message {
                    content: crate::message::require_non_empty(converted_content, || {
                        MessageError::ConversionError(
                            "Assistant message did not contain Anthropic-compatible content"
                                .to_owned(),
                        )
                    })?,
                    role: Role::Assistant,
                }
            }
        })
    }
}

/// Return the published synchronous output limit for a recognized model prefix.
/// Unknown models require an explicit `max_tokens` value.
pub(super) fn default_max_tokens_for_model(model: &str) -> Option<u64> {
    if model.starts_with("claude-fable-5")
        || model.starts_with("claude-opus-5")
        || model.starts_with("claude-sonnet-5")
        || model.starts_with("claude-opus-4-8")
        || model.starts_with("claude-opus-4-7")
        || model.starts_with("claude-opus-4-6")
        || model.starts_with("claude-sonnet-4-6")
    {
        Some(128_000)
    } else if model.starts_with("claude-opus-4")
        || model.starts_with("claude-sonnet-4")
        || model.starts_with("claude-haiku-4-5")
    {
        Some(64_000)
    } else {
        None
    }
}

/// Per Anthropic's mid-conversation system messages docs: Fable 5.x, Opus 4.8 and
/// Opus 5 accept `role: "system"` inside `messages`; Sonnet 5 does not.
pub(super) fn supports_mid_conversation_system_messages(model: &str) -> bool {
    model.starts_with(CLAUDE_FABLE_5)
        || model.starts_with(CLAUDE_OPUS_5)
        || model.starts_with(CLAUDE_OPUS_4_8)
}

#[derive(Default, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    Auto,
    Any,
    None,
    Tool {
        name: String,
    },
}
impl TryFrom<message::ToolChoice> for ToolChoice {
    type Error = EncodeError;

    fn try_from(value: message::ToolChoice) -> Result<Self, Self::Error> {
        let res = match value {
            message::ToolChoice::Auto => Self::Auto,
            message::ToolChoice::None => Self::None,
            message::ToolChoice::Required => Self::Any,
            message::ToolChoice::Specific { function_names } => {
                if function_names.len() != 1 {
                    return Err(EncodeError::request(
                        "Only one tool may be specified to be used by Claude",
                    ));
                }

                let Some(name) = function_names.into_iter().next() else {
                    return Err(EncodeError::request(
                        "Only one tool may be specified to be used by Claude",
                    ));
                };

                Self::Tool { name }
            }
        };

        Ok(res)
    }
}

/// Require all object properties, disallow additional properties, and remove
/// numeric constraints for Anthropic structured output.
fn sanitize_schema(schema: &mut serde_json::Value) {
    crate::providers::internal::schema::sanitize_schema(
        schema,
        crate::providers::internal::schema::SanitizeOptions {
            strip_ref_siblings: false,
            inject_empty_properties: false,
            strip_numeric_constraints: true,
        },
    );
}

/// Adapt a strict tool schema using Anthropic's SDK transformation policy.
///
/// Strict tools support optional parameters, so declared `required` lists are
/// preserved. Unsupported validation keywords are moved into descriptions as
/// model guidance instead of reaching the constrained-decoding compiler.
pub(super) fn sanitize_strict_tool_schema(schema: &mut serde_json::Value) {
    let mut original = std::mem::take(schema);
    inline_local_root_reference(&mut original);
    flatten_root_all_of(&mut original);
    // Tool-input roots require an explicit object type even when properties imply it.
    if let serde_json::Value::Object(source) = &mut original
        && !source.contains_key("type")
        && (source.contains_key("properties") || source.contains_key("$ref"))
    {
        source.insert(
            "type".to_string(),
            serde_json::Value::String("object".to_string()),
        );
    }
    *schema = transform_strict_tool_schema(original);
}

/// Anthropic rejects `allOf` at the top level of a tool input even when every
/// branch describes an object. Merge those object branches into the root while
/// preserving per-property collisions as nested `allOf` constraints.
fn flatten_root_all_of(schema: &mut serde_json::Value) {
    use serde_json::{Map, Value};

    let Value::Object(root) = schema else {
        return;
    };
    let Some(all_of) = root.remove("allOf") else {
        return;
    };
    let mut conflicting_constraints = Map::new();
    merge_root_all_of(root, all_of, &mut conflicting_constraints);
    if !conflicting_constraints.is_empty() {
        root.insert(
            "rootAllOfConstraints".to_string(),
            Value::Object(conflicting_constraints),
        );
    }
}

/// Anthropic requires a tool input's root to have `type: object`, but rejects
/// `type` beside `$ref`. Resolve local root references before transformation so
/// both requirements can be met while retaining definitions needed by nested
/// references.
fn inline_local_root_reference(schema: &mut serde_json::Value) {
    use serde_json::Value;

    let mut seen = std::collections::BTreeSet::new();
    loop {
        let Some(reference) = schema
            .get("$ref")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
        else {
            return;
        };
        let Some(pointer) = reference.strip_prefix('#') else {
            return;
        };
        if !seen.insert(reference.clone()) {
            return;
        }
        let Some(Value::Object(mut referenced)) = schema.pointer(pointer).cloned() else {
            return;
        };
        let Some(mut root) = schema.as_object().cloned() else {
            return;
        };
        root.remove("$ref");

        for keyword in ["$defs", "definitions"] {
            let Some(root_definitions) = root.remove(keyword) else {
                continue;
            };
            let definitions =
                merge_document_definitions(root_definitions, referenced.remove(keyword));
            referenced.insert(keyword.to_string(), definitions);
        }

        merge_root_reference_siblings(&mut referenced, root);

        *schema = Value::Object(referenced);
    }
}

fn merge_document_definitions(
    root_definitions: serde_json::Value,
    local_definitions: Option<serde_json::Value>,
) -> serde_json::Value {
    use serde_json::Value;

    match (root_definitions, local_definitions) {
        (Value::Object(root_definitions), Some(Value::Object(mut local_definitions))) => {
            // Absolute JSON pointers still resolve from the document root.
            // Keep those root targets authoritative when an inlined schema
            // happens to define the same name locally.
            local_definitions.extend(root_definitions);
            Value::Object(local_definitions)
        }
        (root_definitions, _) => root_definitions,
    }
}

/// Merge keywords adjacent to a root `$ref` into its resolved object. JSON
/// Schema applies those siblings conjunctively; simply replacing the root with
/// the referenced object would silently discard valid constraints.
fn merge_root_reference_siblings(
    referenced: &mut serde_json::Map<String, serde_json::Value>,
    siblings: serde_json::Map<String, serde_json::Value>,
) {
    use serde_json::{Map, Value};

    let mut conflicting_constraints = Map::new();
    for (keyword, sibling) in siblings {
        match keyword.as_str() {
            "properties" => merge_schema_properties(referenced, sibling),
            "required" => merge_required_properties(referenced, sibling),
            "allOf" => merge_root_all_of(referenced, sibling, &mut conflicting_constraints),
            // Root unions are unsupported; retain their constraints as model guidance.
            "anyOf" | "oneOf" => {
                conflicting_constraints.insert(keyword, sibling);
            }
            // These describe the root document rather than adding a second
            // validation constraint. Prefer the root-level annotation.
            "description" | "title" | "$schema" | "$id" | "$comment" | "default" | "examples"
            | "deprecated" | "readOnly" | "writeOnly" => {
                referenced.insert(keyword, sibling);
            }
            _ => match referenced.get(&keyword) {
                None => {
                    referenced.insert(keyword, sibling);
                }
                Some(existing) if existing == &sibling => {}
                Some(_) => {
                    conflicting_constraints.insert(keyword, sibling);
                }
            },
        }
    }

    if !conflicting_constraints.is_empty() {
        // Root allOf is unsupported, so unmerged constraints remain model guidance.
        referenced.insert(
            "rootRefSiblingConstraints".to_string(),
            Value::Object(conflicting_constraints),
        );
    }
}

fn merge_root_all_of(
    schema: &mut serde_json::Map<String, serde_json::Value>,
    sibling: serde_json::Value,
    conflicting_constraints: &mut serde_json::Map<String, serde_json::Value>,
) {
    use serde_json::Value;

    let Value::Array(branches) = sibling else {
        conflicting_constraints.insert("allOf".to_string(), sibling);
        return;
    };
    let mut unsupported_branches = Vec::new();
    for branch in branches {
        match branch {
            Value::Object(mut branch) => {
                if branch.contains_key("$ref") {
                    for keyword in ["$defs", "definitions"] {
                        let Some(root_definitions) = schema.get(keyword).cloned() else {
                            continue;
                        };
                        let definitions =
                            merge_document_definitions(root_definitions, branch.remove(keyword));
                        branch.insert(keyword.to_string(), definitions);
                    }
                    let mut branch = Value::Object(branch);
                    inline_local_root_reference(&mut branch);
                    match branch {
                        Value::Object(branch) => merge_root_reference_siblings(schema, branch),
                        branch => unsupported_branches.push(branch),
                    }
                } else {
                    merge_root_reference_siblings(schema, branch);
                }
            }
            branch => unsupported_branches.push(branch),
        }
    }
    if !unsupported_branches.is_empty() {
        conflicting_constraints.insert("allOf".to_string(), Value::Array(unsupported_branches));
    }
}

fn merge_schema_properties(
    schema: &mut serde_json::Map<String, serde_json::Value>,
    sibling: serde_json::Value,
) {
    use serde_json::{Map, Value};

    let Value::Object(sibling_properties) = sibling else {
        schema.entry("properties".to_string()).or_insert(sibling);
        return;
    };
    let properties = schema
        .entry("properties".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    let Value::Object(properties) = properties else {
        return;
    };

    for (name, sibling_schema) in sibling_properties {
        match properties.remove(&name) {
            None => {
                properties.insert(name, sibling_schema);
            }
            Some(existing) if existing == sibling_schema => {
                properties.insert(name, existing);
            }
            Some(existing) => {
                properties.insert(
                    name,
                    Value::Object(Map::from_iter([(
                        "allOf".to_string(),
                        Value::Array(vec![existing, sibling_schema]),
                    )])),
                );
            }
        }
    }
}

fn merge_required_properties(
    schema: &mut serde_json::Map<String, serde_json::Value>,
    sibling: serde_json::Value,
) {
    use serde_json::Value;

    let Value::Array(sibling_required) = sibling else {
        schema.entry("required".to_string()).or_insert(sibling);
        return;
    };
    let required = schema
        .entry("required".to_string())
        .or_insert_with(|| Value::Array(Vec::new()));
    let Value::Array(required) = required else {
        return;
    };
    for name in sibling_required {
        if !required.contains(&name) {
            required.push(name);
        }
    }
}

fn transform_strict_tool_schema(schema: serde_json::Value) -> serde_json::Value {
    use serde_json::{Map, Value};

    let Value::Object(mut source) = schema else {
        return schema;
    };
    let mut strict = Map::new();

    for keyword in ["$defs", "definitions"] {
        if let Some(definitions) = source.remove(keyword) {
            match definitions {
                Value::Object(definitions) => {
                    strict.insert(
                        keyword.to_string(),
                        Value::Object(
                            definitions
                                .into_iter()
                                .map(|(name, schema)| (name, transform_strict_tool_schema(schema)))
                                .collect(),
                        ),
                    );
                }
                definitions => {
                    source.insert(keyword.to_string(), definitions);
                }
            }
        }
    }

    if let Some(reference) = source.remove("$ref") {
        strict.insert("$ref".to_string(), reference);
        return Value::Object(strict);
    }

    let schema_type = source.remove("type");
    let any_of = source.remove("anyOf");
    let one_of = source.remove("oneOf");
    let all_of = source.remove("allOf");
    let alternatives = match (any_of, one_of, all_of) {
        (Some(Value::Array(variants)), _, _) => Some(("anyOf", variants)),
        (_, Some(Value::Array(variants)), _) => Some(("anyOf", variants)),
        (_, _, Some(Value::Array(variants))) => Some(("allOf", variants)),
        _ => None,
    };
    if let Some((keyword, variants)) = alternatives {
        strict.insert(
            keyword.to_string(),
            Value::Array(
                variants
                    .into_iter()
                    .map(transform_strict_tool_schema)
                    .collect(),
            ),
        );
    } else if let Some(schema_type) = schema_type.clone() {
        strict.insert("type".to_string(), schema_type);
    }

    if let Some(Value::Array(values)) = source.remove("enum") {
        strict.insert("enum".to_string(), Value::Array(values));
    }
    if let Some(constant) = source.remove("const") {
        strict.insert("const".to_string(), constant);
    }
    for keyword in ["description", "title"] {
        if let Some(Value::String(value)) = source.remove(keyword) {
            strict.insert(keyword.to_string(), Value::String(value));
        }
    }

    let has_properties = source.contains_key("properties");
    let properties_imply_object = schema_type.is_none() && has_properties;
    if properties_imply_object {
        strict.insert("type".to_string(), Value::String("object".to_string()));
    }
    if schema_has_type(schema_type.as_ref(), "object") || has_properties {
        let properties = match source.remove("properties") {
            Some(Value::Object(properties)) => properties
                .into_iter()
                .map(|(name, schema)| (name, transform_strict_tool_schema(schema)))
                .collect(),
            _ => Map::new(),
        };
        strict.insert("properties".to_string(), Value::Object(properties));
        source.remove("additionalProperties");
        strict.insert("additionalProperties".to_string(), Value::Bool(false));
        if let Some(Value::Array(required)) = source.remove("required") {
            strict.insert("required".to_string(), Value::Array(required));
        }
    }

    if schema_has_type(schema_type.as_ref(), "string")
        && let Some(format) = source.remove("format")
    {
        const SUPPORTED_FORMATS: &[&str] = &[
            "date-time",
            "time",
            "date",
            "duration",
            "email",
            "hostname",
            "uri",
            "ipv4",
            "ipv6",
            "uuid",
        ];
        if format
            .as_str()
            .is_some_and(|format| SUPPORTED_FORMATS.contains(&format))
        {
            strict.insert("format".to_string(), format);
        } else {
            source.insert("format".to_string(), format);
        }
    }

    if schema_has_type(schema_type.as_ref(), "array") {
        if let Some(items) = source.remove("items") {
            strict.insert("items".to_string(), transform_strict_tool_schema(items));
        }
        if let Some(min_items) = source.remove("minItems") {
            if matches!(min_items.as_u64(), Some(0 | 1)) {
                strict.insert("minItems".to_string(), min_items);
            } else {
                source.insert("minItems".to_string(), min_items);
            }
        }
    }

    if !source.is_empty() {
        let hints = source
            .into_iter()
            .map(|(keyword, value)| {
                let value = match value {
                    Value::String(value) => value,
                    value => value.to_string(),
                };
                format!("{keyword}: {value}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        let suffix = format!("{{{hints}}}");
        match strict.get_mut("description") {
            Some(Value::String(description)) => {
                description.push_str("\n\n");
                description.push_str(&suffix);
            }
            _ => {
                strict.insert("description".to_string(), Value::String(suffix));
            }
        }
    }

    Value::Object(strict)
}

fn schema_has_type(schema_type: Option<&serde_json::Value>, expected: &str) -> bool {
    match schema_type {
        Some(serde_json::Value::String(schema_type)) => schema_type == expected,
        Some(serde_json::Value::Array(schema_types)) => schema_types
            .iter()
            .any(|schema_type| schema_type.as_str() == Some(expected)),
        _ => false,
    }
}

/// Output format specifier for Anthropic's structured output.
/// Source: <https://docs.anthropic.com/en/api/messages>
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OutputFormat {
    /// Constrains the model's response to conform to the provided JSON schema.
    JsonSchema { schema: serde_json::Value },
}

/// Configuration for the model's output format.
#[derive(Debug, Deserialize, Serialize)]
struct OutputConfig {
    format: OutputFormat,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct AnthropicCompletionRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u64,
    /// System prompt as array of content blocks to support cache_control
    #[serde(skip_serializing_if = "Vec::is_empty")]
    system: Vec<SystemContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<OutputConfig>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    additional_params: Option<serde_json::Value>,
    /// Top-level cache_control for Anthropic's automatic caching mode. When set, the API
    /// automatically places the cache breakpoint on the last cacheable block and advances it as
    /// the conversation grows. No beta header is required.
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

/// Helper to set cache_control on a Content block
fn set_content_cache_control(content: &mut Content, value: Option<CacheControl>) {
    match content {
        Content::Text { cache_control, .. } => *cache_control = value,
        Content::Image { cache_control, .. } => *cache_control = value,
        Content::ToolResult { cache_control, .. } => *cache_control = value,
        Content::Document { cache_control, .. } => *cache_control = value,
        _ => {}
    }
}

const MAX_CACHE_CONTROL_MARKERS: usize = 4;

fn final_cacheable_tool_idx(tools: &[serde_json::Value]) -> Option<usize> {
    tools.iter().rposition(|tool| {
        tool.as_object().is_some_and(|tool| {
            !matches!(
                tool.get("defer_loading"),
                Some(serde_json::Value::Bool(true))
            )
        })
    })
}

fn tool_cache_control_count(tools: &[serde_json::Value]) -> usize {
    tools
        .iter()
        .filter(|tool| tool_cache_control_value(tool).is_some())
        .count()
}

fn tool_cache_control_value(tool: &serde_json::Value) -> Option<&serde_json::Value> {
    tool.get("cache_control")
        .filter(|cache_control| !cache_control.is_null())
}

fn normalize_tool_cache_control(tools: &mut [serde_json::Value]) {
    for tool in tools.iter_mut() {
        if let Some(tool) = tool.as_object_mut()
            && tool
                .get("cache_control")
                .is_some_and(serde_json::Value::is_null)
        {
            tool.remove("cache_control");
        }
    }
}

fn build_cache_control(ttl: Option<CacheTtl>) -> CacheControl {
    CacheControl::Ephemeral { ttl }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CacheControlTtl {
    FiveMinutes,
    OneHour,
}

fn cache_control_ttl(cache_control: &CacheControl) -> CacheControlTtl {
    match cache_control {
        CacheControl::Ephemeral {
            ttl: Some(CacheTtl::OneHour),
        } => CacheControlTtl::OneHour,
        CacheControl::Ephemeral { .. } => CacheControlTtl::FiveMinutes,
    }
}

fn cache_control_ttl_from_json(cache_control: &serde_json::Value) -> CacheControlTtl {
    match cache_control.get("ttl") {
        Some(serde_json::Value::String(ttl)) if ttl == "1h" => CacheControlTtl::OneHour,
        _ => CacheControlTtl::FiveMinutes,
    }
}

fn content_cache_control(content: &Content) -> Option<&CacheControl> {
    match content {
        Content::Text { cache_control, .. }
        | Content::Image { cache_control, .. }
        | Content::ToolResult { cache_control, .. }
        | Content::Document { cache_control, .. } => cache_control.as_ref(),
        _ => None,
    }
}

fn validate_cache_control_ttl(
    ttl: CacheControlTtl,
    shorter_ttl_seen: &mut bool,
) -> Result<(), EncodeError> {
    match ttl {
        CacheControlTtl::OneHour if *shorter_ttl_seen => Err(EncodeError::request(
            "Anthropic cache_control markers with ttl `1h` must appear before markers with \
                 the default 5-minute TTL",
        )),
        CacheControlTtl::OneHour => Ok(()),
        CacheControlTtl::FiveMinutes => {
            *shorter_ttl_seen = true;
            Ok(())
        }
    }
}

fn validate_cache_control_ttl_order(
    system: &[SystemContent],
    messages: &[Message],
    tools: &[serde_json::Value],
    top_level_cache_control: Option<&CacheControl>,
) -> Result<(), EncodeError> {
    let mut shorter_ttl_seen = false;

    for tool in tools {
        if let Some(cache_control) = tool_cache_control_value(tool) {
            validate_cache_control_ttl(
                cache_control_ttl_from_json(cache_control),
                &mut shorter_ttl_seen,
            )?;
        }
    }

    for SystemContent::Text { cache_control, .. } in system {
        if let Some(cache_control) = cache_control {
            validate_cache_control_ttl(cache_control_ttl(cache_control), &mut shorter_ttl_seen)?;
        }
    }

    for message in messages {
        for content in message.content.iter() {
            if let Some(cache_control) = content_cache_control(content) {
                validate_cache_control_ttl(
                    cache_control_ttl(cache_control),
                    &mut shorter_ttl_seen,
                )?;
            }
        }
    }

    if let Some(cache_control) = top_level_cache_control {
        validate_cache_control_ttl(cache_control_ttl(cache_control), &mut shorter_ttl_seen)?;
    }

    Ok(())
}

fn top_level_cache_control_ttl(cache_control: Option<&CacheControl>) -> Option<CacheTtl> {
    cache_control
        .map(|cache_control| match cache_control {
            CacheControl::Ephemeral { ttl } => ttl.clone(),
        })
        .unwrap_or_default()
}

/// Apply a cache-control breakpoint to the final cacheable tool definition in the request.
fn apply_tool_cache_control(
    tools: &mut [serde_json::Value],
    remaining_cache_markers: &mut usize,
    cache_control: &CacheControl,
) -> Result<(), EncodeError> {
    let Some(idx) = final_cacheable_tool_idx(tools) else {
        return Ok(());
    };

    let Some(tool) = tools
        .get_mut(idx)
        .and_then(serde_json::Value::as_object_mut)
    else {
        return Ok(());
    };

    if tool
        .get("cache_control")
        .is_some_and(|cache_control| !cache_control.is_null())
    {
        return Ok(());
    }

    if *remaining_cache_markers == 0 {
        return Err(EncodeError::request(
            "Anthropic manual prompt caching requires a cache_control marker on the final \
             non-deferred tool, but explicit tool markers exhaust the available cache point budget",
        ));
    }

    tool.insert(
        "cache_control".to_string(),
        serde_json::to_value(cache_control)?,
    );
    *remaining_cache_markers -= 1;

    Ok(())
}

fn apply_system_cache_control(
    system: &mut [SystemContent],
    remaining_cache_markers: &mut usize,
    cache_control_value: &CacheControl,
) {
    if *remaining_cache_markers == 0 {
        return;
    }

    if let Some(SystemContent::Text { cache_control, .. }) = system.last_mut()
        && cache_control.is_none()
    {
        *cache_control = Some(cache_control_value.clone());
        *remaining_cache_markers -= 1;
    }
}

fn clear_message_cache_control(messages: &mut [Message]) {
    for msg in messages.iter_mut() {
        for content in msg.content.iter_mut() {
            set_content_cache_control(content, None);
        }
    }
}

fn apply_message_cache_control(
    messages: &mut [Message],
    remaining_cache_markers: &mut usize,
    cache_control: &CacheControl,
) {
    clear_message_cache_control(messages);

    if *remaining_cache_markers == 0 {
        return;
    }

    if let Some(last_msg) = messages.last_mut()
        && let Some(last_content) = last_msg.content.last_mut()
    {
        set_content_cache_control(last_content, Some(cache_control.clone()));
        *remaining_cache_markers -= 1;
    }
}

pub(super) fn apply_prompt_cache_control(
    system: &mut [SystemContent],
    messages: &mut [Message],
    tools: &mut [serde_json::Value],
    prompt_caching: bool,
    static_prefix_cache_ttl: Option<&CacheTtl>,
    top_level_cache_control: Option<&CacheControl>,
) -> Result<(), EncodeError> {
    normalize_tool_cache_control(tools);

    let max_cache_markers = if top_level_cache_control.is_some() {
        MAX_CACHE_CONTROL_MARKERS - 1
    } else {
        MAX_CACHE_CONTROL_MARKERS
    };
    let tool_cache_markers = tool_cache_control_count(tools);

    if tool_cache_markers > max_cache_markers {
        return Err(EncodeError::request(format!(
            "Too many Anthropic tool `cache_control` markers: {tool_cache_markers} exceeds \
                 the available prompt caching budget of {max_cache_markers}"
        )));
    }

    let mut remaining_cache_markers = max_cache_markers - tool_cache_markers;

    // Diagnose conflicting builder settings before generic TTL-order validation.
    let top_level_ttl = top_level_cache_control_ttl(top_level_cache_control);
    if static_prefix_cache_ttl == Some(&CacheTtl::FiveMinutes)
        && top_level_ttl == Some(CacheTtl::OneHour)
    {
        return Err(EncodeError::request(
            "`with_static_prefix_cache_ttl(CacheTtl::FiveMinutes)` conflicts with the 1-hour \
             top-level cache TTL (`with_automatic_caching_1h` or a raw top-level \
             `cache_control`): Anthropic requires 1h markers to precede 5-minute ones, and the \
             static prefix precedes the conversation tail",
        ));
    }

    // Manual prompt caching marks the prefix and the tail; a static-prefix TTL
    // alone marks just the prefix (the tail stays with the automatic/top-level
    // breakpoint, or uncached).
    if prompt_caching || static_prefix_cache_ttl.is_some() {
        let static_cache_control =
            build_cache_control(static_prefix_cache_ttl.cloned().or(top_level_ttl.clone()));

        apply_tool_cache_control(tools, &mut remaining_cache_markers, &static_cache_control)?;
        apply_system_cache_control(system, &mut remaining_cache_markers, &static_cache_control);
    }

    if prompt_caching {
        if top_level_cache_control.is_some() {
            clear_message_cache_control(messages);
        } else {
            let tail_cache_control = build_cache_control(top_level_ttl);
            apply_message_cache_control(
                messages,
                &mut remaining_cache_markers,
                &tail_cache_control,
            );
        }
    }

    validate_cache_control_ttl_order(system, messages, tools, top_level_cache_control)?;

    Ok(())
}

pub(super) fn extract_top_level_cache_control(
    additional_params: &mut serde_json::Value,
) -> Result<Option<CacheControl>, EncodeError> {
    if let Some(map) = additional_params.as_object_mut()
        && let Some(raw_cache_control) = map.remove("cache_control")
    {
        if raw_cache_control.is_null() {
            return Ok(None);
        }

        return serde_json::from_value::<CacheControl>(raw_cache_control)
            .map(Some)
            .map_err(|err| {
                EncodeError::request(format!(
                    "Invalid Anthropic `additional_params.cache_control` payload: {err}"
                ))
            });
    }

    Ok(None)
}

pub(super) fn resolve_top_level_cache_control(
    automatic_caching: bool,
    automatic_caching_ttl: Option<&CacheTtl>,
    additional_params: &mut serde_json::Value,
) -> Result<Option<CacheControl>, EncodeError> {
    let raw_cache_control = extract_top_level_cache_control(additional_params)?;
    let typed_cache_control = automatic_caching.then_some(CacheControl::Ephemeral {
        ttl: automatic_caching_ttl.cloned(),
    });

    match (typed_cache_control, raw_cache_control) {
        (Some(typed_cache_control), Some(raw_cache_control)) => {
            if automatic_caching_ttl.is_some()
                && cache_control_ttl(&typed_cache_control) != cache_control_ttl(&raw_cache_control)
            {
                return Err(EncodeError::request(
                    "Anthropic `additional_params.cache_control` conflicts with the typed \
                     automatic caching TTL",
                ));
            }

            Ok(Some(raw_cache_control))
        }
        (Some(typed_cache_control), None) => Ok(Some(typed_cache_control)),
        (None, raw_cache_control) => Ok(raw_cache_control),
    }
}

pub(super) fn split_system_messages_from_history(
    history: &[message::Message],
    preserve_mid_conversation_system_messages: bool,
) -> (Vec<SystemContent>, Vec<message::Message>) {
    let mut system = Vec::new();
    let mut remaining = Vec::new();

    for (index, message) in history.iter().enumerate() {
        match message {
            message::Message::System { content } => {
                if !content.is_empty() {
                    if preserve_mid_conversation_system_messages
                        && is_valid_mid_conversation_system_message(history, index)
                    {
                        remaining.push(message.clone());
                    } else {
                        system.push(SystemContent::Text {
                            text: content.clone(),
                            cache_control: None,
                        });
                    }
                }
            }
            other => remaining.push(other.clone()),
        }
    }

    (system, remaining)
}

fn is_valid_mid_conversation_system_message(history: &[message::Message], index: usize) -> bool {
    let follows_valid_turn = index > 0
        && history.get(index - 1).is_some_and(|message| {
            matches!(message, message::Message::User { .. })
                || assistant_ends_in_server_tool_block(message)
        });
    let is_last_or_precedes_assistant = history
        .get(index + 1)
        .is_none_or(|message| matches!(message, message::Message::Assistant { .. }));

    follows_valid_turn && is_last_or_precedes_assistant
}

fn assistant_ends_in_server_tool_block(message: &message::Message) -> bool {
    let message::Message::Assistant { content, .. } = message else {
        return false;
    };

    let Some(message::AssistantContent::Text(text)) = content.iter().last() else {
        return false;
    };

    let Some(raw_type) = text
        .additional_params
        .as_ref()
        .and_then(|params| params.get(ANTHROPIC_RAW_CONTENT_KEY))
        .and_then(|raw_content| raw_content.get("type"))
        .and_then(serde_json::Value::as_str)
    else {
        return false;
    };

    matches!(
        raw_type,
        "server_tool_use" | "web_search_tool_result" | "code_execution_tool_result"
    )
}

/// Parameters for building an AnthropicCompletionRequest
pub struct AnthropicRequestParams<'a> {
    pub model: &'a str,
    pub request: CompletionRequest,
    pub prompt_caching: bool,
    /// Add a top-level `cache_control` field for Anthropic's automatic caching mode.
    pub automatic_caching: bool,
    /// TTL for the top-level cache_control. `None` omits the `ttl` field (API default is 5 min).
    pub automatic_caching_ttl: Option<CacheTtl>,
    /// TTL for the static prefix (tools + system). `None` inherits the top-level TTL.
    pub static_prefix_cache_ttl: Option<CacheTtl>,
}

impl AnthropicCompletionRequest {
    /// Build the typed request, optionally transforming generated tools with `strict`.
    /// Reject missing token limits, invalid message conversions, and cache conflicts.
    pub(super) fn try_from_params(
        params: AnthropicRequestParams<'_>,
        strict: Option<fn(&mut ToolDefinition)>,
    ) -> Result<Self, EncodeError> {
        let AnthropicRequestParams {
            model,
            request: mut req,
            prompt_caching,
            automatic_caching,
            automatic_caching_ttl,
            static_prefix_cache_ttl,
        } = params;
        let chat_history = req.chat_history_with_documents();

        let Some(max_tokens) = req.max_tokens else {
            return Err(EncodeError::request(
                "`max_tokens` must be set for Anthropic",
            ));
        };

        let (history_system, chat_history) = split_system_messages_from_history(
            &chat_history,
            supports_mid_conversation_system_messages(model),
        );
        let mut full_history = vec![];
        full_history.extend(chat_history);

        let mut messages = full_history
            .iter()
            .cloned()
            .map(Message::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        // Server-tool references are preserved opaque content, not local calls.
        // Reserve their genuine handles so arbitrary local hints cannot alias them.
        let server_ids = messages
            .iter()
            .flat_map(|message| &message.content)
            .filter_map(|part| match part {
                Content::ServerToolUse { id, .. } => Some(id.clone()),
                Content::WebSearchToolResult { tool_use_id, .. }
                | Content::CodeExecutionToolResult { tool_use_id, .. } => Some(tool_use_id.clone()),
                _ => None,
            });
        let tool_ids = crate::providers::internal::tool_call_ids::ToolCallIds::with_reserved(
            &full_history,
            server_ids,
        )
        .map_err(EncodeError::request)?;
        for (position, message) in messages.iter_mut().enumerate() {
            tool_ids
                .apply(
                    position,
                    message.content.iter_mut().filter_map(|part| match part {
                        Content::ToolUse { id, .. } => Some(id),
                        Content::ToolResult { tool_use_id, .. } => Some(tool_use_id),
                        _ => None,
                    }),
                )
                .map_err(EncodeError::request)?;
        }

        let mut additional_params_payload = req
            .additional_params
            .take()
            .unwrap_or(serde_json::Value::Null);
        let top_level_cache_control = resolve_top_level_cache_control(
            automatic_caching,
            automatic_caching_ttl.as_ref(),
            &mut additional_params_payload,
        )?;
        let mut tools = build_tool_definitions(req.tools, &mut additional_params_payload, strict)?;

        let mut system = history_system;

        apply_prompt_cache_control(
            &mut system,
            &mut messages,
            &mut tools,
            prompt_caching,
            static_prefix_cache_ttl.as_ref(),
            top_level_cache_control.as_ref(),
        )?;

        let output_config = if let Some(schema) = req.output_schema {
            let mut schema_value = schema.to_value();
            sanitize_schema(&mut schema_value);
            Some(OutputConfig {
                format: OutputFormat::JsonSchema {
                    schema: schema_value,
                },
            })
        } else {
            None
        };

        Ok(Self {
            model: model.to_string(),
            messages,
            max_tokens,
            system,
            temperature: req.temperature,
            tool_choice: req.tool_choice.map(ToolChoice::try_from).transpose()?,
            tools,
            output_config,
            cache_control: top_level_cache_control,
            additional_params: if additional_params_payload.is_null() {
                None
            } else {
                Some(additional_params_payload)
            },
        })
    }
}

impl TryFrom<AnthropicRequestParams<'_>> for AnthropicCompletionRequest {
    type Error = EncodeError;

    fn try_from(params: AnthropicRequestParams<'_>) -> Result<Self, Self::Error> {
        Self::try_from_params(params, None)
    }
}

pub(super) fn extract_tools_from_additional_params(
    additional_params: &mut serde_json::Value,
) -> Result<Vec<serde_json::Value>, EncodeError> {
    if let Some(map) = additional_params.as_object_mut()
        && let Some(raw_tools) = map.remove("tools")
    {
        return serde_json::from_value::<Vec<serde_json::Value>>(raw_tools).map_err(|err| {
            EncodeError::request(format!(
                "Invalid Anthropic `additional_params.tools` payload: {err}"
            ))
        });
    }

    Ok(Vec::new())
}

pub(super) fn build_tool_definitions(
    tools: Vec<completion::ToolDefinition>,
    additional_params_payload: &mut serde_json::Value,
    strict: Option<fn(&mut ToolDefinition)>,
) -> Result<Vec<serde_json::Value>, EncodeError> {
    let mut additional_tools = extract_tools_from_additional_params(additional_params_payload)?;

    let mut tools = tools
        .into_iter()
        .map(|tool| {
            let input_schema = tool.parameters;
            let mut tool = ToolDefinition {
                name: tool.name,
                description: Some(tool.description),
                input_schema,
                strict: false,
                cache_control: None,
            };
            if let Some(strict) = strict {
                strict(&mut tool);
            }

            tool
        })
        .map(serde_json::to_value)
        .collect::<Result<Vec<_>, _>>()?;
    tools.append(&mut additional_tools);

    Ok(tools)
}

#[cfg(test)]
mod tests;
