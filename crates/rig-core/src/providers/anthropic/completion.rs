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
use crate::providers::internal::wire_ids::WireIds;
use crate::{
    completion,
    message::{self, DocumentMediaType, DocumentSourceKind, MessageError, MimeType},
};
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, str::FromStr};

/// Claude Fable 5.1, API ID `claude-fable-5-1`: 128K default `max_tokens`,
/// mid-conversation system messages kept in `messages`. It rejects a forced
/// tool choice, so the extractor uses native structured output instead of
/// forcing its `submit` tool.
pub const CLAUDE_FABLE_5_1: &str = "claude-fable-5-1";
/// Claude Opus 5.5, API ID `claude-opus-5-5`: 128K default `max_tokens`,
/// mid-conversation system messages kept in `messages`. It rejects a forced
/// tool choice, so the extractor uses native structured output instead of
/// forcing its `submit` tool.
pub const CLAUDE_OPUS_5_5: &str = "claude-opus-5-5";
/// Claude Sonnet 5.5, API ID `claude-sonnet-5-5`: 128K default `max_tokens`,
/// mid-conversation system messages kept in `messages` (unlike Claude
/// Sonnet 5). It rejects a forced tool choice, so the extractor uses native
/// structured output instead of forcing its `submit` tool.
pub const CLAUDE_SONNET_5_5: &str = "claude-sonnet-5-5";
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

/// A Messages reply, or the message `message_start` opens a stream with.
/// Each content block is kept as the provider sent it.
#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    pub content: Vec<ContentItem>,
    pub id: String,
    pub model: String,
    pub role: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    /// What stopped the model beyond `stop_reason`, such as a refusal's
    /// explanation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_details: Option<serde_json::Value>,
    /// The code-execution container the reply ran in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<serde_json::Value>,
    pub usage: Usage,
}

/// A response content block or delta as the provider sent it, checked to
/// carry the fields its known kind requires: a defective one fails to
/// deserialize, and a kind rig has never seen passes as it is.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ContentItem(pub serde_json::Map<String, serde_json::Value>);

impl ContentItem {
    /// The item's `type`.
    pub fn kind(&self) -> &str {
        self.str("type")
    }

    /// The string field `key`, empty when absent.
    pub fn str(&self, key: &str) -> &str {
        self.0
            .get(key)
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
    }
}

impl<'de> Deserialize<'de> for ContentItem {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let item = Self(serde_json::Map::deserialize(deserializer)?);
        let kind = item.0.get("type").and_then(serde_json::Value::as_str);
        let required: &[&str] = match kind {
            None => {
                return Err(D::Error::custom(
                    "an Anthropic content item needs a string `type`",
                ));
            }
            Some("text" | "text_delta") => &["text"],
            Some("thinking" | "thinking_delta") => &["thinking"],
            Some("redacted_thinking") => &["data"],
            Some("tool_use") => &["id", "name"],
            Some("signature_delta") => &["signature"],
            Some("input_json_delta") => &["partial_json"],
            Some(_) => &[],
        };
        if let Some(key) = required
            .iter()
            .find(|key| !item.0.get(**key).is_some_and(serde_json::Value::is_string))
        {
            return Err(D::Error::custom(format!(
                "Anthropic `{}` needs a string `{key}`",
                item.kind()
            )));
        }
        if item.kind() == "tool_use" && item.0.get("input").is_some_and(|input| !input.is_object())
        {
            return Err(D::Error::custom(
                "Anthropic `tool_use` needs an object `input`",
            ));
        }
        match (item.kind(), item.0.get("citations"), item.0.get("citation")) {
            ("text", Some(citations), _) if !citations.is_null() => {
                Vec::<Citation>::deserialize(citations).map_err(D::Error::custom)?;
            }
            ("citations_delta", _, citation) => {
                let citation = citation.ok_or_else(|| {
                    D::Error::custom("Anthropic `citations_delta` needs a `citation`")
                })?;
                Citation::deserialize(citation).map_err(D::Error::custom)?;
            }
            _ => {}
        }
        Ok(item)
    }
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

/// Anthropic's `usage`, as sent: `input_tokens` excludes the cache reads and
/// writes counted beside it, which rig's [`Usage`](crate::completion::Usage)
/// counts in its input.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Usage {
    /// Input tokens neither read from nor written to a cache.
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
            "Uncached input tokens: {}\nCache read input tokens: {}\nCache creation input tokens: {}\nOutput tokens: {}",
            self.input_tokens,
            self.cache_read_input_tokens
                .map_or_else(|| "n/a".to_string(), |token| token.to_string()),
            self.cache_creation_input_tokens
                .map_or_else(|| "n/a".to_string(), |token| token.to_string()),
            self.output_tokens
        )
    }
}

/// Rig's input is Anthropic's `input_tokens` plus its cache reads and writes,
/// its output `output_tokens` (thinking included), and its total their sum;
/// without an uncached input count, input and the total stay absent.
pub(super) fn anthropic_usage_totals(
    input_tokens: Option<u64>,
    output_tokens: u64,
    cache_read: Option<u64>,
    cache_creation: Option<u64>,
    output_tokens_details: Option<OutputTokensDetails>,
) -> crate::completion::Usage {
    let input_tokens = input_tokens
        .map(|uncached| uncached + cache_read.unwrap_or(0) + cache_creation.unwrap_or(0));
    crate::completion::Usage {
        input_tokens,
        output_tokens: Some(output_tokens),
        cached_input_tokens: cache_read,
        cache_creation_input_tokens: cache_creation,
        reasoning_tokens: output_tokens_details.map(|details| details.thinking_tokens),
        total_tokens: input_tokens.map(|input| input + output_tokens),
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

/// One request content block. An assistant block replayed to the model
/// that produced it is [`Content::Native`]: the provider's item verbatim.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    Text {
        text: String,
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
    /// A provider content block, sent as it arrived.
    #[serde(untagged)]
    Native(serde_json::Value),
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

/// Extract Anthropic-specific document fields (`title`, `context`, `citations`)
/// from the generic [`message::Document::additional_params`] object.
///
/// Return absent fields when parameters are missing. Ignore non-string title
/// and context values; reject a present, invalid [`CitationsConfig`].
fn extract_anthropic_doc_params(
    additional_params: Option<serde_json::Value>,
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

/// The citations Claude attached to an assistant text block, read from the
/// block's provider item.
///
/// Returns `Ok(vec![])` when the block carries none or was edited since it
/// was decoded. Unknown citation types are preserved as
/// [`Citation::Unknown`]; a known citation type with an invalid shape is an
/// error.
///
/// ```
/// use rig_core::completion::message::{AssistantContent, Text};
/// use rig_core::providers::anthropic::completion::anthropic_citations;
///
/// let block = AssistantContent::Text(Text::new("Rust is safe.")).with_native(serde_json::json!({
///     "type": "text",
///     "text": "Rust is safe.",
///     "citations": [],
/// }));
/// if let AssistantContent::Text(text) = &block {
///     assert!(anthropic_citations(text)?.is_empty());
/// }
/// # Ok::<(), serde_json::Error>(())
/// ```
pub fn anthropic_citations(text: &message::Text) -> Result<Vec<Citation>, serde_json::Error> {
    let block = message::AssistantContent::Text(text.clone());
    match block
        .native_item()
        .and_then(|item| item.get("citations"))
        .filter(|citations| !citations.is_null())
    {
        Some(citations) => <Vec<Citation> as serde::Deserialize>::deserialize(citations),
        None => Ok(Vec::new()),
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

/// One assistant block on the wire, `id` its call's spelling: the
/// provider's item while it is current, else the block rebuilt from its
/// canonical fields as pi rebuilds it. `None` for a block with nothing
/// Anthropic takes.
#[deny(clippy::wildcard_enum_match_arm)]
fn assistant_content(
    block: message::AssistantContent,
    id: Option<&str>,
) -> Result<Option<Content>, MessageError> {
    use message::AssistantContent as Block;
    // Anthropic rejects a blank text block, whatever produced it.
    if let Block::Text(text) = &block
        && text.text.trim().is_empty()
    {
        return Ok(None);
    }
    // Thinking without a signature is rejected as thinking; pi sends it as
    // text.
    if let Block::Reasoning(reasoning) = &block
        && !reasoning.redacted
        && block
            .native_item()
            .and_then(|item| item.get("signature"))
            .and_then(serde_json::Value::as_str)
            .is_none_or(|signature| signature.trim().is_empty())
    {
        let text = &reasoning.text;
        return Ok((!text.trim().is_empty()).then(|| Content::from(text.clone())));
    }
    if let Some(item) = block.native_item() {
        return Ok(Some(Content::Native(item.clone())));
    }
    Ok(match block {
        Block::Text(text) => Some(Content::from(text.text)),
        // A redacted block's payload lives only in its provider item.
        Block::Reasoning(_) => None,
        Block::ToolCall(call) => Some(Content::ToolUse {
            id: id.map_or_else(|| call.id.wire().into_owned(), str::to_owned),
            name: call.function.name.into(),
            input: serde_json::Value::Object(call.function.arguments),
        }),
        Block::Opaque(opaque) => Some(Content::Native(opaque.item)),
        Block::Image(_) => {
            return Err(MessageError::ConversionError(
                "Anthropic currently doesn't support images.".to_string(),
            ));
        }
    })
}

impl Message {
    /// `message` on the wire, its tool ids spelled as `ids` plans them for
    /// position `at` of the history. `None` when no block is left to send.
    fn from_message(
        message: message::Message,
        ids: &WireIds,
        at: usize,
    ) -> Result<Option<Self>, MessageError> {
        let (role, content) = match message {
            message::Message::System { content } => (Role::System, vec![Content::from(content)]),
            message::Message::User { content } => (
                Role::User,
                content
                    .into_iter()
                    .enumerate()
                    .filter_map(|(slot, part)| user_content(part, ids.get(at, slot)).transpose())
                    .collect::<Result<_, _>>()?,
            ),
            message::Message::Assistant(turn) => (
                Role::Assistant,
                turn.content
                    .into_iter()
                    .enumerate()
                    .filter_map(|(slot, block)| {
                        assistant_content(block, ids.get(at, slot)).transpose()
                    })
                    .collect::<Result<_, _>>()?,
            ),
        };
        Ok((!content.is_empty()).then_some(Self { role, content }))
    }

    /// Whether this is a user message of tool results only.
    fn is_tool_results(&self) -> bool {
        self.role == Role::User
            && self
                .content
                .iter()
                .all(|content| matches!(content, Content::ToolResult { .. }))
    }
}

/// One user block on the wire, `id` a tool result's call spelling. `None`
/// for blank text, which Anthropic rejects.
fn user_content(
    content: message::UserContent,
    id: Option<&str>,
) -> Result<Option<Content>, MessageError> {
    Ok(Some(match content {
        message::UserContent::Text(message::Text { text, .. }) => {
            if text.trim().is_empty() {
                return Ok(None);
            }
            Content::from(text)
        }
        message::UserContent::ToolResult(tool_result) => Content::ToolResult {
            tool_use_id: id.map_or_else(|| tool_result.call.wire().into_owned(), str::to_owned),
            content: tool_result
                .content
                .into_iter()
                .map(|content| match content {
                    message::ToolResultContent::Text(message::Text { text, .. }) => {
                        Ok(ToolResultContent::Text { text })
                    }
                    message::ToolResultContent::Json { value } => Ok(ToolResultContent::Text {
                        text: value.to_string(),
                    }),
                    message::ToolResultContent::Image(image) => {
                        let DocumentSourceKind::Base64(data) = image.data else {
                            return Err(MessageError::ConversionError(
                                "Only base64 strings can be used with the Anthropic API"
                                    .to_string(),
                            ));
                        };
                        let media_type = image.media_type.ok_or(MessageError::ConversionError(
                            "Image media type is required".to_owned(),
                        ))?;
                        Ok(ToolResultContent::Image {
                            source: ImageSource::Base64 {
                                data,
                                media_type: media_type.try_into()?,
                            },
                        })
                    }
                })
                .collect::<Result<Vec<_>, _>>()?,
            is_error: None,
            cache_control: None,
        },
        message::UserContent::Image(message::Image {
            data, media_type, ..
        }) => {
            let source = match data {
                DocumentSourceKind::Base64(data) => {
                    let media_type = media_type.ok_or(MessageError::ConversionError(
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
            Content::Image {
                source,
                cache_control: None,
            }
        }
        message::UserContent::Document(message::Document {
            data,
            media_type,
            additional_params,
        }) => {
            let (title, context, citations) = extract_anthropic_doc_params(additional_params)?;
            let source = document_source(data, media_type)?;
            Content::Document {
                source,
                title,
                context,
                citations,
                cache_control: None,
            }
        }
        message::UserContent::Audio { .. } => {
            return Err(MessageError::ConversionError(
                "Audio is not supported in Anthropic".to_owned(),
            ));
        }
        message::UserContent::Video { .. } => {
            return Err(MessageError::ConversionError(
                "Video is not supported in Anthropic".to_owned(),
            ));
        }
    }))
}

/// A document's source on the wire: a file id, a PDF or a plain text body.
fn document_source(
    data: DocumentSourceKind,
    media_type: Option<DocumentMediaType>,
) -> Result<DocumentSource, MessageError> {
    if let DocumentSourceKind::FileId(file_id) = data {
        return Ok(DocumentSource::File { file_id });
    }
    let media_type = match media_type {
        Some(media_type) => media_type,
        // Anthropic's URL document source has no media-type field and is
        // defined specifically for PDFs, so the source itself is sufficient.
        None if matches!(&data, DocumentSourceKind::Url(_)) => DocumentMediaType::PDF,
        None => {
            return Err(MessageError::ConversionError(
                "Document media type is required".to_string(),
            ));
        }
    };
    Ok(match media_type {
        DocumentMediaType::PDF => match data {
            DocumentSourceKind::Base64(data) | DocumentSourceKind::String(data) => {
                DocumentSource::Base64 {
                    data,
                    media_type: DocumentFormat::PDF,
                }
            }
            DocumentSourceKind::Url(url) => DocumentSource::Url { url },
            _ => {
                return Err(MessageError::ConversionError(
                    "Only base64 encoded data or URLs are supported for PDF documents".into(),
                ));
            }
        },
        DocumentMediaType::TXT => {
            let (DocumentSourceKind::String(data) | DocumentSourceKind::Base64(data)) = data else {
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
    })
}

/// Whether `model` is `id` or one of its dated snapshots (`<id>-YYYYMMDD`),
/// and not a later model whose ID merely starts with `id`.
pub(super) fn is_model(model: &str, id: &str) -> bool {
    model
        .strip_prefix(id)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with("-20"))
}

/// Models whose published synchronous output limit is 128K tokens.
const OUTPUT_128K: [&str; 10] = [
    CLAUDE_FABLE_5_1,
    CLAUDE_FABLE_5,
    CLAUDE_OPUS_5_5,
    CLAUDE_OPUS_5,
    CLAUDE_SONNET_5_5,
    CLAUDE_SONNET_5,
    CLAUDE_OPUS_4_8,
    CLAUDE_OPUS_4_7,
    CLAUDE_OPUS_4_6,
    CLAUDE_SONNET_4_6,
];

/// Models that accept `role: "system"` inside `messages`, per Anthropic's
/// mid-conversation system messages page. Claude Sonnet 5 does not.
const MID_CONVERSATION_SYSTEM: [&str; 6] = [
    CLAUDE_FABLE_5_1,
    CLAUDE_FABLE_5,
    CLAUDE_OPUS_5_5,
    CLAUDE_OPUS_5,
    CLAUDE_SONNET_5_5,
    CLAUDE_OPUS_4_8,
];

/// Return the published synchronous output limit for a recognized model.
/// Unknown models require an explicit `max_tokens` value.
pub(super) fn default_max_tokens_for_model(model: &str) -> Option<u64> {
    if OUTPUT_128K.iter().any(|id| is_model(model, id)) {
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

/// Models that answer a forced `tool_choice` (`any` or `tool`) with a 400, per
/// their what's-new pages.
const REJECTS_FORCED_TOOL_CHOICE: [&str; 3] =
    [CLAUDE_OPUS_5_5, CLAUDE_SONNET_5_5, CLAUDE_FABLE_5_1];

/// Whether `model` rejects a forced tool choice.
pub(super) fn rejects_forced_tool_choice(model: &str) -> bool {
    REJECTS_FORCED_TOOL_CHOICE
        .iter()
        .any(|id| is_model(model, id))
}

/// Whether `model` accepts `role: "system"` inside `messages`.
pub(super) fn supports_mid_conversation_system_messages(model: &str) -> bool {
    MID_CONVERSATION_SYSTEM.iter().any(|id| is_model(model, id))
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

                Self::Tool { name: name.into() }
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
    let Some(all_of) = root.shift_remove("allOf") else {
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
        root.shift_remove("$ref");

        for keyword in ["$defs", "definitions"] {
            let Some(root_definitions) = root.shift_remove(keyword) else {
                continue;
            };
            let definitions =
                merge_document_definitions(root_definitions, referenced.shift_remove(keyword));
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
                        let definitions = merge_document_definitions(
                            root_definitions,
                            branch.shift_remove(keyword),
                        );
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
        match properties.shift_remove(&name) {
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
        if let Some(definitions) = source.shift_remove(keyword) {
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

    if let Some(reference) = source.shift_remove("$ref") {
        strict.insert("$ref".to_string(), reference);
        return Value::Object(strict);
    }

    let schema_type = source.shift_remove("type");
    let any_of = source.shift_remove("anyOf");
    let one_of = source.shift_remove("oneOf");
    let all_of = source.shift_remove("allOf");
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

    if let Some(Value::Array(values)) = source.shift_remove("enum") {
        strict.insert("enum".to_string(), Value::Array(values));
    }
    if let Some(constant) = source.shift_remove("const") {
        strict.insert("const".to_string(), constant);
    }
    for keyword in ["description", "title"] {
        if let Some(Value::String(value)) = source.shift_remove(keyword) {
            strict.insert(keyword.to_string(), Value::String(value));
        }
    }

    let has_properties = source.contains_key("properties");
    let properties_imply_object = schema_type.is_none() && has_properties;
    if properties_imply_object {
        strict.insert("type".to_string(), Value::String("object".to_string()));
    }
    if schema_has_type(schema_type.as_ref(), "object") || has_properties {
        let properties = match source.shift_remove("properties") {
            Some(Value::Object(properties)) => properties
                .into_iter()
                .map(|(name, schema)| (name, transform_strict_tool_schema(schema)))
                .collect(),
            _ => Map::new(),
        };
        strict.insert("properties".to_string(), Value::Object(properties));
        source.shift_remove("additionalProperties");
        strict.insert("additionalProperties".to_string(), Value::Bool(false));
        if let Some(Value::Array(required)) = source.shift_remove("required") {
            strict.insert("required".to_string(), Value::Array(required));
        }
    }

    if schema_has_type(schema_type.as_ref(), "string")
        && let Some(format) = source.shift_remove("format")
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
        if let Some(items) = source.shift_remove("items") {
            strict.insert("items".to_string(), transform_strict_tool_schema(items));
        }
        if let Some(min_items) = source.shift_remove("minItems") {
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
            tool.shift_remove("cache_control");
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
        && let Some(raw_cache_control) = map.shift_remove("cache_control")
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

/// Split `history` into the top-level `system` field and `messages`.
///
/// On a model that takes mid-conversation system messages, one in a position
/// Anthropic rejects (after an assistant turn, or before a user turn) moves to
/// the next valid slot: right after the next user turn that ends the array or
/// precedes an assistant turn. Hoisting it into `system` instead would change
/// the prompt prefix, which misses the cache from the first token and, on
/// models that bind thinking blocks to their conversation, turns every earlier
/// thinking block into a 400. It is hoisted only when no such slot exists.
pub(super) fn split_system_messages_from_history(
    history: &[message::Message],
    preserve_mid_conversation_system_messages: bool,
) -> (Vec<SystemContent>, Vec<message::Message>) {
    let mut system = Vec::new();
    let mut remaining = Vec::new();
    let mut deferred: Vec<String> = Vec::new();

    for (index, message) in history.iter().enumerate() {
        match message {
            message::Message::System { content } => {
                if content.is_empty() {
                    continue;
                }
                if preserve_mid_conversation_system_messages {
                    if is_valid_mid_conversation_system_message(history, index) {
                        remaining.push(message.clone());
                        continue;
                    }
                    if index > 0 && next_system_message_slot(history, index).is_some() {
                        deferred.push(content.clone());
                        continue;
                    }
                }
                system.push(SystemContent::Text {
                    text: content.clone(),
                    cache_control: None,
                });
            }
            other => {
                remaining.push(other.clone());
                if !deferred.is_empty() && is_system_message_slot(history, index) {
                    remaining.push(message::Message::System {
                        content: std::mem::take(&mut deferred).join("\n\n"),
                    });
                }
            }
        }
    }

    (system, remaining)
}

/// The first index after `index` a misplaced system message can follow.
fn next_system_message_slot(history: &[message::Message], index: usize) -> Option<usize> {
    (index + 1..history.len()).find(|&slot| is_system_message_slot(history, slot))
}

/// Whether a system message may sit right after `history[index]`: it is a
/// user turn, and what follows is the end of the array or an assistant turn.
fn is_system_message_slot(history: &[message::Message], index: usize) -> bool {
    matches!(history.get(index), Some(message::Message::User { .. }))
        && history
            .get(index + 1)
            .is_none_or(|message| matches!(message, message::Message::Assistant { .. }))
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

/// Whether `message` is an assistant turn ending in a server tool call or
/// result, after which Anthropic takes a system message.
fn assistant_ends_in_server_tool_block(message: &message::Message) -> bool {
    let message::Message::Assistant(turn) = message else {
        return false;
    };
    matches!(
        turn.content.last(),
        Some(message::AssistantContent::Opaque(opaque))
            if opaque.kind().is_some_and(|kind| kind.ends_with("_tool_use") || kind.ends_with("_tool_result"))
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

        let (history_system, full_history) = split_system_messages_from_history(
            &chat_history,
            supports_mid_conversation_system_messages(model),
        );

        let ids = WireIds::new(&full_history);
        let mut messages: Vec<Message> = Vec::new();
        for (at, message) in full_history.into_iter().enumerate() {
            let Some(message) = Message::from_message(message, &ids, at)? else {
                continue;
            };
            // Consecutive tool results go in one user message, which Z.AI
            // requires (pi's rule).
            match messages.last_mut() {
                Some(last) if last.is_tool_results() && message.is_tool_results() => {
                    last.content.extend(message.content);
                }
                _ => messages.push(message),
            }
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
        && let Some(raw_tools) = map.shift_remove("tools")
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
                name: tool.name.into(),
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
