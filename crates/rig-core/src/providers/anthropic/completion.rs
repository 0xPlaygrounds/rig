//! Anthropic completion api implementation

use crate::completion::CompletionRequest;
use crate::completion::NormalizeCompletionResponse;
use crate::json_utils::string_or_vec;
use crate::providers::internal::completion_send::send_completion;
use crate::{
    client::Provider,
    completion::{self, CompletionError},
    http_client::HttpClientExt,
    message::{self, DocumentMediaType, DocumentSourceKind, MessageError, MimeType, Reasoning},
    telemetry::{CompletionOperation, CompletionSpanBuilder, ProviderResponseExt, SpanCombinator},
    wasm_compat::*,
};
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, str::FromStr};
use tracing::Instrument;

// ================================================================
// Anthropic Completion API
// ================================================================

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

pub trait AnthropicCompatibleProvider: Provider {
    const PROVIDER_NAME: &'static str;

    /// Response header carrying the provider's transport request id, when the
    /// provider reports one. Anthropic sends `request-id`; compatible gateways
    /// that mirror Anthropic's shape usually do too, and one that doesn't
    /// simply yields `None` — never an error.
    const REQUEST_ID_HEADER: Option<&'static str> = Some("request-id");

    fn default_max_tokens(model: &str) -> Option<u64> {
        let _ = model;
        None
    }

    /// Apply provider-specific strict tool-use behavior to a Rig-generated tool.
    ///
    /// Anthropic-compatible gateways do not necessarily implement Anthropic's
    /// constrained tool schemas, so the default deliberately leaves tools
    /// unchanged.
    fn enable_strict_tool_use(_tool: &mut ToolDefinition) {}
}

impl AnthropicCompatibleProvider for super::client::AnthropicExt {
    const PROVIDER_NAME: &'static str = "anthropic";

    fn default_max_tokens(model: &str) -> Option<u64> {
        default_max_tokens_for_model(model)
    }

    fn enable_strict_tool_use(tool: &mut ToolDefinition) {
        sanitize_strict_tool_schema(&mut tool.input_schema);
        tool.strict = true;
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    pub content: Vec<Content>,
    pub id: String,
    pub model: String,
    pub role: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
    /// The transport request id from the `request-id` response header — not
    /// part of the response body; stamped by the request driver. This is the
    /// id Anthropic support asks for. `None` when the provider (or a
    /// compatible gateway) did not report one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
}

/// Map an Anthropic Messages `stop_reason` onto the normalized vocabulary,
/// preserving anything unrecognized verbatim.
///
/// Shared by the unary and streaming paths so both agree, and so a stop reason
/// Anthropic adds later surfaces in its own spelling rather than reading as a
/// natural stop.
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

impl ProviderResponseExt for CompletionResponse {
    type Usage = Usage;

    fn get_response_id(&self) -> Option<String> {
        Some(self.id.to_owned())
    }

    fn get_response_model_name(&self) -> Option<String> {
        Some(self.model.to_owned())
    }

    fn get_text_response(&self) -> Option<String> {
        let res = self
            .content
            .iter()
            .filter_map(|x| {
                if let Content::Text { text, .. } = x {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<String>();

        if res.is_empty() { None } else { Some(res) }
    }

    fn get_usage(&self) -> Option<Self::Usage> {
        Some(self.usage.clone())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

/// Breakdown of `usage.output_tokens`.
///
/// The tokens Claude spent on extended thinking are reported here, *inside*
/// `output_tokens` rather than beside it — the name says `details`, and every
/// recorded turn has `thinking_tokens <= output_tokens`. Adding them to a total
/// would double-count. Unknown buckets a provider may add later are ignored on
/// deserialization.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct OutputTokensDetails {
    /// Output tokens spent on extended thinking this turn.
    #[serde(default)]
    pub thinking_tokens: u64,
}

/// Per-TTL breakdown of cache-write tokens (`usage.cache_creation`).
///
/// Distinguishes 1-hour cache writes (~2x base input token price) from
/// 5-minute writes (~1.25x), which is what makes a mixed-TTL configuration
/// (see [`CompletionModel::with_static_prefix_cache_ttl`]) observable.
/// Unknown buckets a provider may add later are ignored on deserialization.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
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
            match self.cache_read_input_tokens {
                Some(token) => token.to_string(),
                None => "n/a".to_string(),
            },
            match self.cache_creation_input_tokens {
                Some(token) => token.to_string(),
                None => "n/a".to_string(),
            },
            self.output_tokens
        )
    }
}

/// Aggregate an Anthropic token report into rig's usage shape.
///
/// Anthropic reports cache reads and cache writes *alongside* `input_tokens`
/// rather than inside it, so the total is the sum of all four counters.
/// `thinking_tokens` is the exception: it is a *breakdown* of `output_tokens`,
/// already counted there, so it populates `reasoning_tokens` without entering
/// the total. Shared with the streaming path, whose `PartialUsage` carries the
/// same counters — the parameter is required rather than defaulted so a new
/// caller cannot silently drop it.
pub(super) fn anthropic_usage_totals(
    input_tokens: u64,
    output_tokens: u64,
    cache_read: Option<u64>,
    cache_creation: Option<u64>,
    output_tokens_details: Option<OutputTokensDetails>,
) -> crate::completion::Usage {
    let mut usage = crate::completion::Usage::new();

    usage.input_tokens = input_tokens;
    usage.output_tokens = output_tokens;
    usage.cached_input_tokens = cache_read.unwrap_or_default();
    usage.cache_creation_input_tokens = cache_creation.unwrap_or_default();
    usage.reasoning_tokens = output_tokens_details
        .map(|details| details.thinking_tokens)
        .unwrap_or_default();
    usage.total_tokens = usage.input_tokens
        + usage.cached_input_tokens
        + usage.cache_creation_input_tokens
        + usage.output_tokens;

    usage
}

impl From<&Usage> for crate::completion::Usage {
    fn from(value: &Usage) -> crate::completion::Usage {
        anthropic_usage_totals(
            value.input_tokens,
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
    #[serde(default, skip_serializing_if = "is_false")]
    pub strict: bool,
    /// Cache breakpoint marker. Set on the last tool in the array to cache
    /// the tools layer independently of the system prompt. Anthropic accepts
    /// up to 4 `cache_control` markers per request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

fn is_false(value: &bool) -> bool {
    !value
}

/// TTL for a cache control breakpoint.
///
/// The Anthropic API supports two TTL values:
/// - `"5m"` — 5 minutes (default when `ttl` is omitted)
/// - `"1h"` — 1 hour
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

/// Normalize an Anthropic Messages response.
///
/// The provider descriptor name is an *input* rather than a constant: this same
/// wire shape is served by every Anthropic-compatible provider (MiniMax, Z.ai,
/// Moonshot, Xiaomi MiMo), so baking in `"anthropic"` here would mislabel all of
/// them. Taking it as part of the conversion makes the correct name impossible
/// to forget.
impl crate::completion::NormalizeCompletionResponse for CompletionResponse {
    fn normalize(self, provider: &str) -> Result<completion::CompletionResponse, CompletionError> {
        let response = self;
        let content = response
            .content
            .iter()
            .map(|content| content.clone().try_into())
            .collect::<Result<Vec<_>, _>>()?;

        // Anthropic has two ways to end a turn that genuinely carried no
        // content, and an empty list says exactly that:
        //
        // - `end_turn` after a tool-result round trip — documented, and it
        //   used to be normalized into a fabricated empty-text part.
        // - `stop_sequence` when the matched sequence is the first thing the
        //   model emits. Anthropic strips the sequence it stopped on, so a
        //   turn that produced nothing before it arrives with `content: []`
        //   and a 200. Rejecting that turned a completed provider turn into
        //   `EMPTY_RESPONSE_ERROR`, and diverged from the streamed twin,
        //   which finishes the same turn with an empty choice and no error.
        //
        // The `stop_sequence` arm additionally requires the sequence itself.
        // Every recorded stop-sequence turn names the sequence that fired, so
        // that is the full extent of the evidence; a turn claiming to have
        // stopped on a sequence while naming none is the malformed shape this
        // guard exists for, not a legal empty turn. This matters most for the
        // Anthropic-compatible gateways sharing this mapping, which are the
        // likeliest to report a stop reason without its companion field.
        //
        // Note this narrow shape — empty, `stop_sequence`, no sequence named —
        // is one the streaming path still finishes cleanly, since it has no
        // equivalent guard. That asymmetry is deliberate: the parity this
        // carve-out restores is for *legal* turns, and widening it to keep a
        // malformed one symmetric would trade a real guard for a cosmetic
        // match.
        //
        // Any *other* empty response is the shared provider defect.
        let legal_empty_turn = match response.stop_reason.as_deref() {
            Some("end_turn") => true,
            Some("stop_sequence") => response.stop_sequence.is_some(),
            _ => false,
        };
        let choice = if content.is_empty() && legal_empty_turn {
            Vec::new()
        } else {
            crate::message::require_non_empty_response(content)?
        };

        let finish_reason = response.stop_reason.as_deref().map(map_finish_reason);

        Ok(completion::CompletionResponse::new(
            choice,
            crate::completion::Usage::from(&response.usage),
            provider,
        )
        .with_optional_message_id(Some(response.id.as_str()).filter(|id| !id.is_empty()))
        .with_optional_provider_request_id(response.provider_request_id.clone())
        .with_model(response.model.as_str())
        .with_optional_finish_reason(finish_reason))
    }
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

/// Configuration for enabling citations on a document content block.
///
/// When enabled, Claude returns citation metadata on response text blocks,
/// allowing applications to track where each piece of information in the
/// response came from. See the [Anthropic citations documentation][docs] for
/// details on the request/response shapes.
///
/// Citations must be enabled on **all or none** of the documents in a request —
/// the API returns an error if the setting is mixed.
///
/// [docs]: https://docs.anthropic.com/en/docs/build-with-claude/citations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CitationsConfig {
    /// Whether citation tracking is enabled for this document.
    pub enabled: bool,
}

/// A citation returned by Claude pointing back to source text.
///
/// The variant determines the locator shape, which depends on the source type:
///
/// - [`Citation::CharLocation`] — for plain text documents; character indices
///   are 0-indexed with an exclusive end.
/// - [`Citation::PageLocation`] — for PDF documents; page numbers are 1-indexed
///   with an exclusive end.
/// - [`Citation::ContentBlockLocation`] — for custom-content documents; block
///   indices are 0-indexed with an exclusive end.
/// - [`Citation::SearchResultLocation`] — for user-provided search-result
///   content blocks.
/// - [`Citation::WebSearchResultLocation`] — for Anthropic's server-side web
///   search tool results.
/// - [`Citation::Unknown`] — a forward-compatible fallback preserving raw
///   citation JSON for citation types this crate does not yet model.
///
/// See the [Anthropic citations documentation][docs] for the exact wire format.
///
/// [docs]: https://docs.anthropic.com/en/docs/build-with-claude/citations
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
    /// Title of the cited source. Unlike the document-citation titles this
    /// carries no `skip_serializing_if`: the wire writes it even when absent,
    /// as an explicit `"title": null`.
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

        // Hand-written tag dispatch rather than `#[serde(untagged)]`: an
        // untagged fallback would swallow a *modeled* citation type carrying a
        // defective payload as `Unknown`, hiding provider drift. Only an
        // absent, non-string or unmodeled tag reaches `Unknown` here.
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

/// Deserialize a `Vec<T>`, treating an explicit JSON `null` as an empty vec.
///
/// `#[serde(default)]` only fills in a *missing* field, but the Anthropic
/// Messages API emits an explicit `"citations": null` on text
/// `content_block_start` events. Without this, `Vec` deserialization rejects the
/// null and the whole stream fails before any text arrives.
fn null_as_empty_vec<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    Ok(Option::<Vec<T>>::deserialize(deserializer)?.unwrap_or_default())
}

/// Decoded Anthropic document fields lifted out of [`message::Document::additional_params`]:
/// optional `title`, optional `context`, and optional [`CitationsConfig`].
type AnthropicDocParams = (Option<String>, Option<String>, Option<CitationsConfig>);

/// Extract Anthropic-specific document fields (`title`, `context`, `citations`)
/// from the generic [`message::Document::additional_params`] JSON blob.
///
/// Returns `Ok((None, None, None))` if `additional_params` is empty. Returns
/// an error only if the `citations` field is present but is not a valid
/// [`CitationsConfig`] — invalid shapes are reported instead of being silently
/// dropped, so users notice typos.
fn extract_anthropic_doc_params(
    additional_params: Option<message::AdditionalParams>,
) -> Result<AnthropicDocParams, MessageError> {
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
/// # Example
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
        Some(c) => serde_json::from_value::<Vec<Citation>>(c.clone()),
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

    let content = serde_json::from_value::<Content>(raw_content.clone()).map_err(|err| {
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

fn anthropic_raw_content_to_message_text(content: Content) -> Result<message::Text, MessageError> {
    let raw_content = serde_json::to_value(content).map_err(|err| {
        MessageError::ConversionError(format!("Failed to preserve Anthropic content block: {err}"))
    })?;

    Ok(message::Text {
        text: String::new(),
        additional_params: message::AdditionalParams::from_entries([(
            ANTHROPIC_RAW_CONTENT_KEY,
            raw_content,
        )]),
    })
}

fn anthropic_document_additional_params(
    title: Option<String>,
    context: Option<String>,
    citations: Option<CitationsConfig>,
) -> Result<Option<message::AdditionalParams>, MessageError> {
    let mut params = serde_json::Map::new();

    if let Some(title) = title {
        params.insert("title".to_string(), serde_json::Value::String(title));
    }
    if let Some(context) = context {
        params.insert("context".to_string(), serde_json::Value::String(context));
    }
    if let Some(citations) = citations {
        params.insert(
            "citations".to_string(),
            serde_json::to_value(citations).map_err(|err| {
                MessageError::ConversionError(format!(
                    "Failed to preserve Anthropic document citations metadata: {err}"
                ))
            })?,
        );
    }

    // The canonical constructor: an empty map is stored as `None`, never as
    // an empty carrier.
    Ok(message::AdditionalParams::new(params))
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

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SourceType {
    BASE64,
    URL,
    TEXT,
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

impl TryFrom<message::ContentFormat> for SourceType {
    type Error = MessageError;

    fn try_from(format: message::ContentFormat) -> Result<Self, Self::Error> {
        match format {
            message::ContentFormat::Base64 => Ok(SourceType::BASE64),
            message::ContentFormat::Url => Ok(SourceType::URL),
            message::ContentFormat::String => Ok(SourceType::TEXT),
        }
    }
}

impl From<SourceType> for message::ContentFormat {
    fn from(source_type: SourceType) -> Self {
        match source_type {
            SourceType::BASE64 => message::ContentFormat::Base64,
            SourceType::URL => message::ContentFormat::Url,
            SourceType::TEXT => message::ContentFormat::String,
        }
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
                return Err(MessageError::ConversionError(
                    format!("Unsupported image media type: {media_type:?}").to_owned(),
                ));
            }
        })
    }
}

impl From<ImageFormat> for message::ImageMediaType {
    fn from(format: ImageFormat) -> Self {
        match format {
            ImageFormat::JPEG => message::ImageMediaType::JPEG,
            ImageFormat::PNG => message::ImageMediaType::PNG,
            ImageFormat::GIF => message::ImageMediaType::GIF,
            ImageFormat::WEBP => message::ImageMediaType::WEBP,
        }
    }
}

impl TryFrom<DocumentMediaType> for DocumentFormat {
    type Error = MessageError;
    fn try_from(value: DocumentMediaType) -> Result<Self, Self::Error> {
        match value {
            DocumentMediaType::PDF => Ok(DocumentFormat::PDF),
            other => Err(MessageError::ConversionError(format!(
                "DocumentFormat only supports PDF for base64 sources, got: {}",
                other.to_mime_type()
            ))),
        }
    }
}

/// The Anthropic Messages API requires `tool_use.input` to be a JSON OBJECT.
/// `ToolCall.function.arguments` can arrive as a JSON-encoded STRING (some
/// providers / replayed conversation history) or as `null`/empty (a tool called
/// with no arguments); sending any of those verbatim is rejected with
/// `messages.N.content.M.tool_use.input: Input should be a valid dictionary` (a
/// deterministic 400 that breaks every multi-turn tool conversation, e.g. on the
/// managed / MiniMax anthropic-shaped endpoint). Coerce to an object at the send
/// boundary so the contract holds regardless of how `arguments` was built. This
/// re-adds the fork's tool_use.input invariant that a rig version bump dropped;
/// the server-tool path already guards empty input in streaming.rs.
fn coerce_tool_input(input: serde_json::Value) -> serde_json::Value {
    match input {
        v @ serde_json::Value::Object(_) => v,
        serde_json::Value::String(s) => match serde_json::from_str::<serde_json::Value>(&s) {
            Ok(serde_json::Value::Object(m)) => serde_json::Value::Object(m),
            _ => serde_json::json!({}),
        },
        // null / array / number / bool: no valid object form -> empty args.
        _ => serde_json::json!({}),
    }
}

fn anthropic_content_from_assistant_content(
    content: message::AssistantContent,
) -> Result<Vec<Content>, MessageError> {
    match content {
        message::AssistantContent::Text(text) => {
            // The same empty-block rule the Responses serializer applies:
            // the API rejects empty text blocks, so an empty text block
            // with no anthropic-deliverable content (raw server-tool
            // content is the one extras shape this wire replays; foreign
            // extras — e.g. a block annotated by the OpenAI Responses
            // ingest — cannot reach this wire) produces no block at all.
            // A message left with no blocks fails loudly and locally at
            // the non-empty check below, never as a wire 400.
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
            id: tool_call.wire_call_id().to_owned(),
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
                        tool_use_id: tool_result.wire_call_id().to_owned(),
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
                                let data = match data {
                                    DocumentSourceKind::String(data)
                                    | DocumentSourceKind::Base64(data) => data,
                                    _ => {
                                        return Err(MessageError::ConversionError(
                                            "Only string or base64 data is supported for plain text documents".into(),
                                        ));
                                    }
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

impl TryFrom<Content> for message::AssistantContent {
    type Error = MessageError;

    fn try_from(content: Content) -> Result<Self, Self::Error> {
        Ok(match content {
            // Keep this destructuring exhaustive so new wire fields force an
            // explicit capture-or-drop decision. `cache_control` is a
            // request-side directive, deliberately dropped on response
            // ingest.
            Content::Text {
                text,
                citations,
                cache_control: _,
            } => {
                // Preserve citation metadata on the generic text block via
                // `additional_params` so callers going through the generic
                // `AssistantContent` surface can still recover them (see
                // [`anthropic_citations`]).
                let additional_params = message::AdditionalParams::from_entries(
                    (!citations.is_empty()).then(|| ("citations", serde_json::json!(citations))),
                );
                message::AssistantContent::Text(message::Text {
                    text,
                    additional_params,
                })
            }
            Content::ToolUse { id, name, input } => {
                message::AssistantContent::tool_call(id, name, input)
            }
            raw @ (Content::ServerToolUse { .. }
            | Content::WebSearchToolResult { .. }
            | Content::CodeExecutionToolResult { .. }) => {
                message::AssistantContent::Text(anthropic_raw_content_to_message_text(raw)?)
            }
            Content::Thinking {
                thinking,
                signature,
            } => message::AssistantContent::Reasoning(Reasoning::new_with_signature(
                &thinking, signature,
            )),
            Content::RedactedThinking { data } => {
                message::AssistantContent::Reasoning(Reasoning::redacted(data))
            }
            _ => {
                return Err(MessageError::ConversionError(
                    "Content did not contain a message, tool call, or reasoning".to_owned(),
                ));
            }
        })
    }
}

impl From<ToolResultContent> for message::ToolResultContent {
    fn from(content: ToolResultContent) -> Self {
        match content {
            ToolResultContent::Text { text, .. } => message::ToolResultContent::text(text),
            ToolResultContent::Image { source } => match source {
                ImageSource::Base64 { data, media_type } => {
                    message::ToolResultContent::image_base64(data, Some(media_type.into()), None)
                }
                ImageSource::Url { url } => message::ToolResultContent::image_url(url, None, None),
            },
        }
    }
}

impl TryFrom<Message> for message::Message {
    type Error = MessageError;

    fn try_from(message: Message) -> Result<Self, Self::Error> {
        Ok(match message.role {
            Role::User => message::Message::User {
                content: message
                    .content
                    .into_iter()
                    .map(|content| {
                        Ok(match content {
                            Content::Text { text, .. } => message::UserContent::text(text),
                            Content::ToolResult {
                                tool_use_id,
                                content,
                                ..
                            } => message::UserContent::tool_result_from_wire(
                                tool_use_id,
                                // Anthropic's wire correlates results by id only
                                // and never carries the tool name; this
                                // conversion is lossy for name-keyed wires.
                                "",
                                content.into_iter().map(|content| content.into()).collect(),
                            ),
                            Content::Image { source, .. } => match source {
                                ImageSource::Base64 { data, media_type } => {
                                    message::UserContent::Image(message::Image {
                                        data: DocumentSourceKind::Base64(data),
                                        media_type: Some(media_type.into()),
                                        detail: None,
                                        additional_params: None,
                                    })
                                }
                                ImageSource::Url { url } => {
                                    message::UserContent::Image(message::Image {
                                        data: DocumentSourceKind::Url(url),
                                        media_type: None,
                                        detail: None,
                                        additional_params: None,
                                    })
                                }
                            },
                            Content::Document {
                                source,
                                title,
                                context,
                                citations,
                                ..
                            } => {
                                let additional_params = anthropic_document_additional_params(
                                    title, context, citations,
                                )?;

                                match source {
                                    DocumentSource::Base64 { data, media_type } => {
                                        let rig_media_type = match media_type {
                                            DocumentFormat::PDF => message::DocumentMediaType::PDF,
                                        };
                                        message::UserContent::Document(message::Document {
                                            data: DocumentSourceKind::String(data),
                                            media_type: Some(rig_media_type),
                                            additional_params,
                                        })
                                    }
                                    DocumentSource::Text { data, .. } => {
                                        message::UserContent::Document(message::Document {
                                            data: DocumentSourceKind::String(data),
                                            media_type: Some(message::DocumentMediaType::TXT),
                                            additional_params,
                                        })
                                    }
                                    DocumentSource::Url { url } => {
                                        message::UserContent::Document(message::Document {
                                            data: DocumentSourceKind::Url(url),
                                            media_type: None,
                                            additional_params,
                                        })
                                    }
                                    DocumentSource::File { file_id } => {
                                        message::UserContent::Document(message::Document {
                                            data: DocumentSourceKind::FileId(file_id),
                                            media_type: None,
                                            additional_params,
                                        })
                                    }
                                }
                            }
                            _ => {
                                return Err(MessageError::ConversionError(
                                    "Unsupported content type for User role".to_owned(),
                                ));
                            }
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            },
            Role::Assistant => message::Message::Assistant {
                id: None,
                content: message
                    .content
                    .into_iter()
                    .map(|content| content.try_into())
                    .collect::<Result<Vec<_>, _>>()?,
            },
            Role::System => {
                let content =
                    message
                        .content
                        .into_iter()
                        .try_fold(String::new(), |mut content, block| {
                            let Content::Text { text, .. } = block else {
                                return Err(MessageError::ConversionError(
                                    "Unsupported content type for System role".to_owned(),
                                ));
                            };

                            content.push_str(&text);
                            Ok(content)
                        })?;

                message::Message::System { content }
            }
        })
    }
}

#[doc(hidden)]
#[derive(Clone)]
pub struct GenericCompletionModel<Ext = super::client::AnthropicExt, T = reqwest::Client> {
    pub(crate) client: crate::client::Client<Ext, T>,
    pub model: String,
    pub default_max_tokens: Option<u64>,
    /// Enable manual prompt caching (adds cache_control breakpoints to system prompt,
    /// tools, and messages)
    pub prompt_caching: bool,
    /// Enable Anthropic's automatic prompt caching (adds a top-level `cache_control` field to the
    /// request). The API automatically places the breakpoint on the last cacheable block and moves
    /// it forward as the conversation grows. No beta header is required.
    pub automatic_caching: bool,
    /// TTL for automatic caching. `None` uses the API default (5 minutes).
    /// Set to `Some(CacheTtl::OneHour)` for a 1-hour TTL.
    pub automatic_caching_ttl: Option<CacheTtl>,
    /// TTL for the static prefix (tool definitions + system prompt),
    /// independent of the conversation-tail breakpoint. `None` inherits the
    /// top-level/automatic TTL.
    pub static_prefix_cache_ttl: Option<CacheTtl>,
    /// Whether Rig-generated tools request provider-supported strict validation.
    pub strict_tools: bool,
}

/// Anthropic completion model.
///
/// This preserves the historical public generic shape where the first generic
/// parameter is the HTTP client type.
pub type CompletionModel<T = reqwest::Client> =
    GenericCompletionModel<super::client::AnthropicExt, T>;

impl<Ext, T> GenericCompletionModel<Ext, T>
where
    T: HttpClientExt,
    Ext: AnthropicCompatibleProvider + Clone + 'static,
{
    /// The request prelude both the unary and the streaming path run: resolve
    /// the model, open the telemetry span, default `max_tokens`, and build the
    /// typed request.
    ///
    /// The span is built *before* `max_tokens` defaulting so a request rejected
    /// for a missing limit is still attributable to a model and operation. The
    /// caller applies the span its own way — `.instrument(..)` around the unary
    /// send, handed to the SSE transport when streaming.
    ///
    /// Each caller TRACE-logs its own final body rather than this typed request:
    /// the streaming body is this request plus `stream` and a reconciled
    /// `tool_choice`, and those two fields are the whole reason the streaming
    /// path logs at all.
    pub(super) fn prepare_request(
        &self,
        mut completion_request: completion::CompletionRequest,
        operation: CompletionOperation,
    ) -> Result<(tracing::Span, AnthropicCompletionRequest), CompletionError> {
        let request_model = completion_request
            .model
            .clone()
            .unwrap_or_else(|| self.model.clone());
        let span = CompletionSpanBuilder::new(Ext::PROVIDER_NAME, &request_model, operation)
            .system_instructions(
                completion_request.preamble.as_deref(),
                completion_request.record_telemetry_content,
            )
            .build();

        if completion_request.max_tokens.is_none() {
            let Some(tokens) = self.default_max_tokens else {
                return Err(CompletionError::RequestError(
                    "`max_tokens` must be set for Anthropic".into(),
                ));
            };
            completion_request.max_tokens = Some(tokens);
        }

        let request = AnthropicCompletionRequest::try_from_params::<Ext>(
            AnthropicRequestParams {
                model: &request_model,
                request: completion_request,
                prompt_caching: self.prompt_caching,
                automatic_caching: self.automatic_caching,
                automatic_caching_ttl: self.automatic_caching_ttl.clone(),
                static_prefix_cache_ttl: self.static_prefix_cache_ttl.clone(),
            },
            self.strict_tools,
        )?;

        Ok((span, request))
    }

    /// A model with every caching / strictness knob off, differing from its
    /// siblings only in how `default_max_tokens` was resolved.
    fn with_defaults(
        client: crate::client::Client<Ext, T>,
        model: String,
        default_max_tokens: Option<u64>,
    ) -> Self {
        Self {
            client,
            model,
            default_max_tokens,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
            strict_tools: false,
        }
    }

    pub fn new(client: crate::client::Client<Ext, T>, model: impl Into<String>) -> Self {
        let model = model.into();
        let default_max_tokens = Ext::default_max_tokens(&model);

        Self::with_defaults(client, model, default_max_tokens)
    }

    pub fn with_model(client: crate::client::Client<Ext, T>, model: &str) -> Self {
        let default_max_tokens = Ext::default_max_tokens(model)
            .or_else(|| Some(default_max_tokens_with_fallback(model)));

        Self::with_defaults(client, model.to_string(), default_max_tokens)
    }

    /// Enable manual prompt caching.
    ///
    /// When enabled, cache_control breakpoints are automatically added to:
    /// - The system prompt (marked with ephemeral cache)
    /// - The final tool definition, when tools are present (marked with ephemeral cache)
    /// - The last content block of the last message (marked with ephemeral cache)
    ///
    /// This allows Anthropic to cache the system prompt, tools layer, and conversation
    /// history for cost savings. Use [`with_automatic_caching`] when you want Anthropic
    /// to choose and advance a single top-level cache breakpoint automatically.
    /// When combined with [`with_automatic_caching`], the top-level automatic breakpoint
    /// owns the moving message cache point while Rig still marks tools and system prompt
    /// blocks when budget permits.
    /// Existing `cache_control` markers in provider-specific tool definitions are preserved
    /// and count toward Anthropic's request limit of 4 cache breakpoints.
    ///
    /// [`with_automatic_caching`]: CompletionModel::with_automatic_caching
    pub fn with_prompt_caching(mut self) -> Self {
        self.prompt_caching = true;
        self
    }

    /// Enable Anthropic's automatic prompt caching.
    ///
    /// When enabled, a top-level `cache_control: { "type": "ephemeral" }` field is added to every
    /// request. Anthropic's API automatically applies the cache breakpoint to the last cacheable
    /// block and moves it forward as the conversation grows — no beta header and no manual
    /// breakpoint management are required.
    ///
    /// This is the recommended approach for multi-turn conversations. Use [`with_prompt_caching`]
    /// instead when you need fine-grained, per-block control over what is cached.
    ///
    /// To use a one-hour TTL instead of the default five minutes, use
    /// [`with_automatic_caching_1h`] or pass top-level `cache_control` with
    /// `ttl: "1h"` via `additional_params`. Rig normalizes raw top-level
    /// `cache_control` before budgeting and ordering manual prompt cache markers.
    ///
    /// ```ignore
    /// let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
    ///     .with_automatic_caching();
    /// ```
    ///
    /// ## Minimum cacheable prompt length
    ///
    /// The combined prompt (tools + system + messages up to the automatically chosen breakpoint)
    /// must meet the model-specific minimum or caching is silently skipped by the API:
    ///
    /// | Model | Minimum tokens |
    /// |-------|---------------|
    /// | `claude-opus-4-7`, `claude-opus-4-6`, `claude-opus-4-5` | 4 096 |
    /// | `claude-sonnet-4-6` | 2 048 |
    /// | `claude-sonnet-4-5`, `claude-opus-4-1`, `claude-opus-4`, `claude-sonnet-4` | 1 024 |
    /// | `claude-haiku-4-5` | 4 096 |
    ///
    /// [`with_prompt_caching`]: CompletionModel::with_prompt_caching
    /// [`with_automatic_caching_1h`]: CompletionModel::with_automatic_caching_1h
    pub fn with_automatic_caching(mut self) -> Self {
        self.automatic_caching = true;
        self
    }

    /// Enable Anthropic's automatic prompt caching with a 1-hour TTL.
    ///
    /// Identical to [`with_automatic_caching`] but sets `ttl: "1h"` on the
    /// top-level `cache_control` field:
    ///
    /// ```ignore
    /// let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
    ///     .with_automatic_caching_1h();
    /// ```
    ///
    /// [`with_automatic_caching`]: CompletionModel::with_automatic_caching
    pub fn with_automatic_caching_1h(mut self) -> Self {
        self.automatic_caching = true;
        self.automatic_caching_ttl = Some(CacheTtl::OneHour);
        self
    }

    /// Set the cache TTL for the static prefix (tool definitions + system
    /// prompt), independent of the moving conversation-tail breakpoint.
    ///
    /// An agent's prompt has two parts with very different volatility: the
    /// system prompt and tool definitions are byte-identical across sessions,
    /// while the conversation tail changes every turn and is worthless an hour
    /// later. A 1-hour cache write costs ~2x base input tokens where a
    /// 5-minute write costs ~1.25x, so the optimal configuration is usually
    /// mixed — `1h` on the prefix, the 5-minute default on the tail:
    ///
    /// ```ignore
    /// let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
    ///     .with_automatic_caching()
    ///     .with_static_prefix_cache_ttl(CacheTtl::OneHour);
    /// ```
    ///
    /// Rig places explicit `cache_control` markers on the final tool
    /// definition and the system prompt at this TTL. The conversation tail is
    /// unaffected: it follows the automatic/top-level TTL (Anthropic's moving
    /// breakpoint in automatic mode, Rig's last-message marker in manual
    /// [`with_prompt_caching`] mode). When this knob is unset, the prefix
    /// inherits the top-level TTL exactly as before.
    ///
    /// Anthropic requires 1-hour markers to precede 5-minute ones. The static
    /// prefix precedes the tail, so `OneHour` here composes with a 5-minute
    /// tail — but setting `FiveMinutes` here alongside
    /// [`with_automatic_caching_1h`] is the illegal inversion and fails before
    /// any request is sent. The model-specific minimum cacheable prompt
    /// lengths tabulated on [`with_automatic_caching`] apply to each marker;
    /// below the minimum, Anthropic silently skips caching.
    ///
    /// [`with_prompt_caching`]: CompletionModel::with_prompt_caching
    /// [`with_automatic_caching`]: CompletionModel::with_automatic_caching
    /// [`with_automatic_caching_1h`]: CompletionModel::with_automatic_caching_1h
    pub fn with_static_prefix_cache_ttl(mut self, ttl: CacheTtl) -> Self {
        self.static_prefix_cache_ttl = Some(ttl);
        self
    }
}

impl<T> GenericCompletionModel<super::client::AnthropicExt, T>
where
    T: HttpClientExt,
{
    /// Enable Anthropic strict tool use for every Rig-generated tool.
    ///
    /// Anthropic constrains tool inputs to the supported JSON Schema subset
    /// when `strict: true` is present on a tool definition. Rig sanitizes each
    /// generated tool schema for that subset and leaves provider-specific tools
    /// supplied through `additional_params` unchanged. Unsupported validation
    /// keywords are retained only as model guidance in schema descriptions;
    /// neither Anthropic nor Rig enforces those original constraints, so
    /// validate tool inputs before execution when those constraints matter.
    ///
    /// Anthropic caches compiled schemas for up to 24 hours. Do not include PHI
    /// in schema property names, enum or const values, or regex patterns. See
    /// Anthropic's [structured output retention guidance](https://platform.claude.com/docs/en/build-with-claude/structured-outputs#data-retention).
    /// Anthropic also limits each request to 20 strict tools and applies
    /// additional schema-complexity limits; see [schema complexity limits](https://platform.claude.com/docs/en/build-with-claude/structured-outputs#schema-complexity-limits).
    pub fn with_strict_tools(mut self) -> Self {
        self.strict_tools = true;
        self
    }
}

/// Anthropic requires a `max_tokens` parameter to be set, which is dependent on the model. If not
/// set or if set too high, the request will fail. The following values are based on Anthropic's
/// published synchronous Messages API output limits for current models.
fn default_max_tokens_for_model(model: &str) -> Option<u64> {
    if model.starts_with("claude-opus-4-8")
        || model.starts_with("claude-opus-4-7")
        || model.starts_with("claude-opus-4-6")
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

fn default_max_tokens_with_fallback(model: &str) -> u64 {
    default_max_tokens_for_model(model).unwrap_or(2_048)
}

pub(super) fn supports_mid_conversation_system_messages(model: &str) -> bool {
    model.starts_with(CLAUDE_OPUS_4_8)
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Metadata {
    user_id: Option<String>,
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
    type Error = CompletionError;

    fn try_from(value: message::ToolChoice) -> Result<Self, Self::Error> {
        let res = match value {
            message::ToolChoice::Auto => Self::Auto,
            message::ToolChoice::None => Self::None,
            message::ToolChoice::Required => Self::Any,
            message::ToolChoice::Specific { function_names } => {
                if function_names.len() != 1 {
                    return Err(CompletionError::ProviderError(
                        "Only one tool may be specified to be used by Claude".into(),
                    ));
                }

                let Some(name) = function_names.into_iter().next() else {
                    return Err(CompletionError::ProviderError(
                        "Only one tool may be specified to be used by Claude".into(),
                    ));
                };

                Self::Tool { name }
            }
        };

        Ok(res)
    }
}

/// Recursively ensures all object schemas respect Anthropic structured output restrictions:
/// - `additionalProperties` must be explicitly set to `false` on every object
/// - All properties must be listed in `required`
///
/// Source: <https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs#json-schema-limitations>
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
fn sanitize_strict_tool_schema(schema: &mut serde_json::Value) {
    let mut original = std::mem::take(schema);
    inline_local_root_reference(&mut original);
    flatten_root_all_of(&mut original);
    // Anthropic requires a tool's top-level input schema to declare an object
    // type even when standard JSON Schema would infer it from `properties` or
    // a resolved root reference. Nested schemas do not have this tool-input
    // restriction.
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
            // Anthropic rejects union combinators at the tool-input root. A
            // conjunction between a referenced object and a union cannot be
            // flattened without duplicating the whole base schema, so retain
            // it as guidance instead of producing a guaranteed 400.
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
        // Anthropic rejects allOf at the top level of a tool input. Preserve
        // constraints that cannot be structurally merged as model guidance,
        // consistent with the rest of the strict-schema transformer.
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
) -> Result<(), CompletionError> {
    match ttl {
        CacheControlTtl::OneHour if *shorter_ttl_seen => Err(CompletionError::RequestError(
            "Anthropic cache_control markers with ttl `1h` must appear before markers with \
                 the default 5-minute TTL"
                .into(),
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
) -> Result<(), CompletionError> {
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
) -> Result<(), CompletionError> {
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
        return Err(CompletionError::RequestError(
            "Anthropic manual prompt caching requires a cache_control marker on the final \
             non-deferred tool, but explicit tool markers exhaust the available cache point budget"
                .into(),
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
) -> Result<(), CompletionError> {
    normalize_tool_cache_control(tools);

    let max_cache_markers = if top_level_cache_control.is_some() {
        MAX_CACHE_CONTROL_MARKERS - 1
    } else {
        MAX_CACHE_CONTROL_MARKERS
    };
    let tool_cache_markers = tool_cache_control_count(tools);

    if tool_cache_markers > max_cache_markers {
        return Err(CompletionError::RequestError(
            format!(
                "Too many Anthropic tool `cache_control` markers: {tool_cache_markers} exceeds \
                 the available prompt caching budget of {max_cache_markers}"
            )
            .into(),
        ));
    }

    let mut remaining_cache_markers = max_cache_markers - tool_cache_markers;

    // The static prefix (tools + system) must not carry a shorter TTL than the
    // tail that follows it — Anthropic requires 1h markers before 5-minute
    // ones. Catch the typed-knob inversion here with an error that names the
    // knobs; the generic marker-order validator below would otherwise report
    // it in terms of raw markers.
    let top_level_ttl = top_level_cache_control_ttl(top_level_cache_control);
    if static_prefix_cache_ttl == Some(&CacheTtl::FiveMinutes)
        && top_level_ttl == Some(CacheTtl::OneHour)
    {
        return Err(CompletionError::RequestError(
            "`with_static_prefix_cache_ttl(CacheTtl::FiveMinutes)` conflicts with the 1-hour \
             top-level cache TTL (`with_automatic_caching_1h` or a raw top-level \
             `cache_control`): Anthropic requires 1h markers to precede 5-minute ones, and the \
             static prefix precedes the conversation tail"
                .into(),
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
) -> Result<Option<CacheControl>, CompletionError> {
    if let Some(map) = additional_params.as_object_mut()
        && let Some(raw_cache_control) = map.remove("cache_control")
    {
        if raw_cache_control.is_null() {
            return Ok(None);
        }

        return serde_json::from_value::<CacheControl>(raw_cache_control)
            .map(Some)
            .map_err(|err| {
                CompletionError::RequestError(
                    format!("Invalid Anthropic `additional_params.cache_control` payload: {err}")
                        .into(),
                )
            });
    }

    Ok(None)
}

pub(super) fn resolve_top_level_cache_control(
    automatic_caching: bool,
    automatic_caching_ttl: Option<CacheTtl>,
    additional_params: &mut serde_json::Value,
) -> Result<Option<CacheControl>, CompletionError> {
    let raw_cache_control = extract_top_level_cache_control(additional_params)?;
    let typed_cache_control = automatic_caching.then_some(CacheControl::Ephemeral {
        ttl: automatic_caching_ttl.clone(),
    });

    match (typed_cache_control, raw_cache_control) {
        (Some(typed_cache_control), Some(raw_cache_control)) => {
            if automatic_caching_ttl.is_some()
                && cache_control_ttl(&typed_cache_control) != cache_control_ttl(&raw_cache_control)
            {
                return Err(CompletionError::RequestError(
                    "Anthropic `additional_params.cache_control` conflicts with the typed \
                     automatic caching TTL"
                        .into(),
                ));
            }

            Ok(Some(raw_cache_control))
        }
        (Some(typed_cache_control), None) => Ok(Some(typed_cache_control)),
        (None, raw_cache_control) => Ok(raw_cache_control),
    }
}

pub(super) fn split_system_messages_from_history(
    history: Vec<message::Message>,
    preserve_mid_conversation_system_messages: bool,
) -> (Vec<SystemContent>, Vec<message::Message>) {
    let mut system = Vec::new();
    let mut remaining = Vec::new();

    for (index, message) in history.iter().enumerate() {
        match message {
            message::Message::System { content } => {
                if !content.is_empty() {
                    if preserve_mid_conversation_system_messages
                        && is_valid_mid_conversation_system_message(&history, index)
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
    pub(super) fn try_from_params<Ext>(
        params: AnthropicRequestParams<'_>,
        strict_tools: bool,
    ) -> Result<Self, CompletionError>
    where
        Ext: AnthropicCompatibleProvider,
    {
        let AnthropicRequestParams {
            model,
            request: mut req,
            prompt_caching,
            automatic_caching,
            automatic_caching_ttl,
            static_prefix_cache_ttl,
        } = params;
        let chat_history = req.chat_history_with_documents();

        // Check if max_tokens is set, required for Anthropic
        let Some(max_tokens) = req.max_tokens else {
            return Err(CompletionError::RequestError(
                "`max_tokens` must be set for Anthropic".into(),
            ));
        };

        let (history_system, chat_history) = split_system_messages_from_history(
            chat_history,
            supports_mid_conversation_system_messages(model),
        );
        let mut full_history = vec![];
        full_history.extend(chat_history);

        let mut messages = full_history
            .into_iter()
            .map(Message::try_from)
            .collect::<Result<Vec<Message>, _>>()?;

        let mut additional_params_payload = req
            .additional_params
            .take()
            .unwrap_or(serde_json::Value::Null);
        let top_level_cache_control = resolve_top_level_cache_control(
            automatic_caching,
            automatic_caching_ttl,
            &mut additional_params_payload,
        )?;
        let mut tools =
            build_tool_definitions::<Ext>(req.tools, &mut additional_params_payload, strict_tools)?;

        // Convert system prompt to array format for cache_control support
        let mut system = if let Some(preamble) = req.preamble {
            if preamble.is_empty() {
                vec![]
            } else {
                vec![SystemContent::Text {
                    text: preamble,
                    cache_control: None,
                }]
            }
        } else {
            vec![]
        };
        system.extend(history_system);

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
            // Automatic caching: one top-level field; the API moves the breakpoint automatically.
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
    type Error = CompletionError;

    fn try_from(params: AnthropicRequestParams<'_>) -> Result<Self, Self::Error> {
        Self::try_from_params::<super::client::AnthropicExt>(params, false)
    }
}

pub(super) fn extract_tools_from_additional_params(
    additional_params: &mut serde_json::Value,
) -> Result<Vec<serde_json::Value>, CompletionError> {
    if let Some(map) = additional_params.as_object_mut()
        && let Some(raw_tools) = map.remove("tools")
    {
        return serde_json::from_value::<Vec<serde_json::Value>>(raw_tools).map_err(|err| {
            CompletionError::RequestError(
                format!("Invalid Anthropic `additional_params.tools` payload: {err}").into(),
            )
        });
    }

    Ok(Vec::new())
}

pub(super) fn build_tool_definitions<Ext>(
    tools: Vec<completion::ToolDefinition>,
    additional_params_payload: &mut serde_json::Value,
    strict_tools: bool,
) -> Result<Vec<serde_json::Value>, CompletionError>
where
    Ext: AnthropicCompatibleProvider,
{
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
            if strict_tools {
                Ext::enable_strict_tool_use(&mut tool);
            }

            tool
        })
        .map(serde_json::to_value)
        .collect::<Result<Vec<_>, _>>()?;
    tools.append(&mut additional_tools);

    Ok(tools)
}

impl<Ext, T> GenericCompletionModel<Ext, T>
where
    T: HttpClientExt + Clone + Default + WasmCompatSend + WasmCompatSync + 'static,
    Ext: AnthropicCompatibleProvider + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Execute a completion and return Anthropic's own wire response.
    ///
    /// This is the escape hatch for provider-specific fields rig does not
    /// normalize. It shares the request builder, transport, telemetry, and
    /// error handling with
    /// [`CompletionModel::completion`](completion::CompletionModel::completion),
    /// which calls it and then applies the provider-local mapping — one network
    /// request either way.
    pub async fn raw_completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        let (span, request) =
            self.prepare_request(completion_request, CompletionOperation::Chat)?;

        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "Anthropic completion request",
            &request,
        );

        let request: Vec<u8> = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/v1/messages")?
            .body(request)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        let (mut completion, provider_request_id) =
            send_completion::<_, ApiResponse<CompletionResponse>, _>(
                &self.client,
                req,
                "Anthropic completion",
                Ext::REQUEST_ID_HEADER,
                |completion| {
                    let span = tracing::Span::current();
                    span.record_response_metadata(completion);
                    span.record_token_usage(&crate::completion::Usage::from(&completion.usage));
                },
            )
            .instrument(span)
            .await?;
        completion.provider_request_id = provider_request_id;
        Ok(completion)
    }
}

impl<Ext, T> completion::CompletionModel for GenericCompletionModel<Ext, T>
where
    T: HttpClientExt + Clone + Default + WasmCompatSend + WasmCompatSync + 'static,
    Ext: AnthropicCompatibleProvider + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    // Anthropic's native structured outputs (constrained decoding) are designed
    // to compose with strict tool use, so the schema constraint does not suppress
    // tool calls. See issue #1928.
    fn capabilities(&self) -> completion::ProviderCapabilities {
        completion::ProviderCapabilities::default().with_native_output_tool_composition(true)
    }

    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        // Capture before `normalize` consumes the raw value.
        let response = self.raw_completion(completion_request).await?;
        let captured = serde_json::to_value(&response)?;
        Ok(response.normalize(Ext::PROVIDER_NAME)?.with_raw(captured))
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError> {
        GenericCompletionModel::stream(self, request).await
    }
}

impl<Ext, T> crate::client::ConstructCompletionModel<crate::client::Client<Ext, T>>
    for GenericCompletionModel<Ext, T>
where
    crate::client::Client<Ext, T>: Clone,
    T: HttpClientExt,
    Ext: AnthropicCompatibleProvider + Clone + 'static,
{
    fn construct(client: &crate::client::Client<Ext, T>, model: String) -> Self {
        Self::new(client.clone(), model)
    }
}

use crate::providers::internal::envelope::ApiErrorResponse;

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiResponse<T> {
    Message(T),
    Error(ApiErrorResponse),
}

impl<T> crate::providers::internal::envelope::ProviderEnvelope for ApiResponse<T> {
    type Payload = T;

    fn into_payload(self) -> Result<T, String> {
        match self {
            Self::Message(payload) => Ok(payload),
            Self::Error(ApiErrorResponse { message }) => Err(message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::EMPTY_RESPONSE_ERROR;
    use serde_json::json;
    use serde_path_to_error::deserialize;

    #[test]
    fn current_model_default_max_tokens_match_anthropic_limits() {
        assert_eq!(default_max_tokens_for_model(CLAUDE_OPUS_4_8), Some(128_000));
        assert_eq!(default_max_tokens_for_model(CLAUDE_OPUS_4_7), Some(128_000));
        assert_eq!(default_max_tokens_for_model(CLAUDE_OPUS_4_6), Some(128_000));
        assert_eq!(
            default_max_tokens_for_model(CLAUDE_SONNET_4_6),
            Some(64_000)
        );
        assert_eq!(default_max_tokens_for_model(CLAUDE_HAIKU_4_5), Some(64_000));
    }

    #[test]
    fn unknown_model_uses_conservative_default_max_tokens_fallback() {
        assert_eq!(default_max_tokens_for_model("claude-unknown"), None);
        assert_eq!(default_max_tokens_with_fallback("claude-unknown"), 2_048);
    }

    #[test]
    fn system_role_message_deserializes_and_round_trips() {
        let message: Message = serde_json::from_str(
            r#"
        {
            "role": "system",
            "content": "From now on, require explicit type annotations."
        }
        "#,
        )
        .unwrap();

        assert_eq!(message.role, Role::System);

        let generic: message::Message = message.try_into().unwrap();
        assert_eq!(
            generic,
            message::Message::System {
                content: "From now on, require explicit type annotations.".to_string()
            }
        );

        let provider: Message = generic.try_into().unwrap();
        assert_eq!(provider.role, Role::System);
    }

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
                },
                {
                    "type": "tool_use",
                    "id": "toolu_01A09q90qw90lq917835lq9",
                    "name": "get_weather",
                    "input": {"location": "San Francisco, CA"}
                }
            ]
        }
        "#;

        let user_message_json = r#"
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "/9j/4AAQSkZJRg..."
                    }
                },
                {
                    "type": "text",
                    "text": "What is in this image?"
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                    "content": "15 degrees"
                }
            ]
        }
        "#;

        let assistant_message: Message = {
            let jd = &mut serde_json::Deserializer::from_str(assistant_message_json);
            deserialize(jd).unwrap_or_else(|err| {
                panic!("Deserialization error at {}: {}", err.path(), err);
            })
        };

        let assistant_message2: Message = {
            let jd = &mut serde_json::Deserializer::from_str(assistant_message_json2);
            deserialize(jd).unwrap_or_else(|err| {
                panic!("Deserialization error at {}: {}", err.path(), err);
            })
        };

        let user_message: Message = {
            let jd = &mut serde_json::Deserializer::from_str(user_message_json);
            deserialize(jd).unwrap_or_else(|err| {
                panic!("Deserialization error at {}: {}", err.path(), err);
            })
        };

        let Message { role, content } = assistant_message;
        assert_eq!(role, Role::Assistant);
        assert_eq!(
            content.first(),
            Some(&Content::Text {
                text: "\n\nHello there, how may I assist you today?".to_owned(),
                citations: Vec::new(),
                cache_control: None,
            })
        );

        let Message { role, content } = assistant_message2;
        {
            assert_eq!(role, Role::Assistant);
            assert_eq!(content.len(), 2);

            let mut iter = content.into_iter();

            match iter.next().unwrap() {
                Content::Text { text, .. } => {
                    assert_eq!(text, "\n\nHello there, how may I assist you today?");
                }
                _ => panic!("Expected text content"),
            }

            match iter.next().unwrap() {
                Content::ToolUse { id, name, input } => {
                    assert_eq!(id, "toolu_01A09q90qw90lq917835lq9");
                    assert_eq!(name, "get_weather");
                    assert_eq!(input, json!({"location": "San Francisco, CA"}));
                }
                _ => panic!("Expected tool use content"),
            }

            assert_eq!(iter.next(), None);
        }

        let Message { role, content } = user_message;
        {
            assert_eq!(role, Role::User);
            assert_eq!(content.len(), 3);

            let mut iter = content.into_iter();

            match iter.next().unwrap() {
                Content::Image { source, .. } => {
                    assert_eq!(
                        source,
                        ImageSource::Base64 {
                            data: "/9j/4AAQSkZJRg...".to_owned(),
                            media_type: ImageFormat::JPEG,
                        }
                    );
                }
                _ => panic!("Expected image content"),
            }

            match iter.next().unwrap() {
                Content::Text { text, .. } => {
                    assert_eq!(text, "What is in this image?");
                }
                _ => panic!("Expected text content"),
            }

            match iter.next().unwrap() {
                Content::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                    ..
                } => {
                    assert_eq!(tool_use_id, "toolu_01A09q90qw90lq917835lq9");
                    assert_eq!(
                        content.first(),
                        Some(&ToolResultContent::Text {
                            text: "15 degrees".to_owned()
                        })
                    );
                    assert_eq!(is_error, None);
                }
                _ => panic!("Expected tool result content"),
            }

            assert_eq!(iter.next(), None);
        }
    }

    #[test]
    fn test_message_to_message_conversion() {
        let user_message: Message = serde_json::from_str(
            r#"
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "/9j/4AAQSkZJRg..."
                    }
                },
                {
                    "type": "text",
                    "text": "What is in this image?"
                },
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "data": "base64_encoded_pdf_data",
                        "media_type": "application/pdf"
                    }
                }
            ]
        }
        "#,
        )
        .unwrap();

        let assistant_message = Message {
            role: Role::Assistant,
            content: vec![Content::ToolUse {
                id: "toolu_01A09q90qw90lq917835lq9".to_string(),
                name: "get_weather".to_string(),
                input: json!({"location": "San Francisco, CA"}),
            }],
        };

        let tool_message = Message {
            role: Role::User,
            content: vec![Content::ToolResult {
                tool_use_id: "toolu_01A09q90qw90lq917835lq9".to_string(),
                content: vec![ToolResultContent::Text {
                    text: "15 degrees".to_string(),
                }],
                is_error: None,
                cache_control: None,
            }],
        };

        let converted_user_message: message::Message = user_message.clone().try_into().unwrap();
        let converted_assistant_message: message::Message =
            assistant_message.clone().try_into().unwrap();
        let converted_tool_message: message::Message = tool_message.clone().try_into().unwrap();

        match converted_user_message.clone() {
            message::Message::User { content } => {
                assert_eq!(content.len(), 3);

                let mut iter = content.into_iter();

                match iter.next().unwrap() {
                    message::UserContent::Image(message::Image {
                        data, media_type, ..
                    }) => {
                        assert_eq!(data, DocumentSourceKind::base64("/9j/4AAQSkZJRg..."));
                        assert_eq!(media_type, Some(message::ImageMediaType::JPEG));
                    }
                    _ => panic!("Expected image content"),
                }

                match iter.next().unwrap() {
                    message::UserContent::Text(message::Text { text, .. }) => {
                        assert_eq!(text, "What is in this image?");
                    }
                    _ => panic!("Expected text content"),
                }

                match iter.next().unwrap() {
                    message::UserContent::Document(message::Document {
                        data, media_type, ..
                    }) => {
                        assert_eq!(
                            data,
                            DocumentSourceKind::String("base64_encoded_pdf_data".into())
                        );
                        assert_eq!(media_type, Some(message::DocumentMediaType::PDF));
                    }
                    _ => panic!("Expected document content"),
                }

                assert_eq!(iter.next(), None);
            }
            _ => panic!("Expected user message"),
        }

        match converted_tool_message.clone() {
            message::Message::User { content } => {
                let message::ToolResult {
                    call,
                    name,
                    content,
                    ..
                } = match content.first() {
                    Some(message::UserContent::ToolResult(tool_result)) => tool_result,
                    _ => panic!("Expected tool result content"),
                };
                assert_eq!(call, "toolu_01A09q90qw90lq917835lq9");
                // The Anthropic wire carries no tool name on `tool_result`
                // blocks, so the inbound conversion is lossy by design.
                assert_eq!(name, "");
                match content.first() {
                    Some(message::ToolResultContent::Text(message::Text { text, .. })) => {
                        assert_eq!(text, "15 degrees");
                    }
                    _ => panic!("Expected text content"),
                }
            }
            _ => panic!("Expected tool result content"),
        }

        match converted_assistant_message.clone() {
            message::Message::Assistant { content, .. } => {
                assert_eq!(content.len(), 1);

                match content.first() {
                    Some(message::AssistantContent::ToolCall(message::ToolCall {
                        id,
                        function,
                        ..
                    })) => {
                        assert_eq!(id, "toolu_01A09q90qw90lq917835lq9");
                        assert_eq!(function.name, "get_weather");
                        assert_eq!(function.arguments, json!({"location": "San Francisco, CA"}));
                    }
                    _ => panic!("Expected tool call content"),
                }
            }
            _ => panic!("Expected assistant message"),
        }

        let original_user_message: Message = converted_user_message.try_into().unwrap();
        let original_assistant_message: Message = converted_assistant_message.try_into().unwrap();
        let original_tool_message: Message = converted_tool_message.try_into().unwrap();

        assert_eq!(user_message, original_user_message);
        assert_eq!(assistant_message, original_assistant_message);
        assert_eq!(tool_message, original_tool_message);
    }

    #[test]
    fn test_content_format_conversion() {
        use crate::completion::message::ContentFormat;

        let source_type: SourceType = ContentFormat::Url.try_into().unwrap();
        assert_eq!(source_type, SourceType::URL);

        let content_format: ContentFormat = SourceType::URL.into();
        assert_eq!(content_format, ContentFormat::Url);

        let source_type: SourceType = ContentFormat::Base64.try_into().unwrap();
        assert_eq!(source_type, SourceType::BASE64);

        let content_format: ContentFormat = SourceType::BASE64.into();
        assert_eq!(content_format, ContentFormat::Base64);

        let source_type: SourceType = ContentFormat::String.try_into().unwrap();
        assert_eq!(source_type, SourceType::TEXT);

        let content_format: ContentFormat = SourceType::TEXT.into();
        assert_eq!(content_format, ContentFormat::String);
    }

    #[test]
    fn test_cache_control_serialization() {
        // Test SystemContent with cache_control
        let system = SystemContent::Text {
            text: "You are a helpful assistant.".to_string(),
            cache_control: Some(CacheControl::ephemeral()),
        };
        let json = serde_json::to_string(&system).unwrap();
        assert!(json.contains(r#""cache_control":{"type":"ephemeral"}"#));
        assert!(json.contains(r#""type":"text""#));

        // Test SystemContent without cache_control (should not have cache_control field)
        let system_no_cache = SystemContent::Text {
            text: "Hello".to_string(),
            cache_control: None,
        };
        let json_no_cache = serde_json::to_string(&system_no_cache).unwrap();
        assert!(!json_no_cache.contains("cache_control"));

        // Test Content::Text with cache_control
        let content = Content::Text {
            text: "Test message".to_string(),
            citations: Vec::new(),
            cache_control: Some(CacheControl::ephemeral()),
        };
        let json_content = serde_json::to_string(&content).unwrap();
        assert!(json_content.contains(r#""cache_control":{"type":"ephemeral"}"#));

        // Manual prompt caching over a bare system prompt + conversation: the
        // system block and the tail of the last message get the marker.
        let mut system_vec = vec![SystemContent::Text {
            text: "System prompt".to_string(),
            cache_control: None,
        }];
        let mut messages = vec![
            Message {
                role: Role::User,
                content: vec![Content::Text {
                    text: "First message".to_string(),
                    citations: Vec::new(),
                    cache_control: None,
                }],
            },
            Message {
                role: Role::Assistant,
                content: vec![Content::Text {
                    text: "Response".to_string(),
                    citations: Vec::new(),
                    cache_control: None,
                }],
            },
        ];

        apply_prompt_cache_control(&mut system_vec, &mut messages, &mut [], true, None, None)
            .unwrap();

        // System should have cache_control
        match &system_vec[0] {
            SystemContent::Text { cache_control, .. } => {
                assert!(cache_control.is_some());
            }
        }

        // Only the last content block of last message should have cache_control
        // First message should NOT have cache_control
        for content in messages[0].content.iter() {
            if let Content::Text { cache_control, .. } = content {
                assert!(cache_control.is_none());
            }
        }

        // Last message SHOULD have cache_control
        for content in messages[1].content.iter() {
            if let Content::Text { cache_control, .. } = content {
                assert!(cache_control.is_some());
            }
        }
    }

    fn generic_tool(name: &str) -> completion::ToolDefinition {
        completion::ToolDefinition {
            name: name.to_string(),
            description: format!("{name} description"),
            parameters: json!({
                "type": "object",
                "properties": {}
            }),
        }
    }

    fn completion_request_with_tools(
        tools: Vec<completion::ToolDefinition>,
        additional_params: Option<serde_json::Value>,
    ) -> CompletionRequest {
        CompletionRequest {
            model: None,
            preamble: Some("System prompt".to_string()),
            chat_history: vec![message::Message::from("Hello")],
            documents: Vec::new(),
            tools,
            temperature: None,
            max_tokens: Some(64),
            tool_choice: None,
            additional_params,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    fn completion_request_with_history(
        chat_history: Vec<message::Message>,
        preamble: Option<String>,
    ) -> CompletionRequest {
        CompletionRequest {
            model: None,
            preamble,
            chat_history,
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: Some(64),
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    #[test]
    fn rig_tools_are_non_strict_by_default() {
        let request = completion_request_with_tools(vec![generic_tool("lookup")], None);
        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: CLAUDE_SONNET_4_6,
            request,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        assert!(value["tools"][0].get("strict").is_none());
        assert!(
            value["tools"][0]["input_schema"]
                .get("additionalProperties")
                .is_none()
        );
    }

    #[test]
    fn strict_tool_hook_is_a_noop_for_anthropic_compatible_gateways() {
        let mut additional_params = serde_json::Value::Null;
        let tools = build_tool_definitions::<crate::providers::minimax::MiniMaxAnthropicExt>(
            vec![generic_tool("lookup")],
            &mut additional_params,
            true,
        )
        .unwrap();

        assert!(tools[0].get("strict").is_none());
        assert!(
            tools[0]["input_schema"]
                .get("additionalProperties")
                .is_none()
        );
    }

    #[test]
    fn strict_tools_opt_in_marks_and_sanitizes_rig_tools_only() {
        let mut tool = generic_tool("lookup");
        tool.parameters = json!({
            "type": "object",
            "additionalProperties": true,
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 2,
                    "maxLength": 20,
                    "pattern": "^[a-z]+$",
                    "format": "uuid"
                },
                "kind": {
                    "type": "string",
                    "const": "lookup"
                },
                "legacy_filter": {
                    "$ref": "#/definitions/LegacyFilter"
                },
                "options": {
                    "type": "object",
                    "additionalProperties": true,
                    "properties": {
                        "limit": {
                            "type": ["integer", "null"],
                            "minimum": 1,
                            "maximum": 100,
                            "format": "uint32"
                        }
                    }
                }
            },
            "definitions": {
                "LegacyFilter": {
                    "type": "object",
                    "properties": {
                        "term": { "type": "string" }
                    }
                }
            },
            "required": ["query"]
        });
        let request = completion_request_with_tools(
            vec![tool],
            Some(json!({
                "tools": [{
                    "type": "mcp_toolset",
                    "name": "remote_tools"
                }]
            })),
        );
        let request = AnthropicCompletionRequest::try_from_params::<
            crate::providers::anthropic::client::AnthropicExt,
        >(
            AnthropicRequestParams {
                model: CLAUDE_SONNET_4_6,
                request,
                prompt_caching: false,
                automatic_caching: false,
                automatic_caching_ttl: None,
                static_prefix_cache_ttl: None,
            },
            true,
        )
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let rig_tool = &value["tools"][0];
        assert_eq!(rig_tool["strict"], true);
        assert_eq!(rig_tool["input_schema"]["additionalProperties"], false);
        let required = rig_tool["input_schema"]["required"]
            .as_array()
            .expect("strict object schema should list required properties");
        assert_eq!(required.len(), 1);
        assert!(required.contains(&json!("query")));
        assert_eq!(
            rig_tool["input_schema"]["properties"]["options"]["additionalProperties"],
            false
        );
        assert!(
            rig_tool["input_schema"]["properties"]["options"]
                .get("required")
                .is_none()
        );
        let query = &rig_tool["input_schema"]["properties"]["query"];
        assert_eq!(query["format"], "uuid");
        for keyword in ["minLength", "maxLength", "pattern"] {
            assert!(query.get(keyword).is_none());
        }
        let query_description = query["description"]
            .as_str()
            .expect("unsupported string constraints should become guidance");
        for guidance in ["minLength: 2", "maxLength: 20", "pattern: ^[a-z]+$"] {
            assert!(query_description.contains(guidance));
        }
        assert_eq!(
            rig_tool["input_schema"]["properties"]["kind"]["const"],
            "lookup"
        );
        assert_eq!(
            rig_tool["input_schema"]["properties"]["legacy_filter"]["$ref"],
            "#/definitions/LegacyFilter"
        );
        assert_eq!(
            rig_tool["input_schema"]["definitions"]["LegacyFilter"]["additionalProperties"],
            false
        );
        let limit = &rig_tool["input_schema"]["properties"]["options"]["properties"]["limit"];
        assert!(limit.get("format").is_none());
        assert!(
            ["minimum", "maximum"]
                .into_iter()
                .all(|keyword| limit.get(keyword).is_none())
        );
        let limit_description = limit["description"]
            .as_str()
            .expect("unsupported numeric constraints should become guidance");
        for guidance in ["minimum: 1", "maximum: 100", "format: uint32"] {
            assert!(limit_description.contains(guidance));
        }

        let provider_tool = &value["tools"][1];
        assert_eq!(provider_tool["type"], "mcp_toolset");
        assert!(provider_tool.get("strict").is_none());
    }

    fn system_has_cache_control(value: &serde_json::Value) -> bool {
        value["system"]
            .as_array()
            .and_then(|blocks| blocks.last())
            .and_then(|block| block.get("cache_control"))
            .is_some()
    }

    fn last_message_has_cache_control(value: &serde_json::Value) -> bool {
        value["messages"]
            .as_array()
            .and_then(|messages| messages.last())
            .and_then(|message| message["content"].as_array())
            .and_then(|content| content.last())
            .and_then(|content| content.get("cache_control"))
            .is_some()
    }

    #[test]
    fn opus_4_8_preserves_mid_conversation_system_message() {
        let request = completion_request_with_history(
            vec![
                message::Message::System {
                    content: "Global history instruction.".to_string(),
                },
                message::Message::from("Review this code."),
                message::Message::System {
                    content: "From now on, require explicit type annotations.".to_string(),
                },
            ],
            Some("Top-level instruction.".to_string()),
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: CLAUDE_OPUS_4_8,
            request,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        assert_eq!(value["system"][0]["text"], "Top-level instruction.");
        assert_eq!(value["system"][1]["text"], "Global history instruction.");

        let messages = value["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "system");
        assert_eq!(
            messages[1]["content"][0]["text"],
            "From now on, require explicit type annotations."
        );
    }

    #[test]
    fn opus_4_8_preserves_mid_conversation_system_message_before_assistant_turn() {
        let request = completion_request_with_history(
            vec![
                message::Message::user("Review this code."),
                message::Message::System {
                    content: "From now on, require explicit type annotations.".to_string(),
                },
                message::Message::assistant("I will enforce explicit type annotations."),
            ],
            None,
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: CLAUDE_OPUS_4_8,
            request,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let messages = value["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "system");
        assert_eq!(messages[2]["role"], "assistant");
        assert!(value.get("system").is_none());
    }

    #[test]
    fn opus_4_8_hoists_leading_system_message_when_documents_are_present() {
        let mut request = completion_request_with_history(
            vec![
                message::Message::System {
                    content: "Global history instruction.".to_string(),
                },
                message::Message::assistant("Acknowledged."),
                message::Message::System {
                    content: "Mid-conversation instruction.".to_string(),
                },
                message::Message::user("Answer from the document."),
            ],
            None,
        );
        request.documents = vec![completion::Document {
            id: "doc".to_string(),
            text: "Document context.".to_string(),
            additional_props: Default::default(),
        }];

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: CLAUDE_OPUS_4_8,
            request,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        assert_eq!(value["system"][0]["text"], "Global history instruction.");
        assert_eq!(value["system"][1]["text"], "Mid-conversation instruction.");

        let messages = value["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[2]["role"], "user");
        assert!(
            messages[0].to_string().contains("<file id: doc>"),
            "document message should follow top-level system: {messages:?}"
        );
        assert_eq!(
            messages
                .iter()
                .filter(|message| message.to_string().contains("<file id: doc>"))
                .count(),
            1,
            "document message should appear exactly once: {messages:?}"
        );
        assert!(
            messages
                .iter()
                .all(|message| message["role"].as_str() != Some("system"))
        );
    }

    #[test]
    fn opus_4_8_preserves_system_message_after_assistant_server_tool_result() {
        let request = completion_request_with_history(
            vec![
                message::Message::Assistant {
                    id: None,
                    content: vec![
                        message::AssistantContent::Text(message::Text {
                            text: String::new(),
                            additional_params: crate::message::AdditionalParams::try_from_value(
                                json!({
                                    ANTHROPIC_RAW_CONTENT_KEY: {
                                        "type": "server_tool_use",
                                        "id": "srvtoolu_01",
                                        "name": "web_search",
                                        "input": {
                                            "query": "clear daytime sky color"
                                        }
                                    }
                                }),
                            )
                            .expect("object params"),
                        }),
                        message::AssistantContent::Text(message::Text {
                            text: String::new(),
                            additional_params: crate::message::AdditionalParams::try_from_value(
                                json!({
                                    ANTHROPIC_RAW_CONTENT_KEY: {
                                        "type": "web_search_tool_result",
                                        "tool_use_id": "srvtoolu_01",
                                        "content": {
                                            "type": "web_search_tool_result_error",
                                            "error_code": "unavailable"
                                        }
                                    }
                                }),
                            )
                            .expect("object params"),
                        }),
                    ],
                },
                message::Message::System {
                    content: "For the rest of this conversation, answer in Spanish.".to_string(),
                },
                message::Message::assistant("Entendido."),
            ],
            None,
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: CLAUDE_OPUS_4_8,
            request,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        assert!(value.get("system").is_none());

        let messages = value["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["content"][0]["type"], "server_tool_use");
        assert_eq!(messages[0]["content"][1]["type"], "web_search_tool_result");
        assert_eq!(messages[1]["role"], "system");
        assert_eq!(
            messages[1]["content"][0]["text"],
            "For the rest of this conversation, answer in Spanish."
        );
        assert_eq!(messages[2]["role"], "assistant");
    }

    #[test]
    fn foreign_annotated_empty_text_produces_no_anthropic_block() {
        // The Responses ingest mints empty text blocks whose params carry
        // that wire's extras; the agent deliberately keeps them in history.
        // Replayed here, they must vanish from the request — the API
        // rejects empty text blocks and foreign extras cannot reach this
        // wire — while sibling content converts unaffected.
        let foreign_annotated_empty = message::AssistantContent::Text(message::Text {
            text: String::new(),
            additional_params: message::AdditionalParams::try_from_value(json!({
                "openai_responses": {"annotations": [{"type": "url_citation"}]}
            }))
            .expect("object params"),
        });
        assert_eq!(
            anthropic_content_from_assistant_content(foreign_annotated_empty.clone())
                .expect("conversion succeeds"),
            Vec::new(),
            "a foreign-annotated empty block must produce no Anthropic content"
        );

        let message = message::Message::Assistant {
            id: None,
            content: vec![
                foreign_annotated_empty,
                message::AssistantContent::text("real answer"),
            ],
        };
        let converted = Message::try_from(message).expect("message converts");
        assert_eq!(converted.content.len(), 1, "only the real block survives");
        assert!(matches!(
            converted.content.first(),
            Some(Content::Text { text, .. }) if text == "real answer"
        ));
    }

    #[test]
    fn opus_4_8_preserves_system_message_after_assistant_server_tool_use() {
        let request = completion_request_with_history(
            vec![
                message::Message::Assistant {
                    id: None,
                    content: vec![message::AssistantContent::Text(message::Text {
                        text: String::new(),
                        additional_params: crate::message::AdditionalParams::try_from_value(
                            json!({
                                ANTHROPIC_RAW_CONTENT_KEY: {
                                    "type": "server_tool_use",
                                    "id": "srvtoolu_01",
                                    "name": "web_search",
                                    "input": {
                                        "query": "clear daytime sky color"
                                    }
                                }
                            }),
                        )
                        .expect("object params"),
                    })],
                },
                message::Message::System {
                    content: "For the rest of this conversation, answer in Spanish.".to_string(),
                },
                message::Message::assistant("Entendido."),
            ],
            None,
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: CLAUDE_OPUS_4_8,
            request,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        assert!(value.get("system").is_none());

        let messages = value["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["content"][0]["type"], "server_tool_use");
        assert_eq!(messages[1]["role"], "system");
        assert_eq!(
            messages[1]["content"][0]["text"],
            "For the rest of this conversation, answer in Spanish."
        );
        assert_eq!(messages[2]["role"], "assistant");
    }

    #[test]
    fn opus_4_8_hoists_system_message_in_invalid_mid_conversation_position() {
        let request = completion_request_with_history(
            vec![
                message::Message::user("Review this code."),
                message::Message::System {
                    content: "From now on, require explicit type annotations.".to_string(),
                },
                message::Message::user("Now review this other file."),
            ],
            None,
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: CLAUDE_OPUS_4_8,
            request,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        assert_eq!(
            value["system"][0]["text"],
            "From now on, require explicit type annotations."
        );

        let messages = value["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "user");
    }

    #[test]
    fn older_anthropic_models_hoist_mid_conversation_system_message() {
        let request = completion_request_with_history(
            vec![
                message::Message::from("Review this code."),
                message::Message::System {
                    content: "From now on, require explicit type annotations.".to_string(),
                },
            ],
            None,
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: CLAUDE_OPUS_4_7,
            request,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        assert_eq!(
            value["system"][0]["text"],
            "From now on, require explicit type annotations."
        );

        let messages = value["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
    }

    #[test]
    fn test_tool_definition_cache_control_serialization() {
        let tool = ToolDefinition {
            name: "cached_tool".to_string(),
            description: Some("Cached tool".to_string()),
            input_schema: json!({"type": "object"}),
            strict: false,
            cache_control: Some(CacheControl::ephemeral()),
        };

        let value = serde_json::to_value(tool).unwrap();
        assert_eq!(value["cache_control"]["type"], "ephemeral");

        let tool_without_cache = ToolDefinition {
            name: "uncached_tool".to_string(),
            description: Some("Uncached tool".to_string()),
            input_schema: json!({"type": "object"}),
            strict: false,
            cache_control: None,
        };

        let value = serde_json::to_value(tool_without_cache).unwrap();
        assert!(value.get("cache_control").is_none());
    }

    #[test]
    fn test_apply_tool_cache_control_marks_only_final_tool() {
        let mut tools = vec![
            json!({
                "name": "first_tool",
                "description": "First tool",
                "input_schema": {"type": "object"}
            }),
            json!({
                "name": "second_tool",
                "description": "Second tool",
                "input_schema": {"type": "object"}
            }),
        ];

        let mut remaining_cache_markers = 4;
        apply_tool_cache_control(
            &mut tools,
            &mut remaining_cache_markers,
            &CacheControl::ephemeral(),
        )
        .unwrap();

        assert!(tools[0].get("cache_control").is_none());
        assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
        assert_eq!(remaining_cache_markers, 3);
    }

    #[test]
    fn test_prompt_caching_skips_final_deferred_tool_in_request() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "tools": [
                    {
                        "name": "regular_tool",
                        "description": "Regular tool",
                        "input_schema": {"type": "object"}
                    },
                    {
                        "name": "deferred_tool",
                        "description": "Deferred tool",
                        "input_schema": {"type": "object"},
                        "defer_loading": true
                    }
                ]
            })),
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert_eq!(tools[0]["name"], "regular_tool");
        assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(tools[1]["name"], "deferred_tool");
        assert!(tools[1].get("cache_control").is_none());
    }

    #[test]
    fn test_prompt_caching_preserves_existing_final_tool_cache_control() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "tools": [{
                    "name": "cached_tool",
                    "description": "Cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}
                }]
            })),
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    }

    #[test]
    fn test_prompt_caching_all_deferred_tools_do_not_receive_cache_control() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "tools": [
                    {
                        "name": "first_deferred_tool",
                        "description": "First deferred tool",
                        "input_schema": {"type": "object"},
                        "defer_loading": true
                    },
                    {
                        "name": "second_deferred_tool",
                        "description": "Second deferred tool",
                        "input_schema": {"type": "object"},
                        "defer_loading": true
                    }
                ]
            })),
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert!(tools[0].get("cache_control").is_none());
        assert!(tools[1].get("cache_control").is_none());
    }

    #[test]
    fn test_prompt_caching_preserves_earlier_tool_cache_control() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "tools": [
                    {
                        "name": "earlier_tool",
                        "description": "Earlier tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral", "ttl": "1h"}
                    },
                    {
                        "name": "later_tool",
                        "description": "Later tool",
                        "input_schema": {"type": "object"}
                    }
                ]
            })),
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
        assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_prompt_caching_deferred_marker_does_not_suppress_loaded_tool_marker() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "tools": [
                    {
                        "name": "regular_tool",
                        "description": "Regular tool",
                        "input_schema": {"type": "object"}
                    },
                    {
                        "name": "deferred_cached_tool",
                        "description": "Deferred cached tool",
                        "input_schema": {"type": "object"},
                        "defer_loading": true,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            })),
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_prompt_caching_errors_when_tool_cache_control_ttl_order_is_invalid() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "tools": [
                    {
                        "name": "first_cached_tool",
                        "description": "First cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "second_cached_tool",
                        "description": "Second cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral", "ttl": "1h"}
                    }
                ]
            })),
        );

        let err = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap_err();

        assert!(err.to_string().contains("ttl `1h`"));
    }

    #[test]
    fn test_prompt_caching_preserves_valid_mixed_ttl_tool_cache_controls() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "tools": [
                    {
                        "name": "first_cached_tool",
                        "description": "First cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral", "ttl": "1h"}
                    },
                    {
                        "name": "second_cached_tool",
                        "description": "Second cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            })),
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
        assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
        assert!(tools[1]["cache_control"].get("ttl").is_none());
    }

    #[test]
    fn test_prompt_caching_preserves_deferred_tool_cache_control() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "tools": [{
                    "name": "deferred_cached_tool",
                    "description": "Deferred cached tool",
                    "input_schema": {"type": "object"},
                    "defer_loading": true,
                    "cache_control": {"type": "ephemeral"}
                }]
            })),
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_prompt_caching_budget_preserves_three_tool_markers_and_skips_message() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "tools": [
                    {
                        "name": "first_cached_tool",
                        "description": "First cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "second_cached_tool",
                        "description": "Second cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "third_cached_tool",
                        "description": "Third cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            })),
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
        assert_eq!(tools[2]["cache_control"]["type"], "ephemeral");
        assert!(system_has_cache_control(&value));
        assert!(!last_message_has_cache_control(&value));
    }

    #[test]
    fn test_prompt_caching_errors_when_explicit_tool_markers_exceed_budget() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "tools": [
                    {
                        "name": "first_cached_tool",
                        "description": "First cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "second_cached_tool",
                        "description": "Second cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "third_cached_tool",
                        "description": "Third cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "fourth_cached_tool",
                        "description": "Fourth cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "fifth_cached_tool",
                        "description": "Fifth cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            })),
        );

        let err = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap_err();

        assert!(err.to_string().contains("Too many Anthropic tool"));
    }

    #[test]
    fn test_prompt_caching_errors_when_final_tool_marker_has_no_budget() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "tools": [
                    {
                        "name": "first_cached_tool",
                        "description": "First cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "second_cached_tool",
                        "description": "Second cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "third_cached_tool",
                        "description": "Third cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "fourth_cached_tool",
                        "description": "Fourth cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "final_uncached_tool",
                        "description": "Final uncached tool",
                        "input_schema": {"type": "object"}
                    }
                ]
            })),
        );

        let err = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap_err();

        assert!(err.to_string().contains("final non-deferred tool"));
    }

    #[test]
    fn test_prompt_caching_replaces_null_final_tool_cache_control() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "tools": [{
                    "name": "final_tool",
                    "description": "Final tool",
                    "input_schema": {"type": "object"},
                    "cache_control": null
                }]
            })),
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_prompt_caching_ignores_null_tool_cache_control_when_budgeting() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "tools": [
                    {
                        "name": "first_null_tool",
                        "description": "First null tool",
                        "input_schema": {"type": "object"},
                        "cache_control": null
                    },
                    {
                        "name": "second_null_tool",
                        "description": "Second null tool",
                        "input_schema": {"type": "object"},
                        "cache_control": null
                    },
                    {
                        "name": "third_null_tool",
                        "description": "Third null tool",
                        "input_schema": {"type": "object"},
                        "cache_control": null
                    },
                    {
                        "name": "fourth_null_tool",
                        "description": "Fourth null tool",
                        "input_schema": {"type": "object"},
                        "cache_control": null
                    },
                    {
                        "name": "final_uncached_tool",
                        "description": "Final uncached tool",
                        "input_schema": {"type": "object"}
                    }
                ]
            })),
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert!(tools[0].get("cache_control").is_none());
        assert!(tools[1].get("cache_control").is_none());
        assert!(tools[2].get("cache_control").is_none());
        assert!(tools[3].get("cache_control").is_none());
        assert_eq!(tools[4]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_prompt_caching_preserves_non_null_provider_tool_cache_control_escape_hatch() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "tools": [{
                    "name": "provider_tool",
                    "description": "Provider tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "provider_specific"}
                }]
            })),
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert_eq!(tools[0]["cache_control"]["type"], "provider_specific");
    }

    #[test]
    fn test_prompt_caching_automatic_mode_uses_reduced_marker_budget() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "tools": [
                    {
                        "name": "first_cached_tool",
                        "description": "First cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "second_cached_tool",
                        "description": "Second cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "third_cached_tool",
                        "description": "Third cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            })),
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: true,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
        assert_eq!(tools[2]["cache_control"]["type"], "ephemeral");
        assert_eq!(value["cache_control"]["type"], "ephemeral");
        assert!(!system_has_cache_control(&value));
        assert!(!last_message_has_cache_control(&value));
    }

    #[test]
    fn test_prompt_caching_automatic_mode_errors_when_final_tool_marker_has_no_budget() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "tools": [
                    {
                        "name": "first_cached_tool",
                        "description": "First cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "second_cached_tool",
                        "description": "Second cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "third_cached_tool",
                        "description": "Third cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "final_uncached_tool",
                        "description": "Final uncached tool",
                        "input_schema": {"type": "object"}
                    }
                ]
            })),
        );

        let err = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: true,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap_err();

        assert!(err.to_string().contains("final non-deferred tool"));
    }

    #[test]
    fn test_automatic_caching_errors_when_explicit_tool_markers_exhaust_budget() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "tools": [
                    {
                        "name": "first_cached_tool",
                        "description": "First cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "second_cached_tool",
                        "description": "Second cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "third_cached_tool",
                        "description": "Third cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "fourth_cached_tool",
                        "description": "Fourth cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            })),
        );

        let err = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: false,
            automatic_caching: true,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap_err();

        assert!(err.to_string().contains("Too many Anthropic tool"));
    }

    #[test]
    fn test_automatic_caching_1h_errors_with_explicit_five_minute_tool_marker() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "tools": [{
                    "name": "cached_tool",
                    "description": "Cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                }]
            })),
        );

        let err = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: false,
            automatic_caching: true,
            automatic_caching_ttl: Some(CacheTtl::OneHour),
            static_prefix_cache_ttl: None,
        })
        .unwrap_err();

        assert!(err.to_string().contains("ttl `1h`"));
    }

    #[test]
    fn test_prompt_and_automatic_caching_1h_uses_1h_generated_markers() {
        let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: true,
            automatic_caching_ttl: Some(CacheTtl::OneHour),
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
        assert_eq!(
            value["system"]
                .as_array()
                .and_then(|blocks| blocks.last())
                .and_then(|block| block["cache_control"].get("ttl")),
            Some(&json!("1h"))
        );
        assert_eq!(value["cache_control"]["ttl"], "1h");
        assert!(!last_message_has_cache_control(&value));
    }

    #[test]
    fn test_prompt_and_raw_top_level_automatic_caching_1h_uses_1h_generated_markers() {
        let request = completion_request_with_tools(
            vec![generic_tool("cached_tool")],
            Some(json!({
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
                "metadata": {"source": "test"}
            })),
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: true,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
        assert_eq!(
            value["system"]
                .as_array()
                .and_then(|blocks| blocks.last())
                .and_then(|block| block["cache_control"].get("ttl")),
            Some(&json!("1h"))
        );
        assert_eq!(value["cache_control"]["ttl"], "1h");
        assert_eq!(value["metadata"]["source"], "test");
        assert!(!last_message_has_cache_control(&value));
    }

    #[test]
    fn test_prompt_caching_uses_raw_top_level_cache_control_ttl() {
        let request = completion_request_with_tools(
            vec![generic_tool("cached_tool")],
            Some(json!({
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
                "metadata": {"source": "raw-cache-control"}
            })),
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
        assert_eq!(
            value["system"]
                .as_array()
                .and_then(|blocks| blocks.last())
                .and_then(|block| block["cache_control"].get("ttl")),
            Some(&json!("1h"))
        );
        assert_eq!(value["cache_control"]["ttl"], "1h");
        assert_eq!(value["metadata"]["source"], "raw-cache-control");
        assert!(!last_message_has_cache_control(&value));
    }

    #[test]
    fn test_static_prefix_ttl_with_manual_caching_splits_prefix_and_tail() {
        let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: Some(CacheTtl::OneHour),
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
        assert_eq!(
            value["system"]
                .as_array()
                .and_then(|blocks| blocks.last())
                .and_then(|block| block["cache_control"].get("ttl")),
            Some(&json!("1h"))
        );
        // The tail keeps the 5-minute default: a marker with no `ttl` field.
        let tail_cache_control = value["messages"]
            .as_array()
            .and_then(|messages| messages.last())
            .and_then(|message| message["content"].as_array())
            .and_then(|content| content.last())
            .map(|block| &block["cache_control"])
            .unwrap();
        assert_eq!(tail_cache_control["type"], "ephemeral");
        assert!(tail_cache_control.get("ttl").is_none());
        assert!(value.get("cache_control").is_none());
    }

    #[test]
    fn test_static_prefix_ttl_with_automatic_caching_marks_prefix_only() {
        let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: false,
            automatic_caching: true,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: Some(CacheTtl::OneHour),
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
        assert_eq!(
            value["system"]
                .as_array()
                .and_then(|blocks| blocks.last())
                .and_then(|block| block["cache_control"].get("ttl")),
            Some(&json!("1h"))
        );
        // The moving tail breakpoint is Anthropic's top-level one at the
        // 5-minute default; no explicit message marker exists.
        assert_eq!(value["cache_control"]["type"], "ephemeral");
        assert!(value["cache_control"].get("ttl").is_none());
        assert!(!last_message_has_cache_control(&value));
    }

    #[test]
    fn test_static_prefix_ttl_alone_marks_prefix_without_tail_or_top_level() {
        let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: Some(CacheTtl::OneHour),
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
        assert_eq!(
            value["system"]
                .as_array()
                .and_then(|blocks| blocks.last())
                .and_then(|block| block["cache_control"].get("ttl")),
            Some(&json!("1h"))
        );
        assert!(value.get("cache_control").is_none());
        assert!(!last_message_has_cache_control(&value));
    }

    #[test]
    fn test_static_prefix_ttl_five_minutes_with_automatic_1h_errors_client_side() {
        let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

        let error = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: false,
            automatic_caching: true,
            automatic_caching_ttl: Some(CacheTtl::OneHour),
            static_prefix_cache_ttl: Some(CacheTtl::FiveMinutes),
        })
        .unwrap_err();

        let message = error.to_string();
        assert!(
            message.contains("with_static_prefix_cache_ttl"),
            "error should name the knob: {message}"
        );
        assert!(
            message.contains("with_automatic_caching_1h"),
            "error should name the conflicting knob: {message}"
        );
    }

    #[test]
    fn test_static_prefix_ttl_five_minutes_matches_automatic_default_ttl() {
        let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

        // 5m prefix + 5m (default) top-level is uniform, not an inversion.
        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: false,
            automatic_caching: true,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: Some(CacheTtl::FiveMinutes),
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
        // The explicit knob serializes an explicit `"5m"`, equivalent to the
        // omitted-`ttl` default.
        assert_eq!(tools[0]["cache_control"]["ttl"], "5m");
    }

    #[test]
    fn test_static_prefix_ttl_preserves_marker_budget_arithmetic() {
        // Automatic mode reserves one marker for the top-level breakpoint; the
        // static-prefix knob spends from the same remaining budget as manual
        // caching does — two markers (final tool + system), no more.
        let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: false,
            automatic_caching: true,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: Some(CacheTtl::OneHour),
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let marker_count = value["tools"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|tool| !tool["cache_control"].is_null())
            .count()
            + value["system"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|block| !block["cache_control"].is_null())
                .count()
            + usize::from(!value["cache_control"].is_null());
        assert_eq!(marker_count, 3);
        assert!(marker_count <= MAX_CACHE_CONTROL_MARKERS);
    }

    #[test]
    fn test_usage_parses_per_ttl_cache_creation_breakdown() {
        let usage: Usage = serde_json::from_str(
            r#"{
                "input_tokens": 3,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 9677,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 9677,
                    "ephemeral_1h_input_tokens": 0,
                    "ephemeral_24h_input_tokens": 0
                },
                "output_tokens": 7
            }"#,
        )
        .unwrap();

        assert_eq!(usage.cache_creation_input_tokens, Some(9677));
        let cache_creation = usage.cache_creation.unwrap();
        assert_eq!(cache_creation.ephemeral_5m_input_tokens, 9677);
        assert_eq!(cache_creation.ephemeral_1h_input_tokens, 0);
    }

    #[test]
    fn test_usage_without_cache_creation_breakdown_parses_as_none() {
        let usage: Usage =
            serde_json::from_str(r#"{"input_tokens": 3, "output_tokens": 7}"#).unwrap();
        assert!(usage.cache_creation.is_none());
    }

    #[test]
    fn test_raw_top_level_automatic_caching_reduces_marker_budget() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "cache_control": {"type": "ephemeral"},
                "tools": [
                    {
                        "name": "first_cached_tool",
                        "description": "First cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "second_cached_tool",
                        "description": "Second cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "third_cached_tool",
                        "description": "Third cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "name": "fourth_cached_tool",
                        "description": "Fourth cached tool",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            })),
        );

        let err = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap_err();

        assert!(err.to_string().contains("Too many Anthropic tool"));
    }

    #[test]
    fn test_raw_top_level_automatic_caching_1h_errors_after_explicit_five_minute_tool_marker() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
                "tools": [{
                    "name": "cached_tool",
                    "description": "Cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                }]
            })),
        );

        let err = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap_err();

        assert!(err.to_string().contains("ttl `1h`"));
    }

    #[test]
    fn test_typed_automatic_caching_ttl_errors_on_conflicting_raw_top_level_ttl() {
        let request = completion_request_with_tools(
            Vec::new(),
            Some(json!({
                "cache_control": {"type": "ephemeral"}
            })),
        );

        let err = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: false,
            automatic_caching: true,
            automatic_caching_ttl: Some(CacheTtl::OneHour),
            static_prefix_cache_ttl: None,
        })
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("conflicts with the typed automatic caching TTL")
        );
    }

    #[test]
    fn test_prompt_caching_marks_final_tool_in_request() {
        let request = completion_request_with_tools(
            vec![generic_tool("first_tool"), generic_tool("second_tool")],
            None,
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        assert!(tools[0].get("cache_control").is_none());
        assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_prompt_caching_marks_final_additional_tool_in_request() {
        let request = completion_request_with_tools(
            vec![generic_tool("rig_tool")],
            Some(json!({
                "tools": [{
                    "name": "provider_tool",
                    "description": "Provider tool",
                    "input_schema": {"type": "object"}
                }],
                "metadata": {"source": "test"}
            })),
        );

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        let tools = value["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        assert!(tools[0].get("cache_control").is_none());
        assert_eq!(tools[1]["name"], "provider_tool");
        assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
        assert_eq!(value["metadata"]["source"], "test");
    }

    #[test]
    fn test_prompt_caching_without_tools_omits_tools() {
        let request = completion_request_with_tools(Vec::new(), None);

        let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: true,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        })
        .unwrap();

        let value = serde_json::to_value(request).unwrap();
        assert!(value.get("tools").is_none());
    }

    #[test]
    fn test_plaintext_document_serialization() {
        let content = Content::Document {
            source: DocumentSource::Text {
                data: "Hello, world!".to_string(),
                media_type: PlainTextMediaType::Plain,
            },
            title: None,
            context: None,
            citations: None,
            cache_control: None,
        };

        let json = serde_json::to_value(&content).unwrap();
        assert_eq!(json["type"], "document");
        assert_eq!(json["source"]["type"], "text");
        assert_eq!(json["source"]["media_type"], "text/plain");
        assert_eq!(json["source"]["data"], "Hello, world!");
    }

    #[test]
    fn test_plaintext_document_deserialization() {
        let json = r#"
        {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": "Hello, world!"
            }
        }
        "#;

        let content: Content = serde_json::from_str(json).unwrap();
        match content {
            Content::Document {
                source,
                cache_control,
                ..
            } => {
                assert_eq!(
                    source,
                    DocumentSource::Text {
                        data: "Hello, world!".to_string(),
                        media_type: PlainTextMediaType::Plain,
                    }
                );
                assert_eq!(cache_control, None);
            }
            _ => panic!("Expected Document content"),
        }
    }

    #[test]
    fn test_base64_pdf_document_serialization() {
        let content = Content::Document {
            source: DocumentSource::Base64 {
                data: "base64data".to_string(),
                media_type: DocumentFormat::PDF,
            },
            title: None,
            context: None,
            citations: None,
            cache_control: None,
        };

        let json = serde_json::to_value(&content).unwrap();
        assert_eq!(json["type"], "document");
        assert_eq!(json["source"]["type"], "base64");
        assert_eq!(json["source"]["media_type"], "application/pdf");
        assert_eq!(json["source"]["data"], "base64data");
    }

    #[test]
    fn test_base64_pdf_document_deserialization() {
        let json = r#"
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "base64data"
            }
        }
        "#;

        let content: Content = serde_json::from_str(json).unwrap();
        match content {
            Content::Document { source, .. } => {
                assert_eq!(
                    source,
                    DocumentSource::Base64 {
                        data: "base64data".to_string(),
                        media_type: DocumentFormat::PDF,
                    }
                );
            }
            _ => panic!("Expected Document content"),
        }
    }

    #[test]
    fn test_file_id_document_serialization() {
        let content = Content::Document {
            source: DocumentSource::File {
                file_id: "file_abc".to_string(),
            },
            title: None,
            context: None,
            citations: None,
            cache_control: None,
        };

        let json = serde_json::to_value(&content).unwrap();
        assert_eq!(json["type"], "document");
        assert_eq!(json["source"]["type"], "file");
        assert_eq!(json["source"]["file_id"], "file_abc");
    }

    #[test]
    fn test_file_id_document_deserialization() {
        let json = r#"
        {
            "type": "document",
            "source": {
                "type": "file",
                "file_id": "file_abc"
            }
        }
        "#;

        let content: Content = serde_json::from_str(json).unwrap();
        match content {
            Content::Document { source, .. } => {
                assert_eq!(
                    source,
                    DocumentSource::File {
                        file_id: "file_abc".to_string(),
                    }
                );
            }
            _ => panic!("Expected Document content"),
        }
    }

    #[test]
    fn test_file_id_rig_to_anthropic_conversion() {
        use crate::completion::message as msg;

        let rig_message = msg::Message::User {
            content: vec![msg::UserContent::Document(msg::Document {
                data: DocumentSourceKind::FileId("file_abc".to_string()),
                media_type: None,
                additional_params: None,
            })],
        };

        let anthropic_message: Message = rig_message.try_into().unwrap();
        assert_eq!(anthropic_message.role, Role::User);

        let mut iter = anthropic_message.content.into_iter();
        match iter.next().unwrap() {
            Content::Document { source, .. } => {
                assert_eq!(
                    source,
                    DocumentSource::File {
                        file_id: "file_abc".to_string(),
                    }
                );
            }
            other => panic!("Expected Document content, got: {other:?}"),
        }
    }

    #[test]
    fn test_file_id_anthropic_to_rig_conversion() {
        use crate::completion::message as msg;

        let anthropic_message = Message {
            role: Role::User,
            content: vec![Content::Document {
                source: DocumentSource::File {
                    file_id: "file_abc".to_string(),
                },
                title: None,
                context: None,
                citations: None,
                cache_control: None,
            }],
        };

        let rig_message: msg::Message = anthropic_message.try_into().unwrap();
        match rig_message {
            msg::Message::User { content } => {
                let mut iter = content.into_iter();
                match iter.next().unwrap() {
                    msg::UserContent::Document(msg::Document {
                        data, media_type, ..
                    }) => {
                        assert_eq!(data, DocumentSourceKind::FileId("file_abc".to_string()));
                        assert_eq!(media_type, None);
                    }
                    other => panic!("Expected Document content, got: {other:?}"),
                }
            }
            _ => panic!("Expected User message"),
        }
    }

    #[test]
    fn test_plaintext_rig_to_anthropic_conversion() {
        use crate::completion::message as msg;

        let rig_message = msg::Message::User {
            content: vec![msg::UserContent::document(
                "Some plain text content".to_string(),
                Some(msg::DocumentMediaType::TXT),
            )],
        };

        let anthropic_message: Message = rig_message.try_into().unwrap();
        assert_eq!(anthropic_message.role, Role::User);

        let mut iter = anthropic_message.content.into_iter();
        match iter.next().unwrap() {
            Content::Document { source, .. } => {
                assert_eq!(
                    source,
                    DocumentSource::Text {
                        data: "Some plain text content".to_string(),
                        media_type: PlainTextMediaType::Plain,
                    }
                );
            }
            other => panic!("Expected Document content, got: {other:?}"),
        }
    }

    #[test]
    fn test_plaintext_anthropic_to_rig_conversion() {
        use crate::completion::message as msg;

        let anthropic_message = Message {
            role: Role::User,
            content: vec![Content::Document {
                source: DocumentSource::Text {
                    data: "Some plain text content".to_string(),
                    media_type: PlainTextMediaType::Plain,
                },
                title: None,
                context: None,
                citations: None,
                cache_control: None,
            }],
        };

        let rig_message: msg::Message = anthropic_message.try_into().unwrap();
        match rig_message {
            msg::Message::User { content } => {
                let mut iter = content.into_iter();
                match iter.next().unwrap() {
                    msg::UserContent::Document(msg::Document {
                        data, media_type, ..
                    }) => {
                        assert_eq!(
                            data,
                            DocumentSourceKind::String("Some plain text content".into())
                        );
                        assert_eq!(media_type, Some(msg::DocumentMediaType::TXT));
                    }
                    other => panic!("Expected Document content, got: {other:?}"),
                }
            }
            _ => panic!("Expected User message"),
        }
    }

    #[test]
    fn test_plaintext_roundtrip_rig_to_anthropic_and_back() {
        use crate::completion::message as msg;

        let original = msg::Message::User {
            content: vec![msg::UserContent::document(
                "Round trip text".to_string(),
                Some(msg::DocumentMediaType::TXT),
            )],
        };

        let anthropic: Message = original.clone().try_into().unwrap();
        let back: msg::Message = anthropic.try_into().unwrap();

        match (&original, &back) {
            (
                msg::Message::User {
                    content: orig_content,
                },
                msg::Message::User {
                    content: back_content,
                },
            ) => match (orig_content.first(), back_content.first()) {
                (
                    Some(msg::UserContent::Document(msg::Document {
                        media_type: orig_mt,
                        ..
                    })),
                    Some(msg::UserContent::Document(msg::Document {
                        media_type: back_mt,
                        ..
                    })),
                ) => {
                    assert_eq!(orig_mt, back_mt);
                }
                _ => panic!("Expected Document content in both"),
            },
            _ => panic!("Expected User messages"),
        }
    }

    #[test]
    fn test_unsupported_document_type_returns_error() {
        use crate::completion::message as msg;

        let rig_message = msg::Message::User {
            content: vec![msg::UserContent::Document(msg::Document {
                data: DocumentSourceKind::String("data".into()),
                media_type: Some(msg::DocumentMediaType::HTML),
                additional_params: None,
            })],
        };

        let result: Result<Message, _> = rig_message.try_into();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Anthropic only supports PDF and plain text documents"),
            "Unexpected error: {err}"
        );
    }

    #[test]
    fn test_plaintext_document_url_source_returns_error() {
        use crate::completion::message as msg;

        let rig_message = msg::Message::User {
            content: vec![msg::UserContent::Document(msg::Document {
                data: DocumentSourceKind::Url("https://example.com/doc.txt".into()),
                media_type: Some(msg::DocumentMediaType::TXT),
                additional_params: None,
            })],
        };

        let result: Result<Message, _> = rig_message.try_into();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Only string or base64 data is supported for plain text documents"),
            "Unexpected error: {err}"
        );
    }

    #[test]
    fn test_plaintext_document_with_cache_control() {
        let content = Content::Document {
            source: DocumentSource::Text {
                data: "cached text".to_string(),
                media_type: PlainTextMediaType::Plain,
            },
            title: None,
            context: None,
            citations: None,
            cache_control: Some(CacheControl::ephemeral()),
        };

        let json = serde_json::to_value(&content).unwrap();
        assert_eq!(json["source"]["type"], "text");
        assert_eq!(json["source"]["media_type"], "text/plain");
        assert_eq!(json["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_message_with_plaintext_document_deserialization() {
        let json = r#"
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "text",
                        "media_type": "text/plain",
                        "data": "Hello from a text file"
                    }
                },
                {
                    "type": "text",
                    "text": "Summarize this document."
                }
            ]
        }
        "#;

        let message: Message = serde_json::from_str(json).unwrap();
        assert_eq!(message.role, Role::User);
        assert_eq!(message.content.len(), 2);

        let mut iter = message.content.into_iter();

        match iter.next().unwrap() {
            Content::Document { source, .. } => {
                assert_eq!(
                    source,
                    DocumentSource::Text {
                        data: "Hello from a text file".to_string(),
                        media_type: PlainTextMediaType::Plain,
                    }
                );
            }
            _ => panic!("Expected Document content"),
        }

        match iter.next().unwrap() {
            Content::Text { text, .. } => {
                assert_eq!(text, "Summarize this document.");
            }
            _ => panic!("Expected Text content"),
        }
    }

    #[test]
    fn test_assistant_reasoning_multiblock_to_anthropic_content() {
        let reasoning = message::Reasoning {
            id: None,
            content: vec![
                message::ReasoningContent::Text {
                    text: "step one".to_string(),
                    signature: Some("sig-1".to_string()),
                },
                message::ReasoningContent::Summary("summary".to_string()),
                message::ReasoningContent::Text {
                    text: "step two".to_string(),
                    signature: Some("sig-2".to_string()),
                },
                message::ReasoningContent::Redacted {
                    data: "redacted block".to_string(),
                },
            ],
        };

        let msg = message::Message::Assistant {
            id: None,
            content: vec![message::AssistantContent::Reasoning(reasoning)],
        };
        let converted: Message = msg.try_into().expect("convert assistant message");
        let converted_content = converted.content.clone();

        assert_eq!(converted.role, Role::Assistant);
        assert_eq!(converted_content.len(), 4);
        assert!(matches!(
            converted_content.first(),
            Some(Content::Thinking { thinking, signature: Some(signature) })
                if thinking == "step one" && signature == "sig-1"
        ));
        assert!(matches!(
            converted_content.get(1),
            Some(Content::Thinking { thinking, signature: None }) if thinking == "summary"
        ));
        assert!(matches!(
            converted_content.get(2),
            Some(Content::Thinking { thinking, signature: Some(signature) })
                if thinking == "step two" && signature == "sig-2"
        ));
        assert!(matches!(
            converted_content.get(3),
            Some(Content::RedactedThinking { data }) if data == "redacted block"
        ));
    }

    #[test]
    fn test_redacted_thinking_content_to_assistant_reasoning() {
        let content = Content::RedactedThinking {
            data: "opaque-redacted".to_string(),
        };
        let converted: message::AssistantContent =
            content.try_into().expect("convert redacted thinking");

        assert!(matches!(
            converted,
            message::AssistantContent::Reasoning(message::Reasoning { content, .. })
                if matches!(
                    content.first(),
                    Some(message::ReasoningContent::Redacted { data }) if data == "opaque-redacted"
                )
        ));
    }

    #[test]
    fn test_assistant_encrypted_reasoning_maps_to_redacted_thinking() {
        let reasoning = message::Reasoning {
            id: None,
            content: vec![message::ReasoningContent::Encrypted(
                "ciphertext".to_string(),
            )],
        };
        let msg = message::Message::Assistant {
            id: None,
            content: vec![message::AssistantContent::Reasoning(reasoning)],
        };

        let converted: Message = msg.try_into().expect("convert assistant message");
        let converted_content = converted.content.clone();

        assert_eq!(converted_content.len(), 1);
        assert!(matches!(
            converted_content.first(),
            Some(Content::RedactedThinking { data }) if data == "ciphertext"
        ));
    }

    #[test]
    fn empty_end_turn_response_normalizes_to_an_empty_choice() {
        let response = CompletionResponse {
            content: vec![],
            id: "msg_123".to_string(),
            model: CLAUDE_SONNET_4_6.to_string(),
            role: "assistant".to_string(),
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            provider_request_id: None,
            usage: Usage {
                input_tokens: 7,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
                cache_creation: None,
                output_tokens: 2,
                output_tokens_details: None,
            },
        };

        let parsed: completion::CompletionResponse = response
            .normalize("anthropic")
            .expect("empty end_turn should not error");

        // Anthropic's documented empty `end_turn` is a turn that carried
        // nothing. It used to normalize to one fabricated empty-text part
        // because the content type could not be empty; the empty list is the
        // same turn, said honestly. Everything else about the response is
        // unchanged, which is the point of asserting it here.
        assert!(parsed.choice.is_empty());
        assert_eq!(parsed.provider, "anthropic");
        assert_eq!(parsed.message_id.as_deref(), Some("msg_123"));
        assert_eq!(parsed.model.as_deref(), Some(CLAUDE_SONNET_4_6));
        assert_eq!(parsed.finish_reason(), Some(completion::FinishReason::Stop));
    }

    /// Build an empty-content response with the given terminal, for exercising
    /// the two legal empty cases against everything else.
    fn empty_response_with(
        stop_reason: Option<&str>,
        stop_sequence: Option<&str>,
    ) -> CompletionResponse {
        CompletionResponse {
            content: vec![],
            id: "msg_123".to_string(),
            model: CLAUDE_SONNET_4_6.to_string(),
            role: "assistant".to_string(),
            stop_reason: stop_reason.map(str::to_string),
            stop_sequence: stop_sequence.map(str::to_string),
            provider_request_id: None,
            usage: Usage {
                input_tokens: 7,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
                cache_creation: None,
                output_tokens: 2,
                output_tokens_details: None,
            },
        }
    }

    #[test]
    fn empty_response_outside_the_legal_terminals_still_errors() {
        for (stop_reason, stop_sequence) in [
            (Some("tool_use"), None),
            (Some("max_tokens"), None),
            (Some("refusal"), None),
            (Some("pause_turn"), None),
            (None, None),
            // Claims to have stopped on a sequence but names none: the
            // malformed shape the guard exists for, not a legal empty turn.
            (Some("stop_sequence"), None),
            // The inverse: naming a sequence does not make an illegal terminal
            // legal. The carve-out gates on the reason first, then the field.
            (Some("max_tokens"), Some("alpha")),
        ] {
            let err = empty_response_with(stop_reason, stop_sequence)
                .normalize("anthropic")
                .expect_err(&format!(
                    "empty {stop_reason:?} response should remain an error"
                ));

            assert!(matches!(
                err,
                CompletionError::ResponseError(message) if message == EMPTY_RESPONSE_ERROR
            ));
        }
    }

    #[test]
    fn empty_stop_sequence_response_naming_its_sequence_is_a_completed_turn() {
        let parsed = empty_response_with(Some("stop_sequence"), Some("alpha"))
            .normalize("anthropic")
            .expect("a completed stop-sequence turn must not normalize into an error");

        assert!(parsed.choice.is_empty());
        assert_eq!(parsed.finish_reason(), Some(completion::FinishReason::Stop));
    }

    #[test]
    fn stop_reason_maps_onto_the_normalized_vocabulary() {
        assert_eq!(
            map_finish_reason("end_turn"),
            completion::FinishReason::Stop
        );
        assert_eq!(
            map_finish_reason("stop_sequence"),
            completion::FinishReason::Stop
        );
        assert_eq!(
            map_finish_reason("max_tokens"),
            completion::FinishReason::Length
        );
        assert_eq!(
            map_finish_reason("tool_use"),
            completion::FinishReason::ToolCalls
        );
        assert_eq!(
            map_finish_reason("refusal"),
            completion::FinishReason::ContentFilter
        );
    }

    #[test]
    fn unknown_stop_reason_is_preserved_verbatim() {
        // Anthropic's own spelling survives, so a reason this crate does not yet
        // model never reads as a natural stop.
        assert_eq!(
            map_finish_reason("pause_turn"),
            completion::FinishReason::Other("pause_turn".to_owned())
        );
        assert_eq!(
            map_finish_reason("model_context_window_exceeded"),
            completion::FinishReason::Other("model_context_window_exceeded".to_owned())
        );
    }

    #[test]
    fn end_turn_with_a_tool_call_is_reconciled_to_tool_calls() {
        // Anthropic reports `tool_use`, but the reconciliation the response
        // builder applies must hold for any provider that reports a plain stop
        // alongside a tool call.
        let response = CompletionResponse {
            content: vec![Content::ToolUse {
                id: "toolu_1".to_string(),
                name: "add".to_string(),
                input: json!({"x": 1}),
            }],
            id: "msg_123".to_string(),
            model: CLAUDE_SONNET_4_6.to_string(),
            role: "assistant".to_string(),
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            provider_request_id: None,
            usage: Usage {
                input_tokens: 7,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
                cache_creation: None,
                output_tokens: 2,
                output_tokens_details: None,
            },
        };

        let parsed = response
            .normalize("anthropic")
            .expect("tool-use response should normalize");

        assert_eq!(
            parsed.finish_reason(),
            Some(completion::FinishReason::ToolCalls)
        );
    }

    #[test]
    fn test_tool_result_content_in_message_roundtrip() {
        let message_json = r#"{
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here is the screenshot:"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgo..."
                            }
                        }
                    ]
                }
            ]
        }"#;

        let message: Message = serde_json::from_str(message_json).unwrap();
        let serialized = serde_json::to_value(&message).unwrap();

        let tool_result = &serialized["content"][0];
        assert_eq!(tool_result["type"], "tool_result");

        let image_content = &tool_result["content"][1];
        assert_eq!(image_content["type"], "image");
        assert_eq!(image_content["source"]["type"], "base64");
        assert_eq!(image_content["source"]["media_type"], "image/png");
        assert_eq!(image_content["source"]["data"], "iVBORw0KGgo...");
    }

    // -------------------------------------------------------------------
    // Citations (#1767)
    // -------------------------------------------------------------------

    #[test]
    fn document_serializes_citations_and_metadata() {
        let doc = Content::Document {
            source: DocumentSource::Text {
                data: "hello".into(),
                media_type: PlainTextMediaType::Plain,
            },
            title: Some("My Doc".into()),
            context: None,
            citations: Some(CitationsConfig { enabled: true }),
            cache_control: None,
        };
        let value = serde_json::to_value(&doc).unwrap();
        assert_eq!(value["citations"]["enabled"], true);
        assert_eq!(value["title"], "My Doc");
        assert!(
            value.get("context").is_none(),
            "context should be skipped when None"
        );
    }

    #[test]
    fn text_serializes_without_citations_when_empty() {
        let content = Content::Text {
            text: "hello".into(),
            citations: Vec::new(),
            cache_control: None,
        };
        let value = serde_json::to_value(&content).unwrap();
        assert!(
            value.get("citations").is_none(),
            "empty citations vec must be skipped"
        );
    }

    #[test]
    fn text_deserializes_char_location_citation() {
        let value = json!({
            "type": "text",
            "text": "the grass is green",
            "citations": [{
                "type": "char_location",
                "cited_text": "The grass is green.",
                "document_index": 0,
                "document_title": "Example",
                "start_char_index": 0,
                "end_char_index": 20
            }]
        });
        let parsed: Content = serde_json::from_value(value).unwrap();
        let Content::Text { citations, .. } = parsed else {
            panic!("expected Content::Text");
        };
        assert_eq!(citations.len(), 1);
        let Citation::CharLocation(citation) = &citations[0] else {
            panic!("expected CharLocation");
        };
        assert_eq!(citation.start_char_index, 0);
        assert_eq!(citation.end_char_index, 20);
    }

    #[test]
    fn text_deserializes_search_result_location_citation() {
        let value = json!({
            "type": "text",
            "text": "API keys are required.",
            "citations": [{
                "type": "search_result_location",
                "cited_text": "All API requests must include an API key.",
                "source": "https://docs.example.com/api-reference",
                "title": "API Reference",
                "search_result_index": 0,
                "start_block_index": 0,
                "end_block_index": 1
            }]
        });

        let parsed: Content = serde_json::from_value(value).unwrap();
        let Content::Text { citations, .. } = parsed else {
            panic!("expected Content::Text");
        };

        assert!(matches!(
            &citations[0],
            Citation::SearchResultLocation(SearchResultLocationCitation {
                source,
                title: Some(title),
                search_result_index: 0,
                start_block_index: 0,
                end_block_index: 1,
                ..
            }) if source == "https://docs.example.com/api-reference" && title == "API Reference"
        ));
    }

    #[test]
    fn text_deserializes_web_search_result_location_citation() {
        let value = json!({
            "type": "text",
            "text": "Claude Shannon worked at Bell Labs.",
            "citations": [{
                "type": "web_search_result_location",
                "cited_text": "Claude Shannon was a mathematician.",
                "url": "https://example.com/shannon",
                "title": "Claude Shannon",
                "encrypted_index": "encrypted-reference"
            }]
        });

        let parsed: Content = serde_json::from_value(value).unwrap();
        let Content::Text { citations, .. } = parsed else {
            panic!("expected Content::Text");
        };

        assert!(matches!(
            &citations[0],
            Citation::WebSearchResultLocation(WebSearchResultLocationCitation {
                url,
                title,
                encrypted_index,
                ..
            }) if url == "https://example.com/shannon"
                && title.as_deref() == Some("Claude Shannon")
                && encrypted_index == "encrypted-reference"
        ));
    }

    #[test]
    fn text_deserializes_web_search_result_location_citation_with_null_title() {
        let value = json!({
            "type": "text",
            "text": "Claude Shannon worked at Bell Labs.",
            "citations": [{
                "type": "web_search_result_location",
                "cited_text": "Claude Shannon was a mathematician.",
                "url": "https://example.com/shannon",
                "title": null,
                "encrypted_index": "encrypted-reference"
            }]
        });

        let parsed: Content = serde_json::from_value(value).unwrap();
        let Content::Text { citations, .. } = parsed else {
            panic!("expected Content::Text");
        };

        let Citation::WebSearchResultLocation(citation) = &citations[0] else {
            panic!("expected WebSearchResultLocation");
        };
        assert_eq!(citation.title, None);

        let serialized = serde_json::to_value(&citations[0]).unwrap();
        assert!(serialized.get("title").is_some());
        assert!(serialized["title"].is_null());
    }

    #[test]
    fn web_search_response_preserves_raw_blocks_and_citations() {
        let value = json!({
            "id": "msg_web_search",
            "model": CLAUDE_SONNET_4_6,
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20
            },
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_01",
                    "name": "web_search",
                    "input": {
                        "query": "claude shannon birth date"
                    }
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_01",
                    "content": [
                        {
                            "type": "web_search_result",
                            "url": "https://example.com/shannon",
                            "title": "Claude Shannon",
                            "encrypted_content": "encrypted-content",
                            "page_age": "April 30, 2025"
                        }
                    ]
                },
                {
                    "type": "text",
                    "text": "Claude Shannon was born on April 30, 1916.",
                    "citations": [{
                        "type": "web_search_result_location",
                        "cited_text": "Claude Shannon was born on April 30, 1916.",
                        "url": "https://example.com/shannon",
                        "title": "Claude Shannon",
                        "encrypted_index": "encrypted-index"
                    }]
                }
            ]
        });

        let response: CompletionResponse = serde_json::from_value(value).unwrap();
        // The wire response is consumed by the conversion, so read the
        // provider-native text off it first.
        let raw_text_response = response.get_text_response();
        let converted = response.normalize("anthropic").unwrap();
        assert_eq!(converted.choice.len(), 3);
        assert_eq!(
            raw_text_response.as_deref(),
            Some("Claude Shannon was born on April 30, 1916.")
        );

        let items = converted.choice.iter().collect::<Vec<_>>();
        let message::AssistantContent::Text(server_tool_use) = items[0] else {
            panic!("expected raw server_tool_use metadata");
        };
        assert_eq!(server_tool_use.text, "");
        assert_eq!(
            server_tool_use.additional_params.as_ref().unwrap()[ANTHROPIC_RAW_CONTENT_KEY]["type"],
            "server_tool_use"
        );

        let message::AssistantContent::Text(web_search_result) = items[1] else {
            panic!("expected raw web_search_tool_result metadata");
        };
        assert_eq!(
            web_search_result.additional_params.as_ref().unwrap()[ANTHROPIC_RAW_CONTENT_KEY]["content"]
                [0]["encrypted_content"],
            "encrypted-content"
        );

        let message::AssistantContent::Text(answer) = items[2] else {
            panic!("expected text answer");
        };
        let citations = anthropic_citations(answer).unwrap();
        assert!(matches!(
            citations.first(),
            Some(Citation::WebSearchResultLocation(citation))
                if citation.encrypted_index == "encrypted-index"
        ));

        let round_trip: Message = message::Message::Assistant {
            id: converted.message_id.clone(),
            content: converted.choice,
        }
        .try_into()
        .unwrap();

        let round_trip_items = round_trip.content.iter().collect::<Vec<_>>();
        assert!(matches!(
            round_trip_items.first(),
            Some(Content::ServerToolUse { id, name, input })
                if id == "srvtoolu_01"
                    && name == "web_search"
                    && input["query"] == "claude shannon birth date"
        ));
        assert!(matches!(
            round_trip_items.get(1),
            Some(Content::WebSearchToolResult {
                tool_use_id,
                content
            }) if tool_use_id == "srvtoolu_01"
                && content[0]["encrypted_content"] == "encrypted-content"
        ));
    }

    #[test]
    fn web_search_tool_result_error_object_is_preserved_raw() {
        let value = json!({
            "id": "msg_web_search_error",
            "model": CLAUDE_SONNET_4_6,
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2
            },
            "content": [{
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_01",
                "content": {
                    "type": "web_search_tool_result_error",
                    "error_code": "max_uses_exceeded"
                }
            }]
        });

        let response: CompletionResponse = serde_json::from_value(value).unwrap();
        let converted = response.normalize("anthropic").unwrap();
        let Some(message::AssistantContent::Text(web_search_result)) = converted.choice.first()
        else {
            panic!("expected raw web_search_tool_result metadata");
        };

        let raw_content =
            &web_search_result.additional_params.as_ref().unwrap()[ANTHROPIC_RAW_CONTENT_KEY];
        assert_eq!(raw_content["type"], "web_search_tool_result");
        assert_eq!(raw_content["content"]["error_code"], "max_uses_exceeded");
        assert_eq!(
            raw_content["content"]["type"],
            "web_search_tool_result_error"
        );

        let round_trip: Message = message::Message::Assistant {
            id: converted.message_id,
            content: converted.choice,
        }
        .try_into()
        .unwrap();

        assert!(matches!(
            round_trip.content.first(),
            Some(Content::WebSearchToolResult {
                tool_use_id,
                content
            }) if tool_use_id == "srvtoolu_01"
                && content["error_code"] == "max_uses_exceeded"
        ));
    }

    #[test]
    fn code_execution_tool_result_variants_deserialize() {
        let normal: Content = serde_json::from_value(json!({
            "type": "code_execution_tool_result",
            "tool_use_id": "srvtoolu_normal",
            "content": {
                "type": "code_execution_result",
                "return_code": 0,
                "stdout": "42\n",
                "stderr": "",
                "content": []
            }
        }))
        .unwrap();
        assert!(matches!(
            normal,
            Content::CodeExecutionToolResult {
                ref tool_use_id,
                ref content
            } if tool_use_id == "srvtoolu_normal"
                && content["type"] == "code_execution_result"
                && content["stdout"] == "42\n"
        ));

        let encrypted: Content = serde_json::from_value(json!({
            "type": "code_execution_tool_result",
            "tool_use_id": "srvtoolu_encrypted",
            "content": {
                "type": "encrypted_code_execution_result",
                "return_code": 1,
                "stderr": "failure",
                "encrypted_stdout": "encrypted-output",
                "content": []
            }
        }))
        .unwrap();
        assert!(matches!(
            encrypted,
            Content::CodeExecutionToolResult {
                ref tool_use_id,
                ref content
            } if tool_use_id == "srvtoolu_encrypted"
                && content["type"] == "encrypted_code_execution_result"
                && content["encrypted_stdout"] == "encrypted-output"
        ));
    }

    #[test]
    fn code_execution_tool_result_is_preserved_and_round_trips() {
        let raw_block = json!({
            "type": "code_execution_tool_result",
            "tool_use_id": "srvtoolu_01",
            "content": {
                "type": "code_execution_result",
                "return_code": 0,
                "stdout": "42\n",
                "stderr": "",
                "content": []
            }
        });
        let value = json!({
            "id": "msg_code_execution",
            "model": CLAUDE_OPUS_4_8,
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20
            },
            "content": [raw_block.clone()]
        });

        let response: CompletionResponse = serde_json::from_value(value).unwrap();
        let converted = response.normalize("anthropic").unwrap();
        let Some(message::AssistantContent::Text(code_execution_result)) = converted.choice.first()
        else {
            panic!("expected raw code_execution_tool_result metadata");
        };
        assert_eq!(
            code_execution_result.additional_params.as_ref().unwrap()[ANTHROPIC_RAW_CONTENT_KEY],
            raw_block
        );

        let round_trip: Message = message::Message::Assistant {
            id: converted.message_id,
            content: converted.choice,
        }
        .try_into()
        .unwrap();
        assert!(matches!(
            round_trip.content.first(),
            Some(Content::CodeExecutionToolResult {
                tool_use_id,
                content
            }) if tool_use_id == "srvtoolu_01"
                && content["type"] == "code_execution_result"
                && content["stdout"] == "42\n"
        ));
    }

    #[test]
    fn text_deserializes_unknown_citation_without_failing() {
        let value = json!({
            "type": "text",
            "text": "future citation",
            "citations": [{
                "type": "future_location",
                "cited_text": "future text",
                "new_field": "kept"
            }]
        });

        let parsed: Content = serde_json::from_value(value).unwrap();
        let Content::Text { citations, .. } = parsed else {
            panic!("expected Content::Text");
        };

        assert!(matches!(
            &citations[0],
            Citation::Unknown(raw)
                if raw["type"] == "future_location" && raw["new_field"] == "kept"
        ));
    }

    #[test]
    fn page_location_citation_roundtrips() {
        let citation = Citation::PageLocation(PageLocationCitation {
            cited_text: "Water is essential for life.".into(),
            document_index: 1,
            document_title: Some("PDF Doc".into()),
            start_page_number: 5,
            end_page_number: 6,
        });
        let value = serde_json::to_value(&citation).unwrap();
        assert_eq!(value["type"], "page_location");
        assert_eq!(value["start_page_number"], 5);
        let back: Citation = serde_json::from_value(value).unwrap();
        assert_eq!(back, citation);
    }

    #[test]
    fn content_block_location_citation_roundtrips() {
        let citation = Citation::ContentBlockLocation(ContentBlockLocationCitation {
            cited_text: "These are important findings.".into(),
            document_index: 2,
            document_title: None,
            start_block_index: 0,
            end_block_index: 1,
        });
        let value = serde_json::to_value(&citation).unwrap();
        assert_eq!(value["type"], "content_block_location");
        assert!(value.get("document_title").is_none());
        let back: Citation = serde_json::from_value(value).unwrap();
        assert_eq!(back, citation);
    }

    #[test]
    fn anthropic_citations_extracts_from_additional_params() {
        let text = message::Text {
            text: "the grass is green".into(),
            additional_params: crate::message::AdditionalParams::try_from_value(json!({
                "citations": [{
                    "type": "char_location",
                    "cited_text": "The grass is green.",
                    "document_index": 0,
                    "start_char_index": 0,
                    "end_char_index": 20
                }]
            }))
            .expect("object params"),
        };
        let citations = anthropic_citations(&text).unwrap();
        assert_eq!(citations.len(), 1);
    }

    #[test]
    fn anthropic_citations_returns_empty_when_absent() {
        let text = message::Text::new("hello".to_string());
        assert!(anthropic_citations(&text).unwrap().is_empty());
    }

    #[test]
    fn content_text_with_citations_survives_assistant_conversion() {
        let content = Content::Text {
            text: "the grass is green".into(),
            citations: vec![Citation::CharLocation(CharLocationCitation {
                cited_text: "The grass is green.".into(),
                document_index: 0,
                document_title: None,
                start_char_index: 0,
                end_char_index: 20,
            })],
            cache_control: None,
        };
        let assistant: message::AssistantContent = content.try_into().unwrap();
        let message::AssistantContent::Text(text) = assistant else {
            panic!("expected text variant");
        };
        let recovered = anthropic_citations(&text).unwrap();
        assert_eq!(recovered.len(), 1);
    }

    #[test]
    fn provider_text_response_concatenates_text_blocks_without_inserted_newlines() {
        let response = CompletionResponse {
            content: vec![
                Content::Text {
                    text: "According to the document, ".into(),
                    citations: Vec::new(),
                    cache_control: None,
                },
                Content::Text {
                    text: "the grass is green".into(),
                    citations: Vec::new(),
                    cache_control: None,
                },
                Content::Text {
                    text: " and the sky is blue.".into(),
                    citations: Vec::new(),
                    cache_control: None,
                },
            ],
            id: "msg_1".into(),
            model: "claude-test".into(),
            role: "assistant".into(),
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            provider_request_id: None,
            usage: Usage {
                input_tokens: 1,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
                cache_creation: None,
                output_tokens: 1,
                output_tokens_details: None,
            },
        };

        assert_eq!(
            response.get_text_response().as_deref(),
            Some("According to the document, the grass is green and the sky is blue.")
        );
    }

    #[test]
    fn assistant_text_citations_survive_anthropic_request_conversion() {
        let assistant = message::Message::Assistant {
            id: None,
            content: vec![message::AssistantContent::Text(message::Text {
                text: "the grass is green".into(),
                additional_params: crate::message::AdditionalParams::try_from_value(json!({
                    "citations": [{
                        "type": "char_location",
                        "cited_text": "The grass is green.",
                        "document_index": 0,
                        "start_char_index": 0,
                        "end_char_index": 20
                    }]
                }))
                .expect("object params"),
            })],
        };

        let converted: Message = assistant.try_into().unwrap();
        let Some(Content::Text {
            citations, text, ..
        }) = converted.content.first()
        else {
            panic!("expected assistant text content");
        };

        assert_eq!(text, "the grass is green");
        assert_eq!(
            citations,
            &vec![Citation::CharLocation(CharLocationCitation {
                cited_text: "The grass is green.".into(),
                document_index: 0,
                document_title: None,
                start_char_index: 0,
                end_char_index: 20,
            })]
        );
    }

    #[test]
    fn assistant_text_invalid_known_citations_are_rejected_for_anthropic_request_conversion() {
        let text = message::AssistantContent::Text(message::Text {
            text: "bad citation".into(),
            additional_params: crate::message::AdditionalParams::try_from_value(json!({
                "citations": [{
                    "type": "char_location",
                    "cited_text": "bad"
                }]
            }))
            .expect("object params"),
        });

        let result = anthropic_content_from_assistant_content(text);

        assert!(
            result.is_err(),
            "invalid Anthropic citation metadata should not be silently dropped"
        );
    }

    #[test]
    fn document_additional_params_forward_to_anthropic_document() {
        let doc = message::UserContent::Document(message::Document {
            data: message::DocumentSourceKind::String("Hello world.".into()),
            media_type: Some(message::DocumentMediaType::TXT),
            additional_params: crate::message::AdditionalParams::try_from_value(json!({
                "title": "Doc1",
                "context": "ctx",
                "citations": { "enabled": true }
            }))
            .expect("object params"),
        });
        let msg = message::Message::User { content: vec![doc] };
        let converted: Message = msg.try_into().unwrap();
        let block = converted.content.first();
        let Some(Content::Document {
            title,
            context,
            citations,
            ..
        }) = block
        else {
            panic!("expected Content::Document");
        };
        assert_eq!(title.as_deref(), Some("Doc1"));
        assert_eq!(context.as_deref(), Some("ctx"));
        assert_eq!(citations, &Some(CitationsConfig { enabled: true }));
    }

    fn assert_reverse_document_metadata(
        source: DocumentSource,
        expected_data: DocumentSourceKind,
        expected_media_type: Option<message::DocumentMediaType>,
    ) -> message::Message {
        let provider_message = Message {
            role: Role::User,
            content: vec![Content::Document {
                source,
                title: Some("Doc1".into()),
                context: Some("ctx".into()),
                citations: Some(CitationsConfig { enabled: true }),
                cache_control: None,
            }],
        };

        let generic: message::Message = provider_message.try_into().unwrap();
        let message::Message::User { content } = &generic else {
            panic!("expected generic user message");
        };
        let Some(message::UserContent::Document(document)) = content.first() else {
            panic!("expected generic document");
        };

        assert_eq!(document.data, expected_data);
        assert_eq!(document.media_type, expected_media_type);
        let additional_params = document
            .additional_params
            .as_ref()
            .expect("expected Anthropic document metadata");
        assert_eq!(additional_params["title"], "Doc1");
        assert_eq!(additional_params["context"], "ctx");
        assert_eq!(additional_params["citations"]["enabled"], true);

        generic
    }

    #[test]
    fn anthropic_document_metadata_survives_reverse_conversion_for_all_sources() {
        assert_reverse_document_metadata(
            DocumentSource::Text {
                data: "Hello world.".into(),
                media_type: PlainTextMediaType::Plain,
            },
            DocumentSourceKind::String("Hello world.".into()),
            Some(message::DocumentMediaType::TXT),
        );
        assert_reverse_document_metadata(
            DocumentSource::Base64 {
                data: "base64-pdf".into(),
                media_type: DocumentFormat::PDF,
            },
            DocumentSourceKind::String("base64-pdf".into()),
            Some(message::DocumentMediaType::PDF),
        );
        assert_reverse_document_metadata(
            DocumentSource::Url {
                url: "https://example.com/doc.pdf".into(),
            },
            DocumentSourceKind::Url("https://example.com/doc.pdf".into()),
            None,
        );
        assert_reverse_document_metadata(
            DocumentSource::File {
                file_id: "file_abc".into(),
            },
            DocumentSourceKind::FileId("file_abc".into()),
            None,
        );
    }

    #[test]
    fn anthropic_document_metadata_survives_reverse_round_trip() {
        let provider_message = Message {
            role: Role::User,
            content: vec![Content::Document {
                source: DocumentSource::Text {
                    data: "Hello world.".into(),
                    media_type: PlainTextMediaType::Plain,
                },
                title: Some("Doc1".into()),
                context: Some("ctx".into()),
                citations: Some(CitationsConfig { enabled: true }),
                cache_control: None,
            }],
        };

        let generic: message::Message = provider_message.try_into().unwrap();
        let message::Message::User { content } = &generic else {
            panic!("expected generic user message");
        };
        let Some(message::UserContent::Document(document)) = content.first() else {
            panic!("expected generic document");
        };
        let additional_params = document
            .additional_params
            .as_ref()
            .expect("expected Anthropic document metadata");
        assert_eq!(additional_params["title"], "Doc1");
        assert_eq!(additional_params["context"], "ctx");
        assert_eq!(additional_params["citations"]["enabled"], true);

        let round_trip: Message = generic.try_into().unwrap();
        let Some(Content::Document {
            title,
            context,
            citations,
            ..
        }) = round_trip.content.first()
        else {
            panic!("expected Anthropic document");
        };
        assert_eq!(title.as_deref(), Some("Doc1"));
        assert_eq!(context.as_deref(), Some("ctx"));
        assert_eq!(citations, &Some(CitationsConfig { enabled: true }));
    }

    #[test]
    fn anthropic_document_empty_metadata_stays_none_on_reverse_conversion() {
        let provider_message = Message {
            role: Role::User,
            content: vec![Content::Document {
                source: DocumentSource::Text {
                    data: "Hello world.".into(),
                    media_type: PlainTextMediaType::Plain,
                },
                title: None,
                context: None,
                citations: None,
                cache_control: None,
            }],
        };

        let generic: message::Message = provider_message.try_into().unwrap();
        let message::Message::User { content } = &generic else {
            panic!("expected generic user message");
        };
        let Some(message::UserContent::Document(document)) = content.first() else {
            panic!("expected generic document");
        };

        assert_eq!(document.additional_params, None);
    }

    #[tokio::test]
    async fn completion_http_non_success_preserves_status_and_body() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel as _;
        use crate::providers::anthropic::Client;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"type":"error","error":{"type":"overloaded_error","message":"slow down"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::TOO_MANY_REQUESTS, body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model(CLAUDE_SONNET_4_6);
        let request = model.completion_request("hello").build();

        let error = model
            .completion(request)
            .await
            .expect_err("completion should fail with non-success status");

        // rig#2314: a provider with a request-id contract preserves its
        // non-success responses as ProviderResponse, so the transport id has
        // a home on the error; this mock sent no header, so the id is None.
        assert!(matches!(error, CompletionError::ProviderResponse(_)));
        assert_eq!(error.provider_request_id(), None);
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::TOO_MANY_REQUESTS)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn completion_2xx_error_envelope_preserves_status_and_body() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel as _;
        use crate::providers::anthropic::Client;
        use crate::test_utils::RecordingHttpClient;

        // Anthropic's `ApiResponse` is internally tagged on `type`; the `Error`
        // arm flattens `ApiErrorResponse { message }`, so a 200-OK error envelope
        // deserializes from `{"type":"error","message":"..."}` and routes through
        // `from_http_response(OK, ..)` into `ProviderResponse`.
        let body = r#"{"type":"error","message":"model overloaded"}"#;
        let http_client = RecordingHttpClient::new(body); // 200 OK
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model(CLAUDE_SONNET_4_6);
        let request = model.completion_request("hello").build();

        let error = model
            .completion(request)
            .await
            .expect_err("completion should fail with provider error envelope");

        match &error {
            CompletionError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::OK));
                assert_eq!(error.provider_response_body(), Some(body));
                assert_eq!(error.provider_response_status(), Some(http::StatusCode::OK));
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn completion_streaming_http_non_success_preserves_status_and_body() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel as _;
        use crate::providers::anthropic::Client;
        use crate::test_utils::HttpErrorStreamingClient;
        use futures::StreamExt;

        let body = r#"{"type":"error","error":{"type":"overloaded_error","message":"slow down"}}"#;
        let http_client =
            HttpErrorStreamingClient::new(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model(CLAUDE_SONNET_4_6);
        let request = model.completion_request("hello").build();

        let mut stream = model.stream(request).await.expect("stream should start");

        // The transport failure surfaces as the first error item yielded by the stream.
        let error = loop {
            match stream.next().await {
                Some(Ok(_)) => continue,
                Some(Err(error)) => break error,
                None => panic!("stream ended without yielding the transport error"),
            }
        };

        // Streaming *connect* failures stay transport-shaped (HttpError):
        // rig#2314's ProviderResponse classification covers the unary driver
        // and in-band stream envelopes, not the SSE handshake.
        assert!(matches!(error, CompletionError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));

        // The transport failure ends the stream: nothing may follow it that
        // would read as a successfully completed turn.
        assert!(stream.next().await.is_none());
        assert!(
            stream.response.is_none(),
            "a stream cut short by a transport error must not synthesize a terminal record"
        );
    }

    #[test]
    fn coerce_tool_input_normalizes_non_object_arguments() {
        use serde_json::json;

        // Object passes through untouched.
        assert_eq!(
            coerce_tool_input(json!({"q": "rust", "n": 3})),
            json!({"q": "rust", "n": 3})
        );

        // A JSON string that encodes an object is parsed into that object.
        assert_eq!(
            coerce_tool_input(json!("{\"q\":\"rust\"}")),
            json!({"q": "rust"})
        );

        // A non-JSON string, a JSON string that is not an object, null, arrays,
        // numbers and bools all collapse to an empty object: the only shape the
        // Anthropic API accepts for tool_use.input.
        assert_eq!(coerce_tool_input(json!("not json")), json!({}));
        assert_eq!(coerce_tool_input(json!("[1,2,3]")), json!({}));
        assert_eq!(coerce_tool_input(json!(null)), json!({}));
        assert_eq!(coerce_tool_input(json!([1, 2, 3])), json!({}));
        assert_eq!(coerce_tool_input(json!(42)), json!({}));
        assert_eq!(coerce_tool_input(json!(true)), json!({}));
    }

    // Regression test for issue #1429: PR #1431 added the `DocumentSource::Url`
    // wire variant and response-side parsing, but the request-side
    // `UserContent::Document` conversion still rejected URL-backed PDFs even
    // though the Anthropic Messages API supports
    // `"source": {"type": "url", ...}` for PDFs.
    // The media type is optional because Anthropic's URL source is implicitly a
    // PDF and does not include a media-type field on the wire.
    //
    // See <https://docs.anthropic.com/en/docs/build-with-claude/pdf-support>
    // for URL-sourced PDF documents.
    #[test]
    fn url_pdf_with_or_without_media_type_converts_to_url_document_source() {
        let pdf_url = "https://example.com/resume.pdf";

        for media_type in [Some(message::DocumentMediaType::PDF), None] {
            let msg = message::Message::User {
                content: vec![message::UserContent::document_url(pdf_url, media_type)],
            };

            let converted = Message::try_from(msg).expect("URL PDF should convert");
            let json = serde_json::to_value(&converted).expect("message should serialize");

            assert_eq!(
                json.pointer("/content/0/source"),
                Some(&json!({ "type": "url", "url": pdf_url })),
                "URL PDF should map to a url document source: {json:#}"
            );
        }
    }

    /// Raw-capture tests: the `normalize` shape through the Anthropic model,
    /// driven end to end over a mock transport that hands back a Messages body
    /// *and* a `request-id` response header. Anthropic's raw type carries the
    /// transport id itself (`CompletionResponse::provider_request_id`, stamped
    /// by the driver), which is why the Part A contract here is a plain
    /// `raw_completion` → `normalize`, with no id to reattach.
    /// `with_error_response_headers` with `200 OK` is the one unary double
    /// that carries response headers.
    mod raw_capture {
        use super::*;
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel as _;
        use crate::providers::anthropic::Client;
        use crate::test_utils::RecordingHttpClient;

        const REQUEST_ID: &str = "req_unit_anthropic_0001";

        /// A Messages body whose `stop_sequence` is set: the normalized
        /// response maps it to `FinishReason::Stop` and drops which sequence
        /// fired, so the capture provably answers more than `completion()`.
        const BODY: &str = r#"{
            "id": "msg_raw_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "stop_sequence",
            "stop_sequence": "alpha",
            "usage": {"input_tokens": 7, "output_tokens": 2}
        }"#;

        fn model() -> CompletionModel<RecordingHttpClient> {
            let mut headers = http::HeaderMap::new();
            headers.insert("request-id", http::HeaderValue::from_static(REQUEST_ID));
            let http_client = RecordingHttpClient::with_error_response_headers(
                http::StatusCode::OK,
                BODY,
                headers,
            );
            let client = Client::builder()
                .api_key("test-key")
                .http_client(http_client)
                .build()
                .expect("build client");
            client.completion_model(CLAUDE_SONNET_4_6)
        }

        /// The load-bearing capture property: `raw` is Anthropic's
        /// `CompletionResponse` as rig parsed it — it deserializes back into
        /// that type and re-serializes to the identical value, including the
        /// transport id the driver stamped onto the raw type — and
        /// re-normalizing that capture reproduces every normalized field, so
        /// `raw` and the typed route tell one story. Also reads
        /// `stop_sequence` off the capture, which the normalized response does
        /// not carry.
        #[tokio::test]
        async fn completion_captures_raw_that_round_trips_into_the_wire_type() {
            let model = model();

            let response = model
                .completion(model.completion_request("hello").build())
                .await
                .expect("completion");

            let raw = &response.raw;
            let typed: CompletionResponse =
                serde_json::from_value(raw.clone()).expect("raw must deserialize");
            assert_eq!(
                serde_json::to_value(&typed).expect("re-serialize"),
                *raw,
                "the capture must be exactly what the wire type serializes to"
            );
            assert_eq!(typed.stop_sequence.as_deref(), Some("alpha"));
            assert_eq!(typed.provider_request_id.as_deref(), Some(REQUEST_ID));
            assert_eq!(raw["stop_sequence"], "alpha");

            let renormalized = typed
                .normalize(<crate::providers::anthropic::client::AnthropicExt as AnthropicCompatibleProvider>::PROVIDER_NAME)
                .expect("re-normalize the capture");
            assert_eq!(response.identity(), renormalized.identity());
            assert_eq!(response.finish_reason(), renormalized.finish_reason());
            assert_eq!(response.model, renormalized.model);
            assert_eq!(response.usage, renormalized.usage);
            assert_eq!(response.choice, renormalized.choice);
            assert_eq!(
                response.finish_reason(),
                Some(completion::FinishReason::Stop)
            );
            assert_eq!(response.provider_request_id.as_deref(), Some(REQUEST_ID));
        }

        /// Part A contract statement for a provider whose raw type carries the
        /// transport id: `raw_completion` → `normalize` reproduces
        /// `completion()` on identity, finish reason, model and usage — the id
        /// included — with nothing to reattach.
        #[tokio::test]
        async fn raw_completion_then_normalize_reproduces_completion() {
            let model = model();

            let raw = model
                .raw_completion(model.completion_request("hello").build())
                .await
                .expect("typed route");
            assert_eq!(raw.provider_request_id.as_deref(), Some(REQUEST_ID));
            let reassembled = raw
                .normalize(<crate::providers::anthropic::client::AnthropicExt as AnthropicCompatibleProvider>::PROVIDER_NAME)
                .expect("normalize");

            let normalized = model
                .completion(model.completion_request("hello").build())
                .await
                .expect("normalized route");

            assert_eq!(reassembled.identity(), normalized.identity());
            assert_eq!(reassembled.finish_reason(), normalized.finish_reason());
            assert_eq!(reassembled.model, normalized.model);
            assert_eq!(reassembled.usage, normalized.usage);
            assert_eq!(reassembled.provider_request_id.as_deref(), Some(REQUEST_ID));
            assert_eq!(normalized.provider_request_id.as_deref(), Some(REQUEST_ID));
        }
    }
}
