use serde::{Deserialize, Deserializer, Serialize};

use super::client::{MistralExt, Usage};
use crate::providers::openai;
use crate::{
    completion::{self, CompletionError},
    json_utils,
};

/// The latest version of the `codestral` Mistral model
pub const CODESTRAL: &str = "codestral-latest";
/// The latest version of the `mistral-large` Mistral model
pub const MISTRAL_LARGE: &str = "mistral-large-latest";
/// The latest version of the `pixtral-large` Mistral multimodal model
///
/// **Retired.** This identifier is no longer in Mistral's `GET /v1/models`
/// catalog; requests naming it fail with `400 Invalid model`.
#[deprecated(
    note = "Mistral no longer serves this model. Pixtral is retired; use `MISTRAL_SMALL` or `MISTRAL_MEDIUM`, which are vision-capable"
)]
pub const PIXTRAL_LARGE: &str = "pixtral-large-latest";
/// The latest version of the `mistral` Mistral multimodal model, trained on datasets from the Middle East & South Asia
///
/// **Retired.** This identifier is no longer in Mistral's `GET /v1/models`
/// catalog; requests naming it fail with `400 Invalid model`.
#[deprecated(
    note = "Mistral no longer serves this model. retired; no replacement in the live catalog"
)]
pub const MISTRAL_SABA: &str = "mistral-saba-latest";
/// The latest version of the `mistral-3b` Mistral completions model
pub const MINISTRAL_3B: &str = "ministral-3b-latest";
/// The latest version of the `mistral-8b` Mistral completions model
pub const MINISTRAL_8B: &str = "ministral-8b-latest";

/// The latest version of the `mistral-small` Mistral completions model
pub const MISTRAL_SMALL: &str = "mistral-small-latest";
/// The `24-09` version of the `pixtral-small` Mistral multimodal model
///
/// **Retired.** This identifier is no longer in Mistral's `GET /v1/models`
/// catalog; requests naming it fail with `400 Invalid model`.
#[deprecated(
    note = "Mistral no longer serves this model. Pixtral is retired; use `MINISTRAL_3B`, which is vision-capable"
)]
pub const PIXTRAL_SMALL: &str = "pixtral-12b-2409";
/// The `open-mistral-nemo` model
///
/// **Retired.** This identifier is no longer in Mistral's `GET /v1/models`
/// catalog; requests naming it fail with `400 Invalid model`.
#[deprecated(
    note = "Mistral no longer serves this model. retired; no replacement in the live catalog"
)]
pub const MISTRAL_NEMO: &str = "open-mistral-nemo";
/// The `open-mistral-mamba` model
///
/// **Retired.** This identifier is no longer in Mistral's `GET /v1/models`
/// catalog; requests naming it fail with `400 Invalid model`.
#[deprecated(note = "Mistral no longer serves this model. retired; use `CODESTRAL`")]
pub const CODESTRAL_MAMBA: &str = "open-codestral-mamba";

/// Mistral completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = reqwest::Client> =
    openai::completion::GenericCompletionModel<MistralExt, H>;

/// Mistral's provider-native terminal streaming record: the value carried by
/// the final item of the stream returned by `CompletionModel::raw_stream`.
/// Shared with the OpenAI Chat Completions path but carrying Mistral's own
/// usage payload (cached-token fallbacks).
pub type MistralStreamingCompletionResponse =
    openai::StreamingCompletionResponse<super::client::Usage>;

// =================================================================
// Rig Implementation Types
// =================================================================

/// The content of an assistant message, in the two forms Mistral sends it.
///
/// An ordinary turn answers with a plain string. A reasoning
/// (magistral-class) turn answers with the chunk array Mistral's reasoning
/// docs describe — a `thinking` chunk carrying the trace, then the answer's
/// `text` chunk:
///
/// ```json
/// [{"type": "thinking", "thinking": [{"type": "text", "text": "…"}], "closed": true},
///  {"type": "text", "text": "42"}]
/// ```
///
/// Modelling that as a bare `String` kept only the answer and dropped the
/// trace — and dropped the *whole* turn when a `max_tokens` cap was spent
/// inside the trace, since the response then carries nothing else. Both
/// halves are kept here, and both are written back: a
/// [`Display`](std::fmt::Display)/[`Deref`](std::ops::Deref) to `str` keeps
/// the text reachable exactly as before.
#[derive(Debug, Default, PartialEq, Clone)]
pub struct AssistantMessageContent {
    /// The turn's visible answer text — every `text` chunk, joined in order.
    pub text: String,
    /// The turn's reasoning trace: one entry per `thinking` chunk, in order.
    /// Empty for every non-reasoning turn.
    pub reasoning: Vec<String>,
}

impl std::ops::Deref for AssistantMessageContent {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.text
    }
}

impl std::fmt::Display for AssistantMessageContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text)
    }
}

impl From<String> for AssistantMessageContent {
    fn from(text: String) -> Self {
        Self {
            text,
            reasoning: Vec::new(),
        }
    }
}

impl From<&str> for AssistantMessageContent {
    fn from(text: &str) -> Self {
        Self::from(text.to_owned())
    }
}

impl From<serde_json::Value> for AssistantMessageContent {
    fn from(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::String(text) => Self::from(text),
            serde_json::Value::Array(parts) => Self {
                text: openai::completion::joined_text_parts(&parts),
                reasoning: parts
                    .iter()
                    .map(std::slice::from_ref)
                    .map(openai::completion::joined_thinking_parts)
                    .filter(|thinking| !thinking.is_empty())
                    .collect(),
            },
            _ => Self::default(),
        }
    }
}

impl Serialize for AssistantMessageContent {
    /// Written back in the form Mistral sent it: a plain string for an
    /// ordinary turn, and the chunk array for a turn carrying a trace, so the
    /// type round-trips its own wire shape.
    ///
    /// This is not the request path — Mistral's requests are built from
    /// [`openai::completion::Message`], and the outbound trace is produced by
    /// [`splice_reasoning_into_content`]. This impl exists so a caller holding
    /// a decoded response can re-serialize it without losing half of it.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if self.reasoning.is_empty() {
            return self.text.serialize(serializer);
        }

        let mut parts: Vec<serde_json::Value> = self
            .reasoning
            .iter()
            .map(|thinking| thinking_chunk(thinking.as_str()))
            .collect();
        if !self.text.is_empty() {
            parts.push(serde_json::json!({"type": TEXT_CHUNK, TEXT_CHUNK: self.text}));
        }
        parts.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AssistantMessageContent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Option::<serde_json::Value>::deserialize(deserializer)?
            .map(Self::from)
            .unwrap_or_default())
    }
}

/// Mistral's `ThinkChunk`, carrying one complete reasoning trace.
///
/// `closed` marks a finished trace, which is what a replayed one always is —
/// Mistral streams `closed: false` only while the trace is still open.
fn thinking_chunk(trace: &str) -> serde_json::Value {
    serde_json::json!({
        "type": THINKING_CHUNK,
        THINKING_CHUNK: [{"type": TEXT_CHUNK, TEXT_CHUNK: trace}],
        "closed": true,
    })
}

/// Mistral's content-chunk tags. The API validates message content as a
/// tagged union over `text`, `image_url`, `document_url`, `reference`, `bbox`,
/// `file_url`, `input_audio`, `file`, `thinking`, `resource` and
/// `resource_link`; the shared OpenAI-compatible message conversion can
/// produce content for the five named here.
const TEXT_CHUNK: &str = "text";
const IMAGE_CHUNK: &str = "image_url";
const AUDIO_CHUNK: &str = "input_audio";
const DOCUMENT_CHUNK: &str = "document_url";
const FILE_CHUNK: &str = "file";
/// Mistral's `ThinkChunk`. Its payload is a nested content-part list rather
/// than a string, and Mistral's reasoning docs require the whole chunk be
/// replayed into the next request for the model to stay coherent.
const THINKING_CHUNK: &str = "thinking";
/// OpenAI's refusal part. Textual content, but under a key Mistral's chunk
/// schema has no field for, so it is re-tagged rather than forwarded.
const REFUSAL_TYPE: &str = "refusal";

/// The text a part carries, under either of the two keys the shared
/// OpenAI-compatible conversion can put it under.
fn part_text(part: &serde_json::Value) -> Option<&str> {
    part.get(TEXT_CHUNK)
        .and_then(serde_json::Value::as_str)
        .or_else(|| part.get(REFUSAL_TYPE).and_then(serde_json::Value::as_str))
}

/// Whether a serialized content part is purely textual, and so belongs in the
/// plain-string form rather than a chunk array.
///
/// Decided on the `type` tag first, and only on the keys for a part that
/// carries no tag. Deciding on the keys alone — as the text-only flattening
/// this replaces does — would let a part that names a chunk kind *and* happens
/// to carry a `text` key be flattened away, which is the same silent drop
/// this whole path exists to prevent.
fn is_text_part(part: &serde_json::Value) -> bool {
    match part.get("type").and_then(serde_json::Value::as_str) {
        Some(TEXT_CHUNK | REFUSAL_TYPE) => true,
        Some(_) => false,
        None => part_text(part).is_some(),
    }
}

fn unsupported_content_error(what: &str) -> CompletionError {
    crate::message::MessageError::ConversionError(format!(
        "Mistral cannot carry {what}. Mistral messages accept text, `{IMAGE_CHUNK}`, \
         `{AUDIO_CHUNK}`, `{DOCUMENT_CHUNK}`, `{FILE_CHUNK}` and `{THINKING_CHUNK}` content; \
         convert the content to one of those before sending it."
    ))
    .into()
}

/// Convert OpenAI's `{"type": "file", "file": {…}}` part into the Mistral
/// chunk carrying the same document.
///
/// Inline bytes become `document_url`, which reads the base64 `data:` URI the
/// shared conversion already built for `file_data`, and carries the filename
/// in its own optional `document_name` field. An uploaded-file reference
/// becomes Mistral's `file` chunk, which names the id at the top level rather
/// than nesting it under `file` as OpenAI does — sending OpenAI's nesting is
/// rejected twice over, for a missing `file_id` and for a forbidden extra
/// `file`, since every Mistral chunk forbids unknown fields.
fn file_part_to_mistral_chunk(
    part: &serde_json::Value,
) -> Result<serde_json::Value, CompletionError> {
    let file = part.get(FILE_CHUNK);
    let field = |name: &str| {
        file.and_then(|file| file.get(name))
            .and_then(serde_json::Value::as_str)
    };

    // Already a Mistral file chunk (`file_id` at the top level, as this
    // function emits): pass it through so finalizing an already-finalized body
    // is a no-op rather than an error about content rig itself built.
    if let Some(file_id) = part.get("file_id").and_then(serde_json::Value::as_str) {
        return Ok(serde_json::json!({"type": FILE_CHUNK, "file_id": file_id}));
    }

    if let Some(data) = field("file_data") {
        // `document_name` is Mistral's own optional filename field; it is left
        // out entirely rather than sent as null when the part has no filename.
        Ok(match field("filename") {
            Some(filename) => serde_json::json!({
                "type": DOCUMENT_CHUNK,
                DOCUMENT_CHUNK: data,
                "document_name": filename,
            }),
            None => serde_json::json!({"type": DOCUMENT_CHUNK, DOCUMENT_CHUNK: data}),
        })
    } else if let Some(file_id) = field("file_id") {
        Ok(serde_json::json!({"type": FILE_CHUNK, "file_id": file_id}))
    } else {
        Err(unsupported_content_error(
            "a file content part carrying neither `file_data` nor `file_id`",
        ))
    }
}

/// Rewrite an `input_audio` part into Mistral's canonical audio chunk, whose
/// payload is the base64 string itself.
///
/// Mistral currently also accepts the `{data, format}` object the shared
/// OpenAI-compatible conversion produces — its schema flattens the object and
/// discards `format` — but the bare string is the form its published schema
/// documents, so that is what rig sends. Nothing is lost: a deliberately wrong
/// `format` changes no result, and a `format` placed as a *sibling* of
/// `input_audio` is rejected outright.
fn audio_part_to_mistral_chunk(
    part: &serde_json::Value,
) -> Result<serde_json::Value, CompletionError> {
    let payload = part.get(AUDIO_CHUNK).ok_or_else(|| {
        unsupported_content_error("an audio content part carrying no `input_audio` payload")
    })?;

    let data = match payload {
        serde_json::Value::String(data) => data.as_str(),
        payload => payload
            .get("data")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                unsupported_content_error(
                    "an audio content part whose `input_audio` payload is not base64 data",
                )
            })?,
    };

    Ok(serde_json::json!({"type": AUDIO_CHUNK, AUDIO_CHUNK: data}))
}

/// Render one serialized content part as the Mistral chunk that carries it.
///
/// Dispatched on the `type` tag, which the shared OpenAI-compatible conversion
/// always emits, so a part naming a chunk kind is converted as that kind
/// regardless of what other keys it carries.
fn into_mistral_chunk(part: serde_json::Value) -> Result<serde_json::Value, CompletionError> {
    /// Text and refusal parts are both re-tagged as `text`: Mistral's chunk
    /// schema has no `refusal` field, and every chunk forbids unknown keys.
    fn text_chunk(part: &serde_json::Value) -> Result<serde_json::Value, CompletionError> {
        let text = part_text(part)
            .ok_or_else(|| unsupported_content_error("a text content part carrying no text"))?;
        Ok(serde_json::json!({"type": TEXT_CHUNK, TEXT_CHUNK: text}))
    }

    match part.get("type").and_then(serde_json::Value::as_str) {
        Some(TEXT_CHUNK | REFUSAL_TYPE) => text_chunk(&part),
        // The payload needs no reshaping — Mistral's image chunk takes the
        // `{url, detail}` object rig sends as readily as a bare URL string, and
        // reads a base64 `data:` URI in either, with `detail` accepting exactly
        // the `low`/`auto`/`high` range [`openai::completion::ImageDetail`]
        // serializes. It is still rebuilt rather than forwarded, because every
        // Mistral chunk forbids unknown fields: a stray sibling key riding on
        // the part would 422 the whole request.
        Some(IMAGE_CHUNK) => {
            let image = part.get(IMAGE_CHUNK).ok_or_else(|| {
                unsupported_content_error("an image content part carrying no `image_url` payload")
            })?;
            Ok(serde_json::json!({"type": IMAGE_CHUNK, IMAGE_CHUNK: image}))
        }
        Some(AUDIO_CHUNK) => audio_part_to_mistral_chunk(&part),
        Some(FILE_CHUNK) => file_part_to_mistral_chunk(&part),
        // Already a Mistral document chunk — see `file_part_to_mistral_chunk`
        // on why an already-converted part passes through.
        Some(DOCUMENT_CHUNK) => {
            let url = part.get(DOCUMENT_CHUNK).ok_or_else(|| {
                unsupported_content_error("a document content part carrying no `document_url`")
            })?;
            Ok(match part.get("document_name") {
                Some(name) => serde_json::json!({
                    "type": DOCUMENT_CHUNK, DOCUMENT_CHUNK: url, "document_name": name,
                }),
                None => serde_json::json!({"type": DOCUMENT_CHUNK, DOCUMENT_CHUNK: url}),
            })
        }
        // A reasoning trace replayed into the next request, as Mistral's
        // reasoning docs require. Rebuilt from the trace text so the chunk
        // rig emits is the one Mistral's schema names, whether it came from
        // `reasoning_content` or from an already-finalized body.
        Some(THINKING_CHUNK) => {
            let trace = match part.get(THINKING_CHUNK) {
                Some(serde_json::Value::String(text)) => text.clone(),
                Some(serde_json::Value::Array(parts)) => {
                    openai::completion::joined_text_parts(parts)
                }
                _ => String::new(),
            };
            // An empty trace is not a trace: `{"thinking": []}` is the closing
            // frame the streaming wire sends, and replaying it as a chunk with
            // an empty text part is the same "empty trace replayed as a trace"
            // that `splice_reasoning_into_content` refuses. The two paths agree.
            if trace.is_empty() {
                return Err(unsupported_content_error(
                    "a thinking content part carrying no reasoning trace",
                ));
            }
            Ok(thinking_chunk(&trace))
        }
        Some(kind) => Err(unsupported_content_error(&format!(
            "`{kind}` message content"
        ))),
        // Untagged, but textual: the shared flattening would have taken it, so
        // it converts rather than failing.
        None if part_text(&part).is_some() => text_chunk(&part),
        None => Err(unsupported_content_error("untyped message content")),
    }
}

/// Rewrite one serialized message `content` value into Mistral's message
/// content schema.
///
/// Mistral accepts content as either a plain string or an array of typed
/// chunks. Text-only content keeps the plain-string form it has always taken.
/// Content carrying anything else keeps the array, with each part rendered the
/// way Mistral's schema names it, instead of being flattened away: the
/// text-only flattening this replaces kept only parts with a `text`/`refusal`
/// key, so an attached image, document or audio clip was dropped from the
/// request and the caller got an ordinary completion answering a prompt it
/// never sent (#2290).
///
/// Content Mistral has no chunk for — video, and any part type a future
/// conversion adds — fails here rather than being silently removed. The one
/// exception is content whose parts are *all* tagged `text`/`refusal`: that
/// takes the flattening path, which drops a part carrying no string payload
/// exactly as it always has, rather than inventing a new failure for a shape
/// rig's own conversion cannot produce.
pub(super) fn normalize_request_content(
    content: &mut serde_json::Value,
) -> Result<(), CompletionError> {
    let Some(parts) = content.as_array() else {
        return Ok(());
    };

    if parts.iter().all(is_text_part) {
        // Flattened unconditionally rather than under `only_if_all_text`, so
        // the helper does not re-decide: it judges per key while the guard
        // above judges on the type tag, and the two disagree for a malformed
        // part such as `{"type": "text"}` carrying no `text`. Letting the
        // helper decline would leave that content as an array of chunks
        // Mistral cannot read; flattening it reproduces what rig sent before.
        openai::completion::flatten_text_content_parts(content, "", false);
        return Ok(());
    }

    // Re-borrowed rather than held across the branch above, which needs
    // `content` itself. The array-ness was just established, so the `else` is
    // unreachable — expressed as a no-op instead of an unwrap.
    if let Some(parts) = content.as_array_mut() {
        for part in parts {
            *part = into_mistral_chunk(part.take())?;
        }
    }

    Ok(())
}

/// Move an assistant message's `reasoning_content` into its `content` as
/// Mistral's `thinking` chunk.
///
/// The shared OpenAI-compatible conversion renders a
/// [`Reasoning`](crate::completion::AssistantContent::Reasoning) block as a
/// `reasoning_content` *string* beside the content — the DeepSeek/llama.cpp
/// dialect. Mistral has no such field (it rejects unknown assistant keys) and
/// instead takes the trace as a content chunk, which its reasoning docs
/// require be replayed into the following request. Stripping the field, as
/// this used to, sent the turn back without its trace — and left rig unable to
/// round-trip a reasoning block it had itself produced.
///
/// The chunk leads the content, matching where Mistral puts it in the
/// response it came from. Empty or non-string reasoning is dropped rather than
/// replayed as an empty trace.
///
/// Note that Mistral answers a reasoning input sent to a model without the
/// capability with `400 Reasoning input is not enabled for this model` — a
/// loud, provider-authored rejection, where the alternative is the silent
/// incoherence a dropped trace causes on the models that do support it.
pub(super) fn splice_reasoning_into_content(
    message: &mut serde_json::Map<String, serde_json::Value>,
) {
    let Some(reasoning) = message.remove("reasoning_content") else {
        return;
    };
    let Some(reasoning) = reasoning.as_str().filter(|trace| !trace.is_empty()) else {
        return;
    };
    let chunk = thinking_chunk(reasoning);

    match message.get_mut("content") {
        Some(serde_json::Value::Array(parts)) => {
            parts.insert(0, chunk);
            return;
        }
        Some(content @ serde_json::Value::String(_)) => {
            let text = std::mem::take(content);
            *content = match text.as_str() {
                Some("") | None => serde_json::json!([chunk]),
                Some(text) => {
                    serde_json::json!([chunk, {"type": TEXT_CHUNK, TEXT_CHUNK: text}])
                }
            };
            return;
        }
        // Absent or null — a turn that was pure tool-call scaffolding, whose
        // trace is now the only content it has.
        _ => {}
    }

    message.insert("content".to_string(), serde_json::json!([chunk]));
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

/// Mistral's provider-native message shape, as it appears in responses.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    User {
        content: String,
    },
    Assistant {
        /// The turn's content. Not a plain string: a reasoning turn answers
        /// with a chunk array carrying the trace beside the answer, which
        /// [`AssistantMessageContent`] keeps both halves of.
        #[serde(default)]
        content: AssistantMessageContent,
        #[serde(
            default,
            deserialize_with = "json_utils::null_or_default",
            skip_serializing_if = "Vec::is_empty"
        )]
        tool_calls: Vec<ToolCall>,
        #[serde(default)]
        prefix: bool,
    },
    System {
        content: String,
    },
    Tool {
        /// The name of the tool that was called
        #[serde(skip_serializing_if = "String::is_empty")]
        name: String,
        /// The content of the tool call
        content: String,
        /// The id of the tool call
        tool_call_id: String,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(default)]
    pub r#type: ToolType,
    pub function: Function,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    #[serde(
        serialize_with = "json_utils::stringified_json::serialize",
        deserialize_with = "deserialize_truncatable_arguments"
    )]
    pub arguments: serde_json::Value,
}

/// Deserialize a tool call's `arguments`, tolerating the truncated JSON a
/// `max_tokens`-capped turn produces.
///
/// Mistral emits the tool call anyway when the budget runs out mid-arguments:
/// a live turn capped at 32 tokens returns `finish_reason: "length"` with
/// `arguments` cut off partway through the object. Parsing that strictly fails
/// the *whole* response — the text, usage, id and finish reason go with it —
/// where the streaming path keeps the turn and drops the unusable call
/// ([`UnparseableToolInput::Drop`](crate::streaming::UnparseableToolInput)).
///
/// Truncated arguments become `null`, which
/// [`NormalizeCompletionResponse`] drops the call on, so the two transports
/// agree: the turn survives and its `Length` finish reason is what tells the
/// caller the call was cut off.
fn deserialize_truncatable_arguments<'de, D>(deserializer: D) -> Result<serde_json::Value, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = String::deserialize(deserializer)?;
    if raw.trim().is_empty() {
        return Ok(serde_json::Value::Object(serde_json::Map::new()));
    }
    Ok(serde_json::from_str(&raw).unwrap_or(serde_json::Value::Null))
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

impl crate::telemetry::ProviderResponseExt for CompletionResponse {
    type Usage = Usage;

    fn get_response_id(&self) -> Option<String> {
        Some(self.id.clone())
    }

    fn get_response_model_name(&self) -> Option<String> {
        Some(self.model.clone())
    }

    fn get_text_response(&self) -> Option<String> {
        let res = self
            .choices
            .iter()
            .filter_map(|choice| match choice.message {
                // The turn's *visible* text: telemetry records the answer,
                // never the hidden trace beside it.
                Message::Assistant { ref content, .. } => {
                    if content.text.is_empty() {
                        None
                    } else {
                        Some(content.text.clone())
                    }
                }
                _ => None,
            })
            .collect::<Vec<String>>()
            .join("\n");

        if res.is_empty() { None } else { Some(res) }
    }

    fn get_usage(&self) -> Option<Self::Usage> {
        self.usage.clone()
    }
}

/// Normalize a Mistral chat completion response.
///
/// The provider descriptor name is an *input* rather than a constant so the
/// shared OpenAI-compatible completion path labels the response with the
/// descriptor that actually produced it.
impl crate::completion::NormalizeCompletionResponse for CompletionResponse {
    fn normalize(self, provider: &str) -> Result<completion::CompletionResponse, CompletionError> {
        use crate::providers::internal::openai_chat_completions_compatible as compat;

        let usage = self
            .usage
            .as_ref()
            .map(completion::Usage::from)
            .unwrap_or_default();
        compat::normalize_openai_response(
            provider,
            &self.choices,
            Some(self.id.as_str()),
            Some(self.model.as_str()),
            usage,
            |choice| choice.finish_reason.as_str(),
            |choice| match &choice.message {
                Message::Assistant {
                    content,
                    tool_calls,
                    ..
                } => {
                    let mut blocks = compat::text_then_tool_calls(
                        &content.text,
                        content.text.is_empty(),
                        // A call whose arguments were truncated mid-JSON is
                        // unusable; the turn's `Length` finish reason is what
                        // reports it, exactly as on the streaming path.
                        tool_calls
                            .iter()
                            .filter(|call| !call.function.arguments.is_null())
                            .map(|call| {
                                (
                                    call.id.as_str(),
                                    call.function.name.as_str(),
                                    call.function.arguments.clone(),
                                )
                            }),
                    );
                    // The trace leads, as it does on the wire and as the
                    // streamed twin emits it: Mistral puts its `thinking`
                    // chunk ahead of the answer's `text` chunk.
                    for (index, trace) in content.reasoning.iter().enumerate() {
                        blocks.insert(index, completion::AssistantContent::reasoning(trace));
                    }
                    Some(blocks)
                }
                _ => None,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::NormalizeCompletionResponse as _;
    use crate::providers::openai::completion::OpenAICompatibleProvider;

    #[test]
    fn deserializes_response_with_array_and_null_content() {
        let data = r#"{
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "mistral-small-latest",
            "system_fingerprint": null,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}]
                    },
                    "logprobs": null,
                    "finish_reason": "stop"
                },
                {
                    "index": 1,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "add", "arguments": "{\"x\":1,\"y\":2}"}
                        }]
                    },
                    "logprobs": null,
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        }"#;

        let response: CompletionResponse =
            serde_json::from_str(data).expect("response should deserialize");
        match &response.choices[0].message {
            Message::Assistant { content, .. } => {
                assert_eq!(content.text, "Hello world");
                assert!(content.reasoning.is_empty());
            }
            _ => panic!("expected assistant message"),
        }
        match &response.choices[1].message {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                assert_eq!(content.text, "");
                assert_eq!(tool_calls[0].function.name, "add");
            }
            _ => panic!("expected assistant message"),
        }
    }

    #[test]
    fn usage_prefers_structured_cached_tokens_and_falls_back() {
        let structured: Usage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "num_cached_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 7}
        }))
        .expect("usage should deserialize");
        assert_eq!(structured.cached_tokens(), 7);

        let fallback: Usage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "num_cached_tokens": 2
        }))
        .expect("usage should deserialize");
        assert_eq!(fallback.cached_tokens(), 2);

        // The singular alias form used by some Mistral responses.
        let aliased: Usage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_token_details": {"cached_tokens": 4}
        }))
        .expect("usage should deserialize");
        assert_eq!(aliased.cached_tokens(), 4);
    }

    /// Mistral reports audio outside `prompt_tokens`, so counting only that
    /// field leaves `input + output` short of `total` by the audio payload.
    /// The numbers are a live Voxtral turn's, quoted verbatim.
    #[test]
    fn usage_counts_audio_tokens_as_input() {
        let usage: Usage = serde_json::from_value(serde_json::json!({
            "prompt_audio_seconds": 0,
            "prompt_tokens": 6,
            "completion_tokens": 2,
            "total_tokens": 383,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 375}
        }))
        .expect("usage should deserialize");

        assert_eq!(usage.audio_tokens(), 375);
        assert_eq!(usage.input_tokens(), 381);

        let normalized = crate::completion::Usage::from(&usage);
        assert_eq!(normalized.input_tokens, 381);
        assert_eq!(normalized.output_tokens, 2);
        assert_eq!(
            normalized.input_tokens + normalized.output_tokens,
            normalized.total_tokens,
            "the parts must add up to the total Mistral reported"
        );
    }

    /// A text turn carries no audio detail, and must be unaffected.
    #[test]
    fn usage_without_audio_is_unchanged() {
        let usage: Usage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 19, "completion_tokens": 2, "total_tokens": 21,
            "prompt_tokens_details": {"cached_tokens": 0}
        }))
        .expect("usage should deserialize");

        assert_eq!(usage.audio_tokens(), 0);
        assert_eq!(crate::completion::Usage::from(&usage).input_tokens, 19);
    }

    /// Mistral emits the tool call anyway when `max_tokens` runs out mid
    /// arguments — a live turn capped at 32 tokens returned
    /// `finish_reason: "length"` with `arguments` cut off partway through the
    /// object. Parsing strictly took the whole response down with it.
    #[test]
    fn truncated_tool_arguments_do_not_destroy_the_response() {
        let data = r#"{
            "id": "cmpl-1", "object": "chat.completion", "created": 1,
            "model": "mistral-small-latest", "system_fingerprint": null,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Recording that now.",
                    "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "record", "arguments": "{\"note\": \"How to bake sour"}
                    }]
                },
                "logprobs": null,
                "finish_reason": "length"
            }],
            "usage": {"prompt_tokens": 30, "completion_tokens": 32, "total_tokens": 62}
        }"#;

        let response: CompletionResponse =
            serde_json::from_str(data).expect("a truncated tool call must not fail the response");

        let normalized = response
            .normalize("mistral")
            .expect("the turn must survive with its text and metadata");
        assert_eq!(
            normalized.finish_reason(),
            Some(crate::completion::FinishReason::Length),
            "the finish reason is what reports the truncation"
        );
        assert_eq!(normalized.usage.total_tokens, 62);
        // The unusable call is dropped, as the streaming path drops it.
        assert!(
            normalized.choice.iter().all(|content| !matches!(
                content,
                crate::completion::AssistantContent::ToolCall(_)
            )),
            "a call with truncated arguments must not be handed to a tool"
        );
    }

    /// A complete tool call is unaffected by the tolerant parse.
    #[test]
    fn complete_tool_arguments_still_parse() {
        let data = r#"{
            "id": "cmpl-1", "object": "chat.completion", "created": 1,
            "model": "mistral-small-latest", "system_fingerprint": null,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "add", "arguments": "{\"x\":1,\"y\":2}"}
                }]},
                "logprobs": null, "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        }"#;

        let normalized = serde_json::from_str::<CompletionResponse>(data)
            .expect("response should deserialize")
            .normalize("mistral")
            .expect("a complete call should normalize");
        assert!(
            normalized
                .choice
                .iter()
                .any(|content| matches!(content, crate::completion::AssistantContent::ToolCall(_))),
            "a complete call must still reach the caller"
        );
    }

    /// Mistral rejects a forced tool choice beside a response format with
    /// "`json_schema` response type with tools is only compatible with
    /// `tool_choice: auto`". Rig reaches that combination by itself on the
    /// turn after a tool result, so finalization relaxes the choice.
    #[test]
    fn finalize_relaxes_a_forced_tool_choice_beside_a_response_format() {
        let mut body = serde_json::json!({
            "model": MISTRAL_SMALL,
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "required",
            "tools": [{"type": "function", "function": {"name": "add", "parameters": {}}}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "Plan"}}
        });
        MistralExt
            .finalize_request_body(&mut body)
            .expect("finalize should succeed");
        assert_eq!(body["tool_choice"], "auto");
        assert!(
            body.get("response_format").is_some(),
            "the caller's schema must survive; relaxing the choice is what gives way"
        );

        // A specific function is forcing too, and equally rejected.
        let mut body = serde_json::json!({
            "model": MISTRAL_SMALL,
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "function", "function": {"name": "add"}},
            "tools": [{"type": "function", "function": {"name": "add", "parameters": {}}}],
            "response_format": {"type": "json_object"}
        });
        MistralExt
            .finalize_request_body(&mut body)
            .expect("finalize should succeed");
        assert_eq!(body["tool_choice"], "auto");
    }

    /// The relaxation is narrow: without a response format, or without tools,
    /// or when the choice is already compatible, nothing moves.
    /// A `text` response format is not the constrained kind either — Mistral
    /// takes it beside a forced choice, verified live.
    #[test]
    fn finalize_leaves_a_forced_tool_choice_alone_without_a_response_format() {
        let mut body = serde_json::json!({
            "model": MISTRAL_SMALL,
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "required",
            "tools": [{"type": "function", "function": {"name": "add", "parameters": {}}}]
        });
        MistralExt
            .finalize_request_body(&mut body)
            .expect("finalize should succeed");
        assert_eq!(body["tool_choice"], "any", "still just the dialect rename");

        let mut body = serde_json::json!({
            "model": MISTRAL_SMALL,
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "none",
            "tools": [{"type": "function", "function": {"name": "add", "parameters": {}}}],
            "response_format": {"type": "json_object"}
        });
        MistralExt
            .finalize_request_body(&mut body)
            .expect("finalize should succeed");
        assert_eq!(body["tool_choice"], "none", "`none` is already compatible");

        let mut body = serde_json::json!({
            "model": MISTRAL_SMALL,
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "required",
            "tools": [{"type": "function", "function": {"name": "add", "parameters": {}}}],
            "response_format": {"type": "text"}
        });
        MistralExt
            .finalize_request_body(&mut body)
            .expect("finalize should succeed");
        assert_eq!(
            body["tool_choice"], "any",
            "a `text` response format is unconstrained; only the structured kinds conflict"
        );
    }

    #[test]
    fn finalize_rewrites_required_tool_choice_to_any() {
        let mut body = serde_json::json!({
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "required"
        });

        MistralExt
            .finalize_request_body(&mut body)
            .expect("finalize should succeed");

        assert_eq!(body["tool_choice"], "any");
    }

    #[test]
    fn finalize_preserves_specific_function_tool_choice() {
        let mut body = serde_json::json!({
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "function", "function": {"name": "beta"}}
        });

        MistralExt
            .finalize_request_body(&mut body)
            .expect("finalize should succeed");

        assert_eq!(
            body["tool_choice"],
            serde_json::json!({"type": "function", "function": {"name": "beta"}})
        );
    }

    #[test]
    fn finalize_flattens_assistant_history_and_adds_prefix() {
        let mut body = serde_json::json!({
            "model": "mistral-small-latest",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello."}],
                    "reasoning_content": "hidden thoughts"
                },
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "add", "arguments": "{}"}
                    }]
                },
                {"role": "assistant", "content": [{"type": "text", "text": "Traceless."}]}
            ]
        });

        MistralExt
            .finalize_request_body(&mut body)
            .expect("finalize should succeed");

        assert_eq!(body["messages"][0]["content"], "Be brief.");
        // The trace does not vanish: `reasoning_content` is the DeepSeek
        // dialect the shared conversion emits, and Mistral's own spelling for
        // it is a `thinking` chunk inside the content — which its reasoning
        // docs require be replayed. See `reasoning_tests` for the whole rule.
        assert_eq!(
            body["messages"][2]["content"],
            serde_json::json!([
                {"type": "thinking",
                 "thinking": [{"type": "text", "text": "hidden thoughts"}],
                 "closed": true},
                {"type": "text", "text": "Hello."}
            ])
        );
        assert_eq!(body["messages"][2]["prefix"], false);
        assert!(
            body["messages"][2].get("reasoning_content").is_none(),
            "Mistral rejects unknown assistant fields; the trace rides in `content`"
        );
        assert_eq!(body["messages"][3]["content"], "");
        assert_eq!(body["messages"][3]["prefix"], false);
        assert_eq!(
            body["messages"][4]["content"], "Traceless.",
            "an assistant turn with no trace still flattens to Mistral's plain string"
        );
    }

    /// Finalize a one-user-message body and return the message's `content`.
    ///
    /// These cells are unit tests rather than cassettes because the behaviour
    /// under test is that **no request is built** — there is no traffic to
    /// record for content that is rejected before the wire. The cells covering
    /// content that *is* sent are recorded, in
    /// `tests/providers/mistral/multimodal_content.rs`.
    fn finalized_content(parts: serde_json::Value) -> Result<serde_json::Value, CompletionError> {
        let mut body = serde_json::json!({
            "model": MISTRAL_SMALL,
            "messages": [{"role": "user", "content": parts}],
        });
        MistralExt.finalize_request_body(&mut body)?;
        Ok(body["messages"][0]["content"].clone())
    }

    /// Video has no Mistral chunk — the API's own content discriminator does
    /// not list one — so it must fail rather than be flattened away.
    #[test]
    fn finalize_rejects_video_content() {
        let error = finalized_content(serde_json::json!([
            {"type": "text", "text": "Describe this."},
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}
        ]))
        .expect_err("video content must not be dropped from the request");

        assert!(matches!(error, CompletionError::RequestError(_)));
        let rendered = error.to_string();
        assert!(rendered.contains("video_url"), "{rendered}");
    }

    /// An unrecognized part type fails closed, so a content kind added to the
    /// shared conversion later cannot start disappearing silently.
    #[test]
    fn finalize_rejects_unrecognized_and_untyped_parts() {
        let error = finalized_content(serde_json::json!([
            {"type": "text", "text": "hi"},
            {"type": "some_future_part", "some_future_part": {}}
        ]))
        .expect_err("an unmodelled part must not be dropped");
        assert!(matches!(error, CompletionError::RequestError(_)));

        let error = finalized_content(serde_json::json!([
            {"type": "text", "text": "hi"},
            {"payload": "no type tag at all"}
        ]))
        .expect_err("an untyped part must not be dropped");
        assert!(error.to_string().contains("untyped"), "{error}");
    }

    /// OpenAI's file part must carry something convertible; a part with
    /// neither inline bytes nor an id names no document at all.
    #[test]
    fn finalize_rejects_a_file_part_with_no_payload() {
        let error = finalized_content(serde_json::json!([
            {"type": "text", "text": "hi"},
            {"type": "file", "file": {"filename": "empty.pdf"}}
        ]))
        .expect_err("a file part naming no document must not be dropped");
        assert!(matches!(error, CompletionError::RequestError(_)));
    }

    /// An audio part whose payload is neither a base64 string nor an object
    /// carrying one cannot be rendered as Mistral's audio chunk.
    #[test]
    fn finalize_rejects_an_audio_part_with_no_payload() {
        let error = finalized_content(serde_json::json!([
            {"type": "text", "text": "hi"},
            {"type": "input_audio", "input_audio": {"format": "mp3"}}
        ]))
        .expect_err("an audio part carrying no data must not be dropped");
        assert!(matches!(error, CompletionError::RequestError(_)));
    }

    /// The document conversions, pinned as exact wire shapes.
    ///
    /// The `document_url` half is also proven live, by the recorded cells that
    /// read `BANANA-7391` back out of an attached PDF. The `file`/`file_id`
    /// half is pinned by shape only: exercising it end to end would mean
    /// uploading a file and committing its account-scoped, expiring id to a
    /// fixture, so this cell is the whole of its coverage.
    #[test]
    fn finalize_maps_openai_file_parts_onto_mistral_chunks() {
        let content = finalized_content(serde_json::json!([
            {"type": "text", "text": "Read these."},
            {"type": "file", "file": {
                "file_data": "data:application/pdf;base64,JVBERi0xLjQK",
                "filename": "document.pdf"
            }},
            {"type": "file", "file": {"file_id": "00000000-0000-0000-0000-000000000000"}}
        ]))
        .expect("file parts should convert");

        assert_eq!(
            content,
            serde_json::json!([
                {"type": "text", "text": "Read these."},
                {
                    "type": "document_url",
                    "document_url": "data:application/pdf;base64,JVBERi0xLjQK",
                    "document_name": "document.pdf"
                },
                // Mistral's file chunk names the id at the top level; OpenAI's
                // nesting under `file` is rejected as an extra field.
                {"type": "file", "file_id": "00000000-0000-0000-0000-000000000000"}
            ])
        );
    }

    /// Audio collapses to Mistral's documented bare-string payload, and the
    /// image chunk forwards unchanged because Mistral accepts rig's object.
    #[test]
    fn finalize_maps_audio_and_image_parts_onto_mistral_chunks() {
        let content = finalized_content(serde_json::json!([
            {"type": "input_audio", "input_audio": {"data": "SUQzBAA=", "format": "mp3"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "auto"}}
        ]))
        .expect("audio and image parts should convert");

        assert_eq!(
            content,
            serde_json::json!([
                {"type": "input_audio", "input_audio": "SUQzBAA="},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "auto"}}
            ])
        );
    }

    /// A refusal travelling beside a chunk becomes a text chunk: Mistral's
    /// schema has no `refusal` field, and every chunk forbids unknown keys.
    #[test]
    fn finalize_retags_a_refusal_beside_a_chunk_as_text() {
        let content = finalized_content(serde_json::json!([
            {"type": "refusal", "refusal": "I cannot help with that."},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}
        ]))
        .expect("a refusal beside a chunk should convert");

        assert_eq!(
            content[0],
            serde_json::json!({"type": "text", "text": "I cannot help with that."})
        );
    }

    /// Text-only content keeps the plain-string form every existing Mistral
    /// fixture pins — including a refusal-only message, which the shared
    /// flattening has always treated as text.
    #[test]
    fn finalize_still_flattens_text_only_content() {
        assert_eq!(
            finalized_content(serde_json::json!([
                {"type": "text", "text": "First."},
                {"type": "text", "text": "Second."}
            ]))
            .expect("text-only content should flatten"),
            serde_json::json!("First.Second.")
        );

        assert_eq!(
            finalized_content(serde_json::json!([
                {"type": "text", "text": "Partly: "},
                {"type": "refusal", "refusal": "I cannot help with that."}
            ]))
            .expect("refusal content should flatten"),
            serde_json::json!("Partly: I cannot help with that.")
        );

        // Content that is already a plain string is left exactly as-is.
        assert_eq!(
            finalized_content(serde_json::json!("already a string"))
                .expect("string content should pass through"),
            serde_json::json!("already a string")
        );

        // An empty array still collapses to the empty string it always did.
        assert_eq!(
            finalized_content(serde_json::json!([])).expect("empty content should flatten"),
            serde_json::json!("")
        );
    }

    /// Textuality is decided on the `type` tag, not on the presence of a
    /// `text` key. A part that names a chunk kind is that kind even if it also
    /// carries text — deciding on the key alone would flatten the chunk away,
    /// which is the silent drop this path exists to prevent.
    ///
    /// The stray key is *dropped*, not forwarded: every Mistral chunk forbids
    /// unknown fields, so carrying it through would 422 the whole request and
    /// lose the image just as surely.
    #[test]
    fn finalize_renders_a_chunk_that_also_carries_text_as_its_own_kind() {
        let content = finalized_content(serde_json::json!([
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}, "text": "cat"}
        ]))
        .expect("a tagged image part should convert");

        assert_eq!(
            content,
            serde_json::json!([
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}
            ]),
            "the image must reach the wire, in a chunk carrying only the fields Mistral names"
        );
    }

    /// Finalizing an already-finalized body is a no-op. `finalize_request_body`
    /// is a public trait method, so a caller can reach it twice; the chunks
    /// this code emits must not read as content Mistral cannot carry.
    #[test]
    fn finalize_is_idempotent_over_the_chunks_it_emits() {
        let parts = serde_json::json!([
            {"type": "text", "text": "Read these."},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            {"type": "input_audio", "input_audio": "SUQzBAA="},
            {"type": "document_url", "document_url": "data:application/pdf;base64,JVBERi0xLjQK",
             "document_name": "document.pdf"},
            {"type": "file", "file_id": "00000000-0000-0000-0000-000000000000"}
        ]);

        let once = finalized_content(parts).expect("emitted chunks should convert");
        let twice = finalized_content(once.clone()).expect("a second pass should be a no-op");

        assert_eq!(once, twice);
    }

    /// An image part with no payload names no image at all.
    #[test]
    fn finalize_rejects_an_image_part_with_no_payload() {
        let error = finalized_content(serde_json::json!([
            {"type": "text", "text": "hi"},
            {"type": "image_url"}
        ]))
        .expect_err("an image part carrying no payload must not be dropped");
        assert!(matches!(error, CompletionError::RequestError(_)));
    }
}

/// Unit cells for the reasoning (magistral-class) content path.
///
/// Recorded coverage lives in `tests/providers/mistral/reasoning_content.rs`;
/// these pin the pieces a recording cannot isolate — the wire shapes Mistral's
/// schema allows but its models have not been observed to send, and the
/// request-side splice, whose output the recorded cells assert only in
/// aggregate.
#[cfg(test)]
mod reasoning_tests {
    use super::*;
    use crate::completion::{AssistantContent, NormalizeCompletionResponse as _};
    use crate::providers::openai::completion::OpenAICompatibleProvider;

    /// The exact assistant content a live `reasoning_effort: "high"` turn
    /// returns, quoted from a recorded response.
    fn reasoning_content() -> serde_json::Value {
        serde_json::json!([
            {"type": "thinking",
             "thinking": [{"type": "text", "text": "6 multiplied by 7 equals 42."}],
             "closed": true},
            {"type": "text", "text": "42"}
        ])
    }

    fn assistant_content(value: serde_json::Value) -> AssistantMessageContent {
        serde_json::from_value(value).expect("assistant content should deserialize")
    }

    /// The trace and the answer come out of the same `content` field, and both
    /// survive. Joining only the text parts — what the plain-`String` model
    /// did — kept the answer and dropped the trace.
    #[test]
    fn reasoning_content_keeps_the_trace_beside_the_answer() {
        let content = assistant_content(reasoning_content());
        assert_eq!(content.text, "42");
        assert_eq!(content.reasoning, vec!["6 multiplied by 7 equals 42."]);
    }

    /// An ordinary turn's plain string is unchanged, and declares no trace.
    #[test]
    fn plain_string_content_is_unchanged() {
        let content = assistant_content(serde_json::json!("42"));
        assert_eq!(content.text, "42");
        assert!(content.reasoning.is_empty());
    }

    /// A cap spent entirely inside the trace returns a content array with the
    /// thinking chunk and nothing else. Under the plain-`String` model the
    /// whole turn came back empty; the trace is the only thing it produced.
    #[test]
    fn a_thinking_only_turn_is_not_empty() {
        let content = assistant_content(serde_json::json!([
            {"type": "thinking", "thinking": [{"type": "text", "text": "Let me work it out"}]}
        ]));
        assert_eq!(content.text, "");
        assert_eq!(
            content.reasoning,
            vec!["Let me work it out"],
            "the trace is the only thing this turn produced"
        );
    }

    /// Absent, null and non-string/array content all read as nothing, exactly
    /// as they did before.
    #[test]
    fn absent_and_unusable_content_reads_as_empty() {
        for value in [
            serde_json::Value::Null,
            serde_json::json!(7),
            serde_json::json!([]),
        ] {
            let content = assistant_content(value.clone());
            assert_eq!(content, AssistantMessageContent::default(), "{value}");
        }
    }

    /// Several traces in one turn stay separate blocks, in wire order, and a
    /// chunk whose payload is a bare string is read too — Mistral's schema
    /// permits it and a replayed chunk can carry one.
    #[test]
    fn multiple_traces_stay_separate_and_ordered() {
        let content = assistant_content(serde_json::json!([
            {"type": "thinking", "thinking": [{"type": "text", "text": "first"}]},
            {"type": "text", "text": "answer "},
            {"type": "thinking", "thinking": "second"},
            {"type": "text", "text": "here"}
        ]));
        assert_eq!(content.text, "answer here");
        assert_eq!(content.reasoning, vec!["first", "second"]);
    }

    /// A thinking chunk with no payload contributes no trace rather than an
    /// empty one — an empty reasoning block is noise, not data.
    #[test]
    fn a_payload_less_thinking_chunk_contributes_nothing() {
        let content = assistant_content(serde_json::json!([
            {"type": "thinking", "closed": true},
            {"type": "thinking", "thinking": []},
            {"type": "text", "text": "42"}
        ]));
        assert_eq!(content.text, "42");
        assert!(content.reasoning.is_empty());
    }

    /// Serialization writes back the shape Mistral reads: a plain string when
    /// there is no trace, and the chunk array — trace first — when there is.
    #[test]
    fn content_serializes_back_into_mistrals_own_shape() {
        let plain = AssistantMessageContent::from("42");
        assert_eq!(
            serde_json::to_value(&plain).unwrap(),
            serde_json::json!("42")
        );

        let reasoned = assistant_content(reasoning_content());
        assert_eq!(
            serde_json::to_value(&reasoned).unwrap(),
            serde_json::json!([
                {"type": "thinking",
                 "thinking": [{"type": "text", "text": "6 multiplied by 7 equals 42."}],
                 "closed": true},
                {"type": "text", "text": "42"}
            ])
        );

        let trace_only = assistant_content(serde_json::json!([
            {"type": "thinking", "thinking": [{"type": "text", "text": "still working"}]}
        ]));
        assert_eq!(
            serde_json::to_value(&trace_only).unwrap(),
            serde_json::json!([
                {"type": "thinking",
                 "thinking": [{"type": "text", "text": "still working"}],
                 "closed": true}
            ]),
            "a turn with no answer must not gain an empty text chunk"
        );
    }

    fn normalized(message: serde_json::Value, finish_reason: &str) -> Vec<AssistantContent> {
        let response: CompletionResponse = serde_json::from_value(serde_json::json!({
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": MISTRAL_SMALL,
            "system_fingerprint": null,
            "choices": [{
                "index": 0,
                "message": message,
                "logprobs": null,
                "finish_reason": finish_reason
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        }))
        .expect("response should deserialize");
        response
            .normalize("mistral")
            .expect("response should normalize")
            .choice
    }

    /// The trace reaches the caller as a reasoning block, ahead of the answer
    /// — the order Mistral sends it in, and the order the streamed twin emits.
    #[test]
    fn normalize_emits_the_trace_before_the_answer() {
        let choice = normalized(
            serde_json::json!({"role": "assistant", "content": reasoning_content()}),
            "stop",
        );
        assert!(
            matches!(&choice[0], AssistantContent::Reasoning(_)),
            "the trace leads: {choice:?}"
        );
        assert!(matches!(&choice[1], AssistantContent::Text(text) if text.text == "42"));
        assert_eq!(choice.len(), 2);
    }

    /// A reasoning turn that ends in a tool call carries the trace beside the
    /// call, which is the shape a reasoning agent sees on every turn.
    #[test]
    fn normalize_keeps_the_trace_beside_a_tool_call() {
        let choice = normalized(
            serde_json::json!({
                "role": "assistant",
                "content": [{"type": "thinking",
                             "thinking": [{"type": "text", "text": "I need the add tool."}],
                             "closed": true}],
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "add", "arguments": "{\"a\":2,\"b\":3}"}
                }]
            }),
            "tool_calls",
        );
        assert!(matches!(&choice[0], AssistantContent::Reasoning(_)));
        assert!(
            matches!(&choice[1], AssistantContent::ToolCall(call) if call.function.name == "add")
        );
        assert_eq!(choice.len(), 2);
    }

    /// A cap spent inside the trace used to normalize to a turn with **no**
    /// content at all — the `Length` finish reason was the only thing left.
    /// The trace it produced is now what the turn carries.
    #[test]
    fn normalize_keeps_a_truncated_turns_trace() {
        let choice = normalized(
            serde_json::json!({
                "role": "assistant",
                "content": [{"type": "thinking",
                             "thinking": [{"type": "text", "text": "Okay, 17 times 23 is"}]}]
            }),
            "length",
        );
        assert_eq!(choice.len(), 1);
        assert!(matches!(&choice[0], AssistantContent::Reasoning(_)));
    }

    /// A non-reasoning turn normalizes exactly as it always has.
    #[test]
    fn normalize_leaves_an_ordinary_turn_alone() {
        let choice = normalized(
            serde_json::json!({"role": "assistant", "content": "42"}),
            "stop",
        );
        assert_eq!(choice.len(), 1);
        assert!(matches!(&choice[0], AssistantContent::Text(text) if text.text == "42"));
    }

    /// The request-side splice: `reasoning_content` — the spelling the shared
    /// OpenAI-compatible conversion emits for a replayed reasoning block — is
    /// moved into `content` as Mistral's thinking chunk, ahead of the text.
    fn finalized_message(message: serde_json::Value) -> serde_json::Value {
        let mut body = serde_json::json!({"model": MISTRAL_SMALL, "messages": [message]});
        MistralExt
            .finalize_request_body(&mut body)
            .expect("body should finalize");
        body["messages"][0].clone()
    }

    #[test]
    fn finalize_replays_reasoning_as_a_thinking_chunk() {
        let message = finalized_message(serde_json::json!({
            "role": "assistant",
            "content": "42",
            "reasoning_content": "6 times 7 is 42."
        }));

        assert_eq!(
            message["content"],
            serde_json::json!([
                {"type": "thinking",
                 "thinking": [{"type": "text", "text": "6 times 7 is 42."}],
                 "closed": true},
                {"type": "text", "text": "42"}
            ])
        );
        assert!(
            message.get("reasoning_content").is_none(),
            "Mistral rejects unknown assistant fields; the trace rides in `content`"
        );
    }

    /// A reasoning turn that only called a tool has no text to carry the trace
    /// beside — the chunk becomes the message's whole content, which is what
    /// Mistral returned in the first place.
    #[test]
    fn finalize_replays_reasoning_beside_a_tool_call() {
        let message = finalized_message(serde_json::json!({
            "role": "assistant",
            "content": "",
            "reasoning_content": "I need the add tool.",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": "{\"a\":2,\"b\":3}"}
            }]
        }));

        assert_eq!(
            message["content"],
            serde_json::json!([
                {"type": "thinking",
                 "thinking": [{"type": "text", "text": "I need the add tool."}],
                 "closed": true}
            ])
        );
        assert_eq!(message["tool_calls"][0]["function"]["name"], "add");
    }

    /// The trace leads content that is already a chunk array, so a reasoning
    /// turn replayed beside an image keeps both.
    #[test]
    fn finalize_replays_reasoning_ahead_of_a_chunk_array() {
        let message = finalized_message(serde_json::json!({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Look:"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}
            ],
            "reasoning_content": "The user asked for a picture."
        }));

        assert_eq!(message["content"][0]["type"], "thinking");
        assert_eq!(message["content"][1]["type"], "text");
        assert_eq!(message["content"][2]["type"], "image_url");
    }

    /// An assistant turn with no content at all is still a place to put a
    /// trace; the message's own `content` default must not overwrite it.
    #[test]
    fn finalize_replays_reasoning_onto_a_contentless_turn() {
        let message = finalized_message(serde_json::json!({
            "role": "assistant",
            "reasoning_content": "Thinking about it."
        }));

        assert_eq!(message["content"][0]["type"], "thinking");
        assert_eq!(
            message["content"][0]["thinking"][0]["text"],
            "Thinking about it."
        );
    }

    /// Empty, null and non-string reasoning replay nothing — an empty trace is
    /// not a trace, and it must not become an empty thinking chunk.
    #[test]
    fn finalize_drops_reasoning_that_carries_no_trace() {
        for reasoning in [
            serde_json::json!(""),
            serde_json::Value::Null,
            serde_json::json!({"text": "nope"}),
        ] {
            let message = finalized_message(serde_json::json!({
                "role": "assistant",
                "content": "42",
                "reasoning_content": reasoning
            }));
            assert_eq!(message["content"], serde_json::json!("42"));
            assert!(message.get("reasoning_content").is_none());
        }
    }

    /// A user message is not an assistant turn: its `reasoning_content` — a
    /// field the conversion never emits there — is left where it is rather
    /// than becoming content the user did not send.
    #[test]
    fn finalize_only_replays_an_assistants_reasoning() {
        let message = finalized_message(serde_json::json!({
            "role": "user",
            "content": "hi",
            "reasoning_content": "not mine"
        }));
        assert_eq!(message["content"], serde_json::json!("hi"));
        assert_eq!(message["reasoning_content"], "not mine");
    }

    /// `finalize_request_body` is a public trait method, so its own output
    /// must survive a second pass: the thinking chunk it emits reads as a
    /// thinking chunk, not as unsupported content.
    #[test]
    fn finalize_is_idempotent_over_a_replayed_trace() {
        let once = finalized_message(serde_json::json!({
            "role": "assistant",
            "content": "42",
            "reasoning_content": "6 times 7 is 42."
        }));
        let twice = finalized_message(once.clone());
        assert_eq!(once, twice);
    }

    /// A thinking chunk with no trace is content Mistral cannot read, so it
    /// fails closed like every other payload-less chunk rather than being
    /// dropped from the request — or, worse, replayed as an empty trace, which
    /// is exactly what [`splice_reasoning_into_content`] refuses on the other
    /// side. `{"thinking": []}` is included because that is the closing frame
    /// the streaming wire sends.
    #[test]
    fn finalize_rejects_a_thinking_part_with_no_trace() {
        for part in [
            serde_json::json!({"type": "thinking"}),
            serde_json::json!({"type": "thinking", "thinking": []}),
            serde_json::json!({"type": "thinking", "thinking": ""}),
            serde_json::json!({"type": "thinking", "thinking": {"text": "nope"}}),
        ] {
            let mut body = serde_json::json!({
                "model": MISTRAL_SMALL,
                "messages": [{"role": "assistant", "content": [
                    part.clone(),
                    {"type": "text", "text": "42"}
                ]}]
            });
            let error = MistralExt
                .finalize_request_body(&mut body)
                .expect_err(&format!("{part} carries no trace and must not be sent"));
            assert!(matches!(error, CompletionError::RequestError(_)));
        }
    }
}
