use serde::{Deserialize, Deserializer, Serialize};

use super::client::{Mistral, Usage};
use crate::providers::openai;
use crate::{
    completion::{self, CompletionError},
    json_utils,
};

/// The latest version of the `codestral` Mistral model
pub const CODESTRAL: &str = "codestral-latest";
/// The latest version of the `mistral-large` Mistral model
pub const MISTRAL_LARGE: &str = "mistral-large-latest";
/// The latest version of the `mistral-3b` Mistral completions model
pub const MINISTRAL_3B: &str = "ministral-3b-latest";
/// The latest version of the `mistral-8b` Mistral completions model
pub const MINISTRAL_8B: &str = "ministral-8b-latest";

/// The latest version of the `mistral-small` Mistral completions model
pub const MISTRAL_SMALL: &str = "mistral-small-latest";

/// Mistral completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = crate::http_client::BoxedHttpClient> =
    openai::completion::GenericCompletionModel<Mistral, H>;

/// Mistral's provider-native terminal streaming record: the value carried by
/// the final item of the stream returned by `CompletionModel::raw_stream`.
/// Shared with the OpenAI Chat Completions path but carrying Mistral's own
/// usage payload (cached-token fallbacks).
pub type MistralStreamingCompletionResponse =
    openai::StreamingCompletionResponse<super::client::Usage>;

// =================================================================
// Rig Implementation Types
// =================================================================

fn mistral_content_value_to_text(value: serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text,
        serde_json::Value::Array(parts) => openai::completion::joined_text_parts(&parts),
        _ => String::new(),
    }
}

fn deserialize_mistral_content_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<serde_json::Value>::deserialize(deserializer)?
        .map(mistral_content_value_to_text)
        .unwrap_or_default())
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
         `{AUDIO_CHUNK}`, `{DOCUMENT_CHUNK}` and `{FILE_CHUNK}` content; convert the content \
         to one of those before sending it."
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
fn into_mistral_chunk(part: &serde_json::Value) -> Result<serde_json::Value, CompletionError> {
    /// Text and refusal parts are both re-tagged as `text`: Mistral's chunk
    /// schema has no `refusal` field, and every chunk forbids unknown keys.
    fn text_chunk(part: &serde_json::Value) -> Result<serde_json::Value, CompletionError> {
        let text = part_text(part)
            .ok_or_else(|| unsupported_content_error("a text content part carrying no text"))?;
        Ok(serde_json::json!({"type": TEXT_CHUNK, TEXT_CHUNK: text}))
    }

    match part.get("type").and_then(serde_json::Value::as_str) {
        Some(TEXT_CHUNK | REFUSAL_TYPE) => text_chunk(part),
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
        Some(AUDIO_CHUNK) => audio_part_to_mistral_chunk(part),
        Some(FILE_CHUNK) => file_part_to_mistral_chunk(part),
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
        Some(kind) => Err(unsupported_content_error(&format!(
            "`{kind}` message content"
        ))),
        // Untagged, but textual: the shared flattening would have taken it, so
        // it converts rather than failing.
        None if part_text(part).is_some() => text_chunk(part),
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
            *part = into_mistral_chunk(part)?;
        }
    }

    Ok(())
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
        #[serde(default, deserialize_with = "deserialize_mistral_content_string")]
        content: String,
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
    #[serde(with = "json_utils::stringified_json")]
    pub arguments: serde_json::Value,
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
    #[serde(
        deserialize_with = "crate::providers::internal::openai_chat_completions_compatible::deserialize_choices_dropping_incomplete_tool_calls"
    )]
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

impl crate::telemetry::ProviderResponseExt for CompletionResponse {
    type Usage = Usage;

    fn response_id(&self) -> Option<&str> {
        Some(self.id.as_str())
    }

    fn response_model_name(&self) -> Option<&str> {
        Some(self.model.as_str())
    }

    fn text_response(&self) -> Option<String> {
        let res = self
            .choices
            .iter()
            .filter_map(|choice| match choice.message {
                Message::Assistant { ref content, .. } => {
                    if content.is_empty() {
                        None
                    } else {
                        Some(content.clone())
                    }
                }
                _ => None,
            })
            .collect::<Vec<String>>()
            .join("\n");

        if res.is_empty() { None } else { Some(res) }
    }

    fn usage(&self) -> Option<Self::Usage> {
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
                } => Some(compat::text_then_tool_calls(
                    content,
                    content.is_empty(),
                    tool_calls.iter().map(|call| {
                        (
                            call.id.as_str(),
                            call.function.name.as_str(),
                            call.function.arguments.clone(),
                        )
                    }),
                )),
                _ => None,
            },
        )
    }
}

#[cfg(test)]
mod tests;
