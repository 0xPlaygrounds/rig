//! The one place rig's message model meets Gemini. Every Gemini wire
//! (GenerateContent, Interactions, and companion transports such as gRPC and
//! Vertex) speaks through it: rig's messages become [`Unit`]s and units become
//! rig's parts, and every normalization decision lives here.
//!
//! Text, thoughts, function calls, function results and media map to units
//! one to one. Anything else a wire returns is a [`Unit::Native`] holding the
//! part's JSON exactly as it arrived, re-sent verbatim to the dialect that
//! issued it.
//!
//! ```
//! use rig_core::message::UserContent;
//! use rig_core::providers::gemini::edge::{self, Unit};
//!
//! let unit = edge::user_unit(UserContent::text("hi"))?;
//! assert_eq!(unit, Unit::Text { text: "hi".into(), signature: None });
//! # Ok::<(), rig_core::error::EncodeError>(())
//! ```

use serde_json::value::RawValue;
use serde_json::{Map, Value};

use super::ISSUER;
use crate::completion::{FinishReason, Usage};
use crate::error::{EncodeError, ProviderError};
use crate::message::{
    self, AssistantContent, CallId, DocumentSourceKind, MediaDetail, MimeType, NativePart,
    ReasoningContent, Signature, ToolCall, ToolFunction, ToolName, ToolResultContent, UserContent,
};
use crate::operation::{Completion, TextPart};
use crate::providers::internal::thoughts::Thoughts;
use crate::wire::Out;

/// One normalized piece of a Gemini turn, the vocabulary both dialects share.
#[derive(Clone, Debug, PartialEq)]
pub enum Unit {
    /// Visible text, and the signature Gemini attached to its part.
    Text {
        text: String,
        signature: Option<String>,
    },
    /// A thought, and the signature that closes it.
    Thought {
        text: String,
        signature: Option<String>,
    },
    /// A function call the model asks the client to run.
    Call {
        id: Option<String>,
        name: String,
        args: Map<String, Value>,
        signature: Option<String>,
    },
    /// The client's answer to a call.
    Result {
        id: Option<String>,
        name: String,
        response: Option<Map<String, Value>>,
        media: Vec<Media>,
    },
    /// An image, audio clip, video or document.
    Media(Media),
    /// A part in the dialect's own schema, re-sent as it arrived.
    Native(NativePart),
}

/// Media in a user turn or a function result.
#[derive(Clone, Debug, PartialEq)]
pub struct Media {
    pub mime_type: Option<String>,
    pub source: Source,
    pub detail: Option<MediaDetail>,
}

/// Where a media part's bytes are.
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    /// Base64 bytes in the request.
    Inline(String),
    /// A URI Gemini fetches: a Files API resource, a bucket or a public URL.
    Uri(String),
}

/// One Gemini wire's part vocabulary.
pub trait Dialect {
    /// The schema a native part of this dialect records.
    const SCHEMA: &'static str;
    /// A part of a turn on this wire.
    type Part: serde::Serialize;

    /// The units of one part as received. A part rig cannot represent
    /// exactly is [`Unit::Native`] with `raw` as its JSON.
    fn units(raw: &RawValue) -> Result<Vec<Unit>, serde_json::Error>;

    /// The wire part for `unit`. A native part of another schema is refused.
    fn part(unit: Unit) -> Result<Encoded<Self::Part>, EncodeError>;
}

/// A part ready for a request body: built from a unit, or a native part's
/// JSON sent unchanged.
#[derive(Debug)]
pub enum Encoded<P> {
    Typed(P),
    Raw(Box<RawValue>),
}

impl<P: serde::Serialize> serde::Serialize for Encoded<P> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Typed(part) => part.serialize(serializer),
            Self::Raw(raw) => raw.serialize(serializer),
        }
    }
}

/// A native part of `dialect`, or the error for a part it cannot send.
pub fn native_raw(native: NativePart, schema: &str) -> Result<Box<RawValue>, EncodeError> {
    if native.schema != schema {
        return Err(EncodeError::request(format!(
            "a native `{}` part cannot be sent as `{schema}`",
            native.schema
        )));
    }
    Ok(native.part)
}

/// The native part for `raw` in `schema`.
pub fn native(raw: &RawValue, schema: &'static str) -> Unit {
    Unit::Native(NativePart::new(schema, raw.to_owned()))
}

/// The unit of a user part.
pub fn user_unit(content: UserContent) -> Result<Unit, EncodeError> {
    Ok(match content {
        UserContent::Text(text) => Unit::Text {
            text: text.text,
            signature: None,
        },
        UserContent::ToolResult(result) => {
            let mut values = Vec::new();
            let mut media = Vec::new();
            for item in result.content {
                match item {
                    ToolResultContent::Text(text) => values.push(Value::String(text.text)),
                    ToolResultContent::Json { value } => values.push(value),
                    ToolResultContent::Image(image) => {
                        let mime_type = tool_result_image_mime(image.media_type.as_ref())?;
                        let DocumentSourceKind::Base64(data) = image.data else {
                            return Err(EncodeError::request(
                                "a Gemini tool result image must be base64 data",
                            ));
                        };
                        media.push(Media {
                            mime_type: Some(mime_type.to_owned()),
                            source: Source::Inline(data),
                            detail: image.detail,
                        });
                    }
                }
            }
            // Gemini's response is an object; the result sits under one key.
            let response = match values.len() {
                0 => None,
                1 => values.pop(),
                _ => Some(Value::Array(values)),
            }
            .map(|result| Map::from_iter([("result".to_owned(), result)]));
            Unit::Result {
                id: result.call.provider().map(|id| id.call_id.clone()),
                name: result.name.into(),
                response,
                media,
            }
        }
        UserContent::Image(image) => {
            let mime_type = match image.media_type {
                Some(
                    media_type @ (message::ImageMediaType::JPEG
                    | message::ImageMediaType::PNG
                    | message::ImageMediaType::WEBP
                    | message::ImageMediaType::HEIC
                    | message::ImageMediaType::HEIF),
                ) => media_type.to_mime_type(),
                Some(other) => {
                    return Err(EncodeError::request(format!(
                        "Gemini does not take {other:?} images"
                    )));
                }
                None => return Err(EncodeError::request("a Gemini image needs a media type")),
            };
            Unit::Media(media(
                image.data,
                Some(mime_type.to_owned()),
                image.detail,
                "image",
            )?)
        }
        UserContent::Audio(audio) => {
            let mime_type = audio
                .media_type
                .ok_or_else(|| EncodeError::request("Gemini audio needs a media type"))?;
            Unit::Media(media(
                audio.data,
                Some(mime_type.to_mime_type().to_owned()),
                None,
                "audio",
            )?)
        }
        UserContent::Video(video) => {
            if video.additional_params.is_some() {
                return Err(EncodeError::request(
                    "Gemini does not read a video's additional_params",
                ));
            }
            let mime_type = video
                .media_type
                .map(|media| media.to_mime_type().to_owned());
            // A YouTube link is the one source Gemini resolves without a type.
            let youtube = matches!(
                &video.data,
                DocumentSourceKind::Url(url) if url.starts_with("https://www.youtube.com")
            );
            if mime_type.is_none() && !youtube {
                return Err(EncodeError::request(
                    "a Gemini video needs a media type unless it is a YouTube link",
                ));
            }
            Unit::Media(media(video.data, mime_type, video.detail, "video")?)
        }
        UserContent::Document(document) => {
            let media_type = document
                .media_type
                .ok_or_else(|| EncodeError::request("a Gemini document needs a media type"))?;
            if media_type.is_code() || text_like(&media_type) {
                // Text documents are context, not files: inline ones are text.
                match document.data {
                    DocumentSourceKind::String(text) => {
                        return Ok(Unit::Text {
                            text,
                            signature: None,
                        });
                    }
                    DocumentSourceKind::Base64(data) => {
                        use base64::Engine;
                        let bytes = base64::engine::general_purpose::STANDARD
                            .decode(&data)
                            .map_err(|error| {
                                EncodeError::request(format!("a document is not base64: {error}"))
                            })?;
                        let text = String::from_utf8(bytes).map_err(|error| {
                            EncodeError::request(format!("a text document is not UTF-8: {error}"))
                        })?;
                        return Ok(Unit::Text {
                            text,
                            signature: None,
                        });
                    }
                    _ => {}
                }
            }
            Unit::Media(media(
                document.data,
                Some(media_type.to_mime_type().to_owned()),
                document.detail,
                "document",
            )?)
        }
    })
}

fn text_like(media_type: &message::DocumentMediaType) -> bool {
    matches!(
        media_type,
        message::DocumentMediaType::TXT
            | message::DocumentMediaType::RTF
            | message::DocumentMediaType::HTML
            | message::DocumentMediaType::CSS
            | message::DocumentMediaType::MARKDOWN
            | message::DocumentMediaType::CSV
            | message::DocumentMediaType::XML
    )
}

/// Gemini takes PNG, JPEG and WEBP images inside a function response.
fn tool_result_image_mime(
    media_type: Option<&message::ImageMediaType>,
) -> Result<&'static str, EncodeError> {
    match media_type {
        Some(
            media_type @ (message::ImageMediaType::JPEG
            | message::ImageMediaType::PNG
            | message::ImageMediaType::WEBP),
        ) => Ok(media_type.to_mime_type()),
        Some(other) => Err(EncodeError::request(format!(
            "a Gemini tool result cannot carry {other:?} images"
        ))),
        None => Err(EncodeError::request(
            "a Gemini tool result image needs a media type",
        )),
    }
}

fn media(
    source: DocumentSourceKind,
    mime_type: Option<String>,
    detail: Option<MediaDetail>,
    kind: &str,
) -> Result<Media, EncodeError> {
    let source = match source {
        DocumentSourceKind::Base64(data) => Source::Inline(data),
        DocumentSourceKind::Raw(bytes) => {
            use base64::Engine;
            Source::Inline(base64::engine::general_purpose::STANDARD.encode(bytes))
        }
        // A Files API resource is addressed by its URI.
        DocumentSourceKind::Url(uri) | DocumentSourceKind::FileId(uri) => Source::Uri(uri),
        DocumentSourceKind::String(_) => {
            return Err(EncodeError::request(format!(
                "a Gemini {kind} cannot be a plain string; use base64 or a URL"
            )));
        }
        DocumentSourceKind::Unknown => {
            return Err(EncodeError::request(format!(
                "a Gemini {kind} has no source"
            )));
        }
    };
    Ok(Media {
        mime_type,
        source,
        detail,
    })
}

/// The units of an assistant part for a wire that replays what `issuer`
/// sealed. Reasoning and signatures another service issued are not
/// replayed; a native part another service issued cannot be sent.
pub fn assistant_units(
    content: AssistantContent,
    issuer: &message::Issuer,
) -> Result<Vec<Unit>, EncodeError> {
    let signature = |signature: Option<message::Sealed<Signature>>| {
        signature
            .as_ref()
            .and_then(|signature| signature.open(issuer))
            .map(|signature| signature.signature.clone())
    };
    Ok(match content {
        AssistantContent::Text(text) => vec![Unit::Text {
            signature: signature(text.signature),
            text: text.text,
        }],
        AssistantContent::Reasoning(reasoning) => {
            let Some(reasoning) = reasoning.open(issuer) else {
                return Ok(Vec::new());
            };
            reasoning
                .content
                .iter()
                .filter_map(|block| match block {
                    ReasoningContent::Text { text, signature } => Some(Unit::Thought {
                        text: text.clone(),
                        signature: signature.clone(),
                    }),
                    ReasoningContent::Summary(text) => Some(Unit::Thought {
                        text: text.clone(),
                        signature: None,
                    }),
                    // Gemini issues neither; nothing of Gemini's is left out.
                    ReasoningContent::Encrypted(_) | ReasoningContent::Redacted { .. } => None,
                })
                .collect()
        }
        AssistantContent::ToolCall(call) => {
            let Value::Object(args) = call.function.arguments else {
                return Err(EncodeError::request(format!(
                    "Gemini function call `{}` needs object arguments",
                    call.function.name
                )));
            };
            vec![Unit::Call {
                id: call.id.provider().map(|id| id.call_id.clone()),
                name: call.function.name.into(),
                args,
                signature: signature(call.signature),
            }]
        }
        AssistantContent::Image(image) => {
            let mime_type = image
                .media_type
                .map(|media| media.to_mime_type().to_owned());
            vec![Unit::Media(media(
                image.data,
                mime_type,
                image.detail,
                "image",
            )?)]
        }
        AssistantContent::Native(native) => match native.open(issuer) {
            Some(part) => vec![Unit::Native(part.clone())],
            None => return Err(AssistantContent::foreign_native(&native, "gemini").into()),
        },
    })
}

/// Writes one reply's units into the completion fold: text and thought
/// fragments extend their open part until a signature, or another kind of
/// unit, ends it.
pub struct Writer<'id> {
    thoughts: Thoughts<'id>,
    text: Option<TextPart<'id>>,
    /// A unit that answers the prompt arrived: text, a thought or a call.
    delivered: bool,
}

impl Default for Writer<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'id> Writer<'id> {
    pub fn new() -> Self {
        Self {
            thoughts: Thoughts::new(),
            text: None,
            delivered: false,
        }
    }

    /// Whether anything but hosted-tool work arrived.
    pub fn delivered(&self) -> bool {
        self.delivered
    }

    /// A call the wire streams in pieces starts: it answers the prompt and
    /// interrupts open text and thought.
    pub fn interrupt(&mut self, out: &mut Out<'id, Completion>) {
        self.delivered = true;
        self.thoughts.boundary();
        self.split_text(out);
    }

    /// End the open text part: the next text fragment starts another.
    pub fn split_text(&mut self, out: &mut Out<'id, Completion>) {
        if let Some(part) = self.text.take() {
            out.close_text(part);
        }
    }

    /// Write `unit`. `answers` says whether a native unit is part of the
    /// answer rather than hosted-tool work.
    pub fn unit(
        &mut self,
        unit: Unit,
        answers: bool,
        out: &mut Out<'id, Completion>,
    ) -> Result<(), ProviderError> {
        match unit {
            Unit::Thought { text, signature } => {
                self.delivered = true;
                if !text.is_empty() {
                    self.split_text(out);
                }
                self.thoughts.fragment(out, &text);
                if let Some(signature) = signature {
                    self.thoughts.signature(out, signature);
                }
            }
            Unit::Text { text, signature } => {
                if !text.is_empty() {
                    self.delivered = true;
                }
                // A signature on an empty part is a part of its own.
                if signature.is_some() && text.is_empty() {
                    self.split_text(out);
                }
                if text.is_empty() && signature.is_none() {
                    return Ok(());
                }
                self.thoughts.boundary();
                let part = self.text.get_or_insert_with(|| out.text());
                out.push_text(part, &text);
                if let Some(signature) = signature {
                    out.text_signature(part, signature);
                    self.split_text(out);
                }
            }
            Unit::Call {
                id,
                name,
                args,
                signature,
            } => {
                self.delivered = true;
                self.thoughts.boundary();
                self.split_text(out);
                let name = ToolName::new(name).map_err(|_| {
                    ProviderError::Response("Gemini sent a function call without a name".into())
                })?;
                out.tool_call(ToolCall {
                    id: CallId::from_wire(id.unwrap_or_default()),
                    function: ToolFunction::new(name, Value::Object(args)),
                    signature: signature.map(|signature| Signature::sealed(ISSUER, signature)),
                    additional_params: None,
                })?;
            }
            Unit::Native(native) => {
                self.delivered |= answers;
                self.thoughts.boundary();
                self.split_text(out);
                out.native(native);
            }
            Unit::Result { .. } | Unit::Media(_) => {
                return Err(ProviderError::Response(
                    "Gemini returned client content in a model turn".into(),
                ));
            }
        }
        Ok(())
    }

    /// Close every open part.
    pub fn close(&mut self, out: &mut Out<'id, Completion>) {
        self.split_text(out);
        self.thoughts.close(out, None);
    }
}

/// The token counts both dialects report, under Google's names.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Counts {
    pub prompt: Option<u64>,
    pub tool_use_prompt: Option<u64>,
    pub cached: Option<u64>,
    pub candidates: Option<u64>,
    pub thoughts: Option<u64>,
}

/// Rig's usage for Gemini's counts: input = prompt + toolUsePrompt, output =
/// candidates + thoughts, total = input + output. Cached tokens are part of
/// the prompt, and thoughts are billed as output.
pub fn usage(counts: Counts) -> Usage {
    let add = |a: Option<u64>, b: Option<u64>| match (a, b) {
        (None, None) => None,
        (a, b) => Some(a.unwrap_or(0).saturating_add(b.unwrap_or(0))),
    };
    let input = add(counts.prompt, counts.tool_use_prompt);
    let output = add(counts.candidates, counts.thoughts);
    Usage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: add(input, output),
        cached_input_tokens: counts.cached,
        cache_creation_input_tokens: None,
        reasoning_tokens: counts.thoughts,
        tool_use_prompt_tokens: counts.tool_use_prompt,
    }
}

/// A count Google sent, when it is one.
pub fn count(value: Option<i32>) -> Option<u64> {
    value.and_then(|value| u64::try_from(value).ok())
}

/// Rig's finish reason for Gemini's wire spelling. `None` for the
/// unspecified reason.
pub fn finish_reason(wire: &str) -> Option<FinishReason> {
    Some(match wire {
        "FINISH_REASON_UNSPECIFIED" => return None,
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY"
        | "BLOCKLIST"
        | "PROHIBITED_CONTENT"
        | "SPII"
        | "IMAGE_SAFETY"
        | "IMAGE_PROHIBITED_CONTENT" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_owned()),
    })
}

/// The error for a finish reason that means the tool protocol failed.
pub fn tool_protocol_error(wire: &str, message: Option<&str>) -> Option<ProviderError> {
    matches!(
        wire,
        "MALFORMED_FUNCTION_CALL"
            | "UNEXPECTED_TOOL_CALL"
            | "MISSING_THOUGHT_SIGNATURE"
            | "TOO_MANY_TOOL_CALLS"
            | "MALFORMED_RESPONSE"
    )
    .then(|| {
        ProviderError::Response(format!(
            "Gemini stopped with finish_reason={wire}: {}",
            message.unwrap_or("no finish message provided")
        ))
    })
}

/// The error for a blocked prompt. A content refusal is final; any other
/// reason may pass on retry.
pub fn blocked(reason: &str, ratings: &[(String, String)]) -> Option<ProviderError> {
    if reason == "BLOCK_REASON_UNSPECIFIED" {
        return None;
    }
    let ratings = if ratings.is_empty() {
        String::new()
    } else {
        let joined = ratings
            .iter()
            .map(|(category, probability)| format!("{category}={probability}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(", safety_ratings=[{joined}]")
    };
    let error = crate::provider_response::ProviderResponseError::without_status(format!(
        "Gemini blocked the prompt: block_reason={reason}{ratings}"
    ))
    .with_code(Some(reason.to_owned()));
    let refusal = matches!(
        reason,
        "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "IMAGE_SAFETY"
    );
    Some(ProviderError::ProviderResponse(if refusal {
        error.with_refusal(true)
    } else {
        error.with_transient(Some(true))
    }))
}

#[cfg(test)]
mod tests;
