//! The Converse request body, built as JSON from a prepared request. An
//! assistant block goes back through [`AssistantContent::replay`]: a current
//! provider item as it came, anything else from its canonical fields for
//! the target model's family. Every call id is spelled by one [`WireIds`].

use std::collections::HashMap;

use base64::Engine as _;
use base64::alphabet::STANDARD;
use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
use base64::prelude::BASE64_STANDARD;
use rig_core::completion::{CompletionRequest, Message, Replay};
use rig_core::error::EncodeError;
use rig_core::message::{
    AssistantContent, Document, DocumentMediaType, DocumentSourceKind, Image, ImageMediaType,
    ToolChoice, ToolResultContent, UserContent, Video, VideoMediaType,
};
use rig_core::providers::internal::wire_ids::WireIds;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::completion::{Converse, Family};

/// Standard base64, padded or not.
const BASE64: GeneralPurpose = GeneralPurpose::new(
    &STANDARD,
    GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent),
);

/// What stands in for a user message left empty once blank text is skipped:
/// Converse rejects blank text and empty content.
const EMPTY_TEXT: &str = "<empty>";

/// The Converse body of `request` for `model` on `wire`. A guardrail
/// applies to a `unary` request only.
pub(crate) fn body(
    wire: &Converse,
    request: CompletionRequest,
    model: &str,
    unary: bool,
) -> Result<Value, EncodeError> {
    let family = wire.family(model);
    let mut body = Map::new();
    // The system messages that lead the history are the system prompt. A
    // later one stays where the history puts it, as user text, so adding
    // one never changes the cached prefix before it.
    let leading = request
        .chat_history
        .iter()
        .take_while(|message| matches!(message, Message::System { .. }))
        .count();
    let mut system: Vec<Value> = request
        .chat_history
        .iter()
        .take(leading)
        .filter_map(|message| match message {
            Message::System { content } if !content.is_empty() => Some(json!({ "text": content })),
            _ => None,
        })
        .collect();
    if !system.is_empty() {
        if wire.prompt_caching {
            system.push(json!({ "cachePoint": { "type": "default" } }));
        }
        body.insert("system".to_owned(), Value::Array(system));
    }
    let mut inference = Map::new();
    if let Some(temperature) = request.temperature {
        inference.insert(
            "temperature".to_owned(),
            json!(f64::from(temperature as f32)),
        );
    }
    if let Some(max_tokens) = request.max_tokens {
        inference.insert("maxTokens".to_owned(), json!(max_tokens as i32));
    }
    body.insert("inferenceConfig".to_owned(), Value::Object(inference));
    if let Some(config) = tool_config(&request) {
        body.insert("toolConfig".to_owned(), config);
    }
    if let Some(params) = &request.additional_params {
        body.insert("additionalModelRequestFields".to_owned(), params.clone());
    }
    if let Some(schema) = &request.output_schema {
        let schema = serde_json::to_string(schema.as_value())?;
        let name = request.output_schema_name();
        let format = json!({ "type": "json_schema", "structure": { "jsonSchema": { "schema": schema, "name": name } } });
        body.insert("outputConfig".to_owned(), json!({ "textFormat": format }));
    }
    if let Some(guardrail) = wire.guardrail.as_ref().filter(|_| unary) {
        body.insert("guardrailConfig".to_owned(), guardrail.clone());
    }
    let history = request.chat_history.get(leading..).unwrap_or_default();
    let ids = WireIds::for_target(history, wire, model);
    let mut messages: Vec<Value> = Vec::new();
    for message in history {
        let (role, mut content) = match message {
            Message::System { content } => ("user", vec![json!({ "text": content })]),
            Message::User { content } => {
                let mut blocks = Vec::new();
                for part in content {
                    blocks.extend(user(part, family, &ids)?);
                }
                if blocks.is_empty() {
                    blocks.push(json!({ "text": EMPTY_TEXT }));
                }
                ("user", blocks)
            }
            Message::Assistant(turn) => {
                let mut blocks = Vec::new();
                for block in &turn.content {
                    blocks.extend(assistant(block, wire, family, &ids)?);
                }
                // Converse rejects an empty message.
                if blocks.is_empty() {
                    continue;
                }
                ("assistant", blocks)
            }
        };
        // Converse alternates roles: a turn the adapter left beside another
        // of its role, such as two turns an orphan result separated, joins it.
        match messages.last_mut().and_then(|last| last.as_object_mut()) {
            Some(last) if last.get("role") == Some(&json!(role)) => {
                if let Some(Value::Array(previous)) = last.get_mut("content") {
                    previous.append(&mut content);
                }
            }
            _ => messages.push(json!({ "role": role, "content": content })),
        }
    }
    // Bedrock requires a name on every document and Rig's `Document` carries
    // none. A name repeated within one request takes a counter.
    let mut seen = HashMap::<String, usize>::new();
    for document in
        blocks(&mut messages).filter_map(|block| block.get_mut("document")?.as_object_mut())
    {
        if let Some(Value::String(name)) = document.get_mut("name") {
            let count = seen.entry(name.clone()).or_insert(0);
            *count += 1;
            if *count > 1 {
                *name = format!("{name}-{count}");
            }
        }
    }
    // Bedrock rejects a cache point anywhere after a reasoning turn, even on
    // the user's side.
    let reasoning = blocks(&mut messages).any(|block| block.get("reasoningContent").is_some());
    if wire.prompt_caching
        && !reasoning
        && let Some(Value::Array(last)) =
            messages.last_mut().and_then(|last| last.get_mut("content"))
    {
        last.push(json!({ "cachePoint": { "type": "default" } }));
    }
    body.insert("messages".to_owned(), Value::Array(messages));
    Ok(Value::Object(body))
}

/// Every content block of `messages`.
fn blocks(messages: &mut [Value]) -> impl Iterator<Item = &mut Value> {
    messages
        .iter_mut()
        .filter_map(|message| message.get_mut("content")?.as_array_mut())
        .flatten()
}

/// The `toolConfig` of `request`: none without tools or with
/// `ToolChoice::None`, when the adapter sends calls and results as text.
fn tool_config(request: &CompletionRequest) -> Option<Value> {
    let choice = match &request.tool_choice {
        Some(ToolChoice::None) => return None,
        Some(ToolChoice::Auto) => Some(json!({ "auto": {} })),
        Some(ToolChoice::Required) => Some(json!({ "any": {} })),
        Some(ToolChoice::Specific { function_names }) => function_names
            .first()
            .map(|name| json!({ "tool": { "name": name.as_str() } })),
        None => None,
    };
    if request.tools.is_empty() {
        return None;
    }
    let tools: Vec<Value> = request
        .tools
        .iter()
        .map(|tool| {
            json!({ "toolSpec": {
                "name": tool.name.as_str(),
                "description": tool.description,
                "inputSchema": { "json": tool.parameters },
            } })
        })
        .collect();
    let mut config = Map::from_iter([("tools".to_owned(), Value::Array(tools))]);
    if let Some(choice) = choice {
        config.insert("toolChoice".to_owned(), choice);
    }
    Some(Value::Object(config))
}

/// The Converse block for one assistant block, or `None` when there is
/// nothing to send.
fn assistant(
    block: &AssistantContent,
    wire: &Converse,
    family: Family,
    ids: &WireIds,
) -> Result<Option<Value>, EncodeError> {
    if let AssistantContent::Opaque(opaque) = block {
        return Ok(opaque.replay.then(|| whole_numbers(opaque.item.clone())));
    }
    if let Replay::Item(item) = block.replay(wire, ids) {
        return Ok(Some(whole_numbers(item.into_owned())));
    }
    Ok(match block {
        AssistantContent::Text(text) => Some(json!({ "text": text.text })),
        AssistantContent::ToolCall(call) => Some(json!({ "toolUse": {
            "toolUseId": ids.of(&call.id).map_or_else(|| call.id.wire(), Into::into),
            "name": call.function.name.as_str(),
            "input": call.function.arguments,
        } })),
        // Redacted reasoning is nothing without its bytes.
        AssistantContent::Reasoning(reasoning)
            if reasoning.redacted || reasoning.text.trim().is_empty() =>
        {
            None
        }
        // Claude rejects unsigned reasoning: it goes back as text.
        AssistantContent::Reasoning(reasoning) if family == Family::Claude => {
            Some(json!({ "text": reasoning.text }))
        }
        AssistantContent::Reasoning(reasoning) => {
            Some(json!({ "reasoningContent": { "reasoningText": { "text": reasoning.text } } }))
        }
        AssistantContent::Image(image) => Some(json!({ "image": self::image(image)? })),
        AssistantContent::Opaque(_) => None,
    })
}

/// `value` with each whole number written as an integer: a store may write
/// one as a float, and Converse's integer fields take integers.
fn whole_numbers(value: Value) -> Value {
    match value {
        Value::Number(number) => match number.as_f64() {
            Some(float) if number.is_f64() && float.fract() == 0.0 && float.abs() < 1e15 => {
                json!(float as i64)
            }
            _ => Value::Number(number),
        },
        Value::Array(values) => Value::Array(values.into_iter().map(whole_numbers).collect()),
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| (key, whole_numbers(value)))
                .collect(),
        ),
        value => value,
    }
}

/// The Converse blocks for one piece of user content. Converse rejects
/// blank text, so it has none.
fn user(content: &UserContent, family: Family, ids: &WireIds) -> Result<Vec<Value>, EncodeError> {
    let text = |text: &str| (!text.trim().is_empty()).then(|| json!({ "text": text }));
    Ok(match content {
        UserContent::Text(part) => text(&part.text).into_iter().collect(),
        UserContent::ToolResult(result) => {
            let mut parts = Vec::new();
            for part in &result.content {
                parts.extend(match part {
                    ToolResultContent::Text(part) => text(&part.text),
                    ToolResultContent::Image(part) => Some(json!({ "image": image(part)? })),
                    // Converse takes an object as a result's JSON.
                    ToolResultContent::Json { value } if value.is_object() => {
                        Some(json!({ "json": value }))
                    }
                    ToolResultContent::Json { value } => {
                        Some(json!({ "json": { "result": value } }))
                    }
                });
            }
            let mut block = Map::from_iter([
                (
                    "toolUseId".to_owned(),
                    json!(
                        ids.of(&result.call)
                            .map_or_else(|| result.call.wire(), Into::into)
                    ),
                ),
                ("content".to_owned(), Value::Array(parts)),
            ]);
            // Converse reads a result with no status as a success, and
            // documents the field for Nova and Claude only.
            if result.is_error && family != Family::Other {
                block.insert("status".to_owned(), json!("error"));
            }
            vec![json!({ "toolResult": block })]
        }
        UserContent::Image(part) => vec![json!({ "image": image(part)? })],
        // Converse requires accompanying prompt text for document blocks.
        UserContent::Document(part) => vec![
            json!({ "text": "Use provided document" }),
            json!({ "document": document(part)? }),
        ],
        UserContent::Video(part) => vec![json!({ "video": video(part)? })],
        UserContent::Audio(_) => return Err(EncodeError::request("Converse takes no audio")),
    })
}

/// The source of a media part, as base64 bytes or an S3 object, and the
/// bytes that name it.
fn source(data: &DocumentSourceKind) -> Result<(Value, Vec<u8>), EncodeError> {
    let bytes = match data {
        DocumentSourceKind::Base64(data) => BASE64.decode(data).map_err(EncodeError::request)?,
        DocumentSourceKind::Raw(bytes) => bytes.clone(),
        DocumentSourceKind::Url(uri) if uri.starts_with("s3://") => {
            return Ok((
                json!({ "s3Location": { "uri": uri } }),
                uri.as_bytes().to_vec(),
            ));
        }
        data => {
            return Err(EncodeError::request(format!(
                "Converse takes base64 data or an s3:// URL, not {data}"
            )));
        }
    };
    Ok((json!({ "bytes": BASE64_STANDARD.encode(&bytes) }), bytes))
}

/// The Converse image block for `image`: PNG, JPEG, GIF or WEBP.
pub(crate) fn image(image: &Image) -> Result<Value, EncodeError> {
    let format = match &image.media_type {
        Some(ImageMediaType::JPEG) => "jpeg",
        Some(ImageMediaType::PNG) => "png",
        Some(ImageMediaType::GIF) => "gif",
        Some(ImageMediaType::WEBP) => "webp",
        _ => {
            return Err(EncodeError::request(
                "Converse takes PNG, JPEG, GIF or WEBP images",
            ));
        }
    };
    Ok(json!({ "format": format, "source": source(&image.data)?.0 }))
}

/// The Converse document block for `document`. A text format Converse does
/// not list is plain text. The name is the content's digest, so a request
/// is byte-stable across turns and runs, which prompt-cache prefixes and
/// recorded replays rely on.
pub(crate) fn document(document: &Document) -> Result<Value, EncodeError> {
    let format = match &document.media_type {
        Some(DocumentMediaType::PDF) => "pdf",
        Some(DocumentMediaType::HTML) => "html",
        Some(DocumentMediaType::MARKDOWN) => "md",
        Some(DocumentMediaType::CSV) => "csv",
        Some(_) => "txt",
        None => return Err(EncodeError::request("Converse needs a document's format")),
    };
    let (source, bytes) = source(&document.data)?;
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .take(8)
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Ok(json!({ "format": format, "name": format!("document-{digest}"), "source": source }))
}

/// The Converse video block for `video`, in a format Converse lists.
pub(crate) fn video(video: &Video) -> Result<Value, EncodeError> {
    let format = match &video.media_type {
        Some(VideoMediaType::MP4) => "mp4",
        Some(VideoMediaType::MPEG) => "mpeg",
        Some(VideoMediaType::MOV) => "mov",
        Some(VideoMediaType::WEBM) => "webm",
        _ => {
            return Err(EncodeError::request(
                "Converse takes no video in this format",
            ));
        }
    };
    Ok(json!({ "format": format, "source": source(&video.data)?.0 }))
}

#[cfg(test)]
mod tests;
