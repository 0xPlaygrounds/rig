//! The Converse reply decoder. A stream arrives as SDK events; a whole reply
//! is the SDK's output, or its JSON when the SDK could not read it, written
//! block by block through the same helpers, so both fold into the same turn.
//!
//! Every Converse content block becomes a block. Reasoning, cited text, and
//! a hosted tool's use and result keep their Converse JSON as the provider
//! item, set only when the block is complete: at its `contentBlockStop`, or
//! in a whole reply. A hosted (`server_tool_use`) call is an opaque item
//! that replays to the same model with its result, never a call Rig runs. A block this crate
//! cannot send back is a marker naming its kind. The SDK's `Unknown`
//! variants carry no payload, so an item this SDK version does not model is
//! a marker too.

use std::collections::BTreeMap;

use crate::completion::ConverseFrame;
use crate::types::assistant_content::finish;
use crate::types::{block, json};
use aws_sdk_bedrockruntime::operation::converse::ConverseOutput;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use base64::{Engine, prelude::BASE64_STANDARD};
use rig_core::error::ProviderError;
use rig_core::message::{DocumentSourceKind, Image, ImageMediaType};
use rig_core::operation::{Block, CallFragment, Completion};
use rig_core::wire::{Flow, Out, WireEvent};
use serde_json::{Map, Value};

/// The buffer index of a Converse content block: every signed index,
/// negative ones included, maps to a distinct one.
fn block_index(content_block_index: i32) -> usize {
    (i64::from(content_block_index) - i64::from(i32::MIN)) as usize
}

/// What a stream has stated of one block that becomes whole at its stop.
enum Draft {
    /// Text, and the citations Converse attached to it.
    Text { text: String, citations: Vec<Value> },
    /// Reasoning text and its signature, or its redacted bytes.
    Reasoning {
        text: String,
        signature: Option<String>,
        redacted: Option<Vec<u8>>,
    },
    /// A hosted tool's call: its input arrives as JSON text.
    Hosted {
        id: String,
        name: String,
        kind: String,
        input: String,
    },
    /// A hosted tool's result; `modeled` is false once a part arrived that
    /// this SDK version does not model.
    Result {
        id: String,
        status: Option<String>,
        kind: Option<String>,
        content: Vec<Value>,
        modeled: bool,
    },
    /// An image, written whole at its stop.
    Image {
        format: aws_bedrock::ImageFormat,
        source: Option<aws_bedrock::ImageSource>,
    },
}

/// One Converse reply's state: a whole reply or a stream of events.
#[derive(Default)]
pub struct StreamState {
    drafts: BTreeMap<usize, Draft>,
    stop_reason: Option<aws_bedrock::StopReason>,
    /// The provider's JSON of the stream's message-level events, by event
    /// type: the response's `raw`.
    raw: Map<String, Value>,
}

impl StreamState {
    /// The block at `index` stopped: it becomes whole.
    fn stop(&mut self, out: &mut Out<'_, Completion>, index: usize) -> Result<(), ProviderError> {
        let Some(draft) = self.drafts.remove(&index) else {
            if out.is_open(index) {
                out.finish(index)?;
            }
            return Ok(());
        };
        match draft {
            Draft::Text { text, citations } => {
                let cited = (!citations.is_empty()).then(|| {
                    serde_json::json!({ "citationsContent": {
                        "content": [{ "text": text }],
                        "citations": citations,
                    } })
                });
                finish_text(out, index, cited)
            }
            Draft::Reasoning {
                text,
                signature,
                redacted,
            } => finish_reasoning(out, index, &text, signature.as_deref(), redacted.as_deref()),
            Draft::Hosted {
                id,
                name,
                kind,
                input,
            } => {
                // Converse takes a JSON object as a tool's input.
                let input = serde_json::from_str::<Value>(&input)
                    .ok()
                    .filter(Value::is_object)
                    .unwrap_or_else(|| Value::Object(Map::new()));
                out.finish_with(index, block::tool_use_json(&id, &name, input, Some(&kind)))
            }
            Draft::Result {
                id,
                status,
                kind,
                content,
                modeled,
            } => {
                if !modeled {
                    return out.close(index);
                }
                let item =
                    block::tool_result_json(&id, content, status.as_deref(), kind.as_deref());
                out.finish_with(index, item)
            }
            Draft::Image { format, source } => image(out, index, &format, source.as_ref()),
        }
    }

    fn text(&mut self, index: usize) -> &mut Draft {
        self.drafts.entry(index).or_insert_with(|| Draft::Text {
            text: String::new(),
            citations: Vec::new(),
        })
    }

    fn reasoning(
        &mut self,
        index: usize,
    ) -> Option<(&mut String, &mut Option<String>, &mut Option<Vec<u8>>)> {
        let draft = self
            .drafts
            .entry(index)
            .or_insert_with(|| Draft::Reasoning {
                text: String::new(),
                signature: None,
                redacted: None,
            });
        match draft {
            Draft::Reasoning {
                text,
                signature,
                redacted,
            } => Some((text, signature, redacted)),
            Draft::Text { .. }
            | Draft::Hosted { .. }
            | Draft::Result { .. }
            | Draft::Image { .. } => None,
        }
    }

    /// Write one Converse event in delivery order.
    #[deny(clippy::wildcard_enum_match_arm)]
    fn event(
        &mut self,
        event: aws_bedrock::ConverseStreamOutput,
        mut out: Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        use aws_bedrock::{
            ContentBlockDelta as Delta, ContentBlockStart as Start, ConverseStreamOutput as Event,
            ReasoningContentBlockDelta as Thought, ToolResultBlockDelta as ResultDelta,
        };
        match event {
            Event::ContentBlockStart(event) => {
                let index = block_index(event.content_block_index);
                match event.start {
                    Some(Start::ToolUse(start)) => match start.r#type {
                        None => call(
                            &mut out,
                            index,
                            Some(&start.tool_use_id),
                            Some(&start.name),
                            None,
                        )?,
                        Some(kind) => {
                            let kind = kind.as_str().to_owned();
                            let item = block::tool_use_json(
                                &start.tool_use_id,
                                &start.name,
                                Value::Object(Map::new()),
                                Some(&kind),
                            );
                            out.open(index, Block::Opaque { replay: true }, item)?;
                            self.drafts.insert(
                                index,
                                Draft::Hosted {
                                    id: start.tool_use_id,
                                    name: start.name,
                                    kind,
                                    input: String::new(),
                                },
                            );
                        }
                    },
                    Some(Start::ToolResult(result)) => {
                        let status = result
                            .status
                            .as_ref()
                            .map(|status| status.as_str().to_owned());
                        let item = block::tool_result_json(
                            &result.tool_use_id,
                            Vec::new(),
                            status.as_deref(),
                            result.r#type.as_deref(),
                        );
                        out.open(index, Block::Opaque { replay: true }, item)?;
                        self.drafts.insert(
                            index,
                            Draft::Result {
                                id: result.tool_use_id,
                                status,
                                kind: result.r#type,
                                content: Vec::new(),
                                modeled: true,
                            },
                        );
                    }
                    Some(Start::Image(image)) => {
                        self.drafts.insert(
                            index,
                            Draft::Image {
                                format: image.format,
                                source: None,
                            },
                        );
                    }
                    Some(_) | None => unknown(&mut out, index, "content block start")?,
                }
            }
            Event::ContentBlockDelta(event) => {
                let index = block_index(event.content_block_index);
                let not_started = || {
                    ProviderError::Response(format!(
                        "Converse sent a delta for block {} it did not start",
                        event.content_block_index
                    ))
                };
                match event.delta {
                    Some(Delta::Text(fragment)) => {
                        open_once(&mut out, index, Block::Text)?;
                        if let Draft::Text { text, .. } = self.text(index) {
                            text.push_str(&fragment);
                        }
                        out.push(index, &fragment)?;
                    }
                    Some(Delta::Citation(citation)) => {
                        open_once(&mut out, index, Block::Text)?;
                        match block::citation_delta_json(&citation) {
                            Some(citation) => {
                                if let Draft::Text { citations, .. } = self.text(index) {
                                    citations.push(citation);
                                }
                            }
                            None => skip("citation"),
                        }
                    }
                    Some(Delta::ToolUse(fragment)) => match self.drafts.get_mut(&index) {
                        Some(Draft::Hosted { input, .. }) => input.push_str(&fragment.input),
                        _ => call(&mut out, index, None, None, Some(&fragment.input))?,
                    },
                    Some(Delta::ReasoningContent(thought)) => match thought {
                        Thought::Text(fragment) => {
                            open_once(&mut out, index, Block::Reasoning { redacted: false })?;
                            if let Some((text, ..)) = self.reasoning(index) {
                                text.push_str(&fragment);
                            }
                            out.push(index, &fragment)?;
                        }
                        Thought::Signature(fragment) => {
                            open_once(&mut out, index, Block::Reasoning { redacted: false })?;
                            if let Some((_, signature, _)) = self.reasoning(index) {
                                signature.get_or_insert_default().push_str(&fragment);
                            }
                        }
                        Thought::RedactedContent(blob) => {
                            open_once(&mut out, index, Block::Reasoning { redacted: true })?;
                            if let Some((.., redacted)) = self.reasoning(index) {
                                // Encoded once whole: base64 of each chunk would
                                // not concatenate.
                                redacted
                                    .get_or_insert_default()
                                    .extend_from_slice(blob.as_ref());
                            }
                        }
                        Thought::Unknown { .. } | _ => skip("reasoning delta"),
                    },
                    Some(Delta::Image(image)) => {
                        let Some(Draft::Image { source, .. }) = self.drafts.get_mut(&index) else {
                            return Err(not_started());
                        };
                        *source = image.source;
                    }
                    Some(Delta::ToolResult(parts)) => {
                        let Some(Draft::Result {
                            content, modeled, ..
                        }) = self.drafts.get_mut(&index)
                        else {
                            return Err(not_started());
                        };
                        for part in parts {
                            match part {
                                ResultDelta::Json(value) => content
                                    .push(serde_json::json!({ "json": json::to_value(value) })),
                                ResultDelta::Text(text) => {
                                    content.push(serde_json::json!({ "text": text }));
                                }
                                ResultDelta::Unknown { .. } | _ => *modeled = false,
                            }
                        }
                    }
                    Some(_) | None => unknown(&mut out, index, "content block delta")?,
                }
            }
            Event::ContentBlockStop(event) => {
                self.stop(&mut out, block_index(event.content_block_index))?
            }
            Event::MessageStart(_) => {}
            Event::MessageStop(event) => {
                self.stop_reason = Some(event.stop_reason);
            }
            Event::Metadata(metadata) => {
                // A block the stream never stopped is not complete: images
                // are written as they arrived, and the end closes the rest
                // with no provider item.
                let images: Vec<usize> = self
                    .drafts
                    .iter()
                    .filter_map(|(index, draft)| {
                        matches!(draft, Draft::Image { .. }).then_some(*index)
                    })
                    .collect();
                for index in images {
                    self.stop(&mut out, index)?;
                }
                if !self.raw.is_empty() {
                    out.raw(Value::Object(std::mem::take(&mut self.raw)));
                }
                let end = finish(metadata.usage.as_ref(), self.stop_reason.as_ref());
                return Ok(out.end(end));
            }
            Event::Unknown { .. } | _ => skip("stream event"),
        }
        Ok(Flow::More)
    }
}

/// Finish the text at `index`: with `cited`, the whole cited block, as its
/// provider item.
fn finish_text(
    out: &mut Out<'_, Completion>,
    index: usize,
    cited: Option<Value>,
) -> Result<(), ProviderError> {
    match cited {
        Some(item) => out.finish_with(index, item),
        None => out.finish(index),
    }
}

/// Finish the reasoning at `index` with its Converse block as the provider
/// item. Reasoning with no text, signature or bytes is no content.
fn finish_reasoning(
    out: &mut Out<'_, Completion>,
    index: usize,
    text: &str,
    signature: Option<&str>,
    redacted: Option<&[u8]>,
) -> Result<(), ProviderError> {
    match (redacted, signature) {
        (Some(bytes), _) => out.finish_with(index, block::redacted_json(bytes)),
        (None, None) if text.is_empty() => out.finish(index),
        (None, signature) => out.finish_with(index, block::reasoning_json(text, signature)),
    }
}

/// Write the image at `index`, or a marker when Converse sent no bytes or
/// a format Rig does not name.
fn image(
    out: &mut Out<'_, Completion>,
    index: usize,
    format: &aws_bedrock::ImageFormat,
    source: Option<&aws_bedrock::ImageSource>,
) -> Result<(), ProviderError> {
    use aws_bedrock::ImageFormat as Format;
    let media_type = match format {
        Format::Gif => Some(ImageMediaType::GIF),
        Format::Jpeg => Some(ImageMediaType::JPEG),
        Format::Png => Some(ImageMediaType::PNG),
        Format::Webp => Some(ImageMediaType::WEBP),
        _ => None,
    };
    let (Some(media_type), Some(aws_bedrock::ImageSource::Bytes(blob))) = (media_type, source)
    else {
        return marker(out, index, "image");
    };
    let image = Image {
        data: DocumentSourceKind::Base64(String::new()),
        media_type: Some(media_type),
        ..Image::default()
    };
    out.whole(
        index,
        Block::Image(image),
        Value::Null,
        &BASE64_STANDARD.encode(blob.as_ref()),
    )
}

/// A block with no canonical meaning Rig can send back, kept as a marker
/// naming its `kind`.
fn marker(out: &mut Out<'_, Completion>, index: usize, kind: &str) -> Result<(), ProviderError> {
    out.whole(
        index,
        Block::Opaque { replay: false },
        serde_json::json!({ "type": kind }),
        "",
    )
}

/// A part this SDK version does not model: its block is a marker.
fn unknown(out: &mut Out<'_, Completion>, index: usize, item: &str) -> Result<(), ProviderError> {
    skip(item);
    if !out.is_open(index) {
        out.open(
            index,
            Block::Opaque { replay: false },
            serde_json::json!({ "type": "unknown" }),
        )?;
    }
    Ok(())
}

fn skip(item: &str) {
    tracing::warn!(item, "a Converse item this SDK version does not model");
}

/// Open the item at `index` as `block` unless an earlier delta opened it:
/// Converse starts only tool-use, tool-result and image blocks explicitly.
fn open_once(
    out: &mut Out<'_, Completion>,
    index: usize,
    block: Block,
) -> Result<(), ProviderError> {
    if !out.is_open(index) {
        out.open(index, block, Value::Null)?;
    }
    Ok(())
}

/// One fragment of the client call at `index`. A call that never names
/// its tool is dropped at its end, since nothing can answer it.
fn call(
    out: &mut Out<'_, Completion>,
    index: usize,
    id: Option<&str>,
    name: Option<&str>,
    arguments: Option<&str>,
) -> Result<(), ProviderError> {
    out.fragment(
        index,
        CallFragment {
            id,
            name,
            arguments,
        },
    )
}

/// Write one whole Converse block at `index`.
#[deny(clippy::wildcard_enum_match_arm)]
fn whole_block(
    out: &mut Out<'_, Completion>,
    index: usize,
    content: aws_bedrock::ContentBlock,
) -> Result<(), ProviderError> {
    use aws_bedrock::{ContentBlock as Content, ReasoningContentBlock as Reasoning};
    let item = block::to_json(&content);
    match content {
        Content::Text(text) => {
            out.open(index, Block::Text, Value::Null)?;
            out.push(index, &text)?;
            out.finish(index)
        }
        Content::CitationsContent(cited) => {
            out.open(index, Block::Text, Value::Null)?;
            for part in cited.content.unwrap_or_default() {
                match part {
                    aws_bedrock::CitationGeneratedContent::Text(text) => out.push(index, &text)?,
                    aws_bedrock::CitationGeneratedContent::Unknown { .. } | _ => {
                        skip("cited content");
                    }
                }
            }
            finish_text(out, index, item)
        }
        Content::ReasoningContent(Reasoning::ReasoningText(reasoning)) => {
            out.open(index, Block::Reasoning { redacted: false }, Value::Null)?;
            out.push(index, &reasoning.text)?;
            finish_reasoning(
                out,
                index,
                &reasoning.text,
                reasoning.signature.as_deref(),
                None,
            )
        }
        Content::ReasoningContent(Reasoning::RedactedContent(blob)) => {
            out.open(index, Block::Reasoning { redacted: true }, Value::Null)?;
            finish_reasoning(out, index, "", None, Some(blob.as_ref()))
        }
        Content::ToolUse(client) if client.r#type.is_none() => {
            let input = json::to_value(client.input).to_string();
            call(
                out,
                index,
                Some(&client.tool_use_id),
                Some(&client.name),
                Some(&input),
            )?;
            out.finish(index)
        }
        Content::ToolUse(_) | Content::ToolResult(_) => match item {
            Some(item) => out.whole(index, Block::Opaque { replay: true }, item, ""),
            None => marker(out, index, "tool_result"),
        },
        Content::Image(image) => self::image(out, index, &image.format, image.source.as_ref()),
        Content::Audio(_) => marker(out, index, "audio"),
        Content::CachePoint(_) => marker(out, index, "cache_point"),
        Content::Document(_) => marker(out, index, "document"),
        Content::GuardContent(_) => marker(out, index, "guard_content"),
        Content::SearchResult(_) => marker(out, index, "search_result"),
        Content::Video(_) => marker(out, index, "video"),
        Content::ReasoningContent(_) | _ => {
            skip("content block");
            marker(out, index, "unknown")
        }
    }
}

/// A whole Converse reply, written block by block.
#[deny(clippy::wildcard_enum_match_arm)]
fn whole(output: ConverseOutput, mut out: Out<'_, Completion>) -> Result<Flow, ProviderError> {
    match output.output {
        Some(aws_bedrock::ConverseOutput::Message(message)) => {
            for (index, content) in message.content.into_iter().enumerate() {
                whole_block(&mut out, index, content)?;
            }
        }
        Some(_) => skip("converse output"),
        None => {}
    }
    Ok(out.end(finish(output.usage.as_ref(), Some(&output.stop_reason))))
}

/// A whole reply the SDK could not read, from the JSON Bedrock sent: each
/// block that still converts, a marker for the rest, and the stop reason
/// and usage counts that hold the type Converse documents.
fn whole_json(document: &Value, mut out: Out<'_, Completion>) -> Result<Flow, ProviderError> {
    let content = document
        .pointer("/output/message/content")
        .and_then(Value::as_array);
    for (index, item) in content.into_iter().flatten().enumerate() {
        match block::from_json(item) {
            Some(content) => whole_block(&mut out, index, content)?,
            None => marker(&mut out, index, "unknown")?,
        }
    }
    let stop_reason = document
        .get("stopReason")
        .and_then(Value::as_str)
        .map(aws_bedrock::StopReason::from);
    let usage = document.get("usage").and_then(|usage| {
        let count = |key: &str| {
            let count = usage.get(key)?.as_i64()?;
            i32::try_from(count).ok()
        };
        aws_bedrock::TokenUsage::builder()
            .input_tokens(count("inputTokens").unwrap_or(0))
            .output_tokens(count("outputTokens").unwrap_or(0))
            .total_tokens(count("totalTokens").unwrap_or(0))
            .set_cache_read_input_tokens(count("cacheReadInputTokens"))
            .set_cache_write_input_tokens(count("cacheWriteInputTokens"))
            .build()
            .ok()
    });
    Ok(out.end(finish(usage.as_ref(), stop_reason.as_ref())))
}

impl<'id> rig_core::wire::Decoder<'id, Completion, ConverseFrame> for StreamState {
    type Event = ConverseFrame;

    fn classify(&self, frame: ConverseFrame) -> WireEvent<Self::Event> {
        // The SDK handles byte decoding; only unknown union variants need
        // classification here.
        match &frame {
            ConverseFrame::Event(event) if event.is_unknown() => {
                WireEvent::unrecognized("unknown", format!("{event:?}"))
            }
            _ => WireEvent::Known(frame),
        }
    }

    /// EOF without Bedrock's `Metadata` event is truncation.
    fn decode(
        &mut self,
        event: ConverseFrame,
        out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event {
            ConverseFrame::Opened { .. } => Ok(Flow::More),
            ConverseFrame::Whole(output) => whole(*output, out),
            ConverseFrame::Document(document) => whole_json(&document, out),
            ConverseFrame::Event(event) => self.event(event, out),
            ConverseFrame::Raw(Value::Object(event)) => {
                for (kind, payload) in event {
                    if matches!(kind.as_str(), "messageStart" | "messageStop" | "metadata") {
                        self.raw.insert(kind, payload);
                    }
                }
                Ok(Flow::More)
            }
            ConverseFrame::Raw(_) => Ok(Flow::More),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
