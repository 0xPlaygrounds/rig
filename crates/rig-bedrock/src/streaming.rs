//! The Converse reply decoder. A stream arrives as SDK events; a whole reply
//! is restated block by block through the same writer calls, so both fold
//! into the same turn. Only reasoning keeps a provider item: its
//! `signature`, or its `redacted` bytes as base64.
//!
//! The SDK's `Unknown` variants carry no payload, so an item this SDK
//! version does not model is dropped with a warning: there is nothing to
//! keep or send back.

use std::collections::BTreeMap;

use crate::completion::ConverseFrame;
use crate::types::assistant_content::finish;
use crate::types::converse_output::{
    CitationGeneratedContent, ContentBlock, ConversationRole, ConverseOutput, ImageBlock,
    ImageFormat, ImageSource, InternalConverseOutput, ReasoningContentBlock, StopReason,
    TokenUsage, ToolResultBlock, ToolResultContentBlock,
};
use crate::types::json;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use base64::{Engine, prelude::BASE64_STANDARD};
use rig_core::error::ProviderError;
use rig_core::message::{AssistantContent, CallId, ImageMediaType, ToolName};
use rig_core::operation::{Block, Completion, IfMalformed};
use rig_core::wire::{Flow, Out, WireEvent};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Deserialize, Serialize)]
pub struct BedrockStreamingResponse {
    pub usage: Option<TokenUsage>,
    /// Bedrock's own `stopReason` from the terminal `MessageStop` event, when
    /// the stream reported one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    /// AWS request ID from SDK operation metadata, or `None` when absent.
    /// Individual stream events do not contain this identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
}

/// The buffer index of a Converse content block: every signed index,
/// negative ones included, maps to a distinct one.
fn block_index(content_block_index: i32) -> usize {
    (i64::from(content_block_index) - i64::from(i32::MIN)) as usize
}

/// One Converse reply's state: a whole reply or a stream of events.
#[derive(Default)]
pub struct StreamState {
    /// Blocks a stream delivers in pieces but Rig takes whole (images, tool
    /// results), until their stop.
    assembling: BTreeMap<usize, ContentBlock>,
    /// Redacted reasoning bytes by block, encoded when the block stops.
    redacted: BTreeMap<usize, Vec<u8>>,
    stop_reason: Option<StopReason>,
    /// The AWS request id read off the SDK operation output before the event
    /// stream is opened; the reply's end carries it.
    provider_request_id: Option<String>,
}

impl StreamState {
    /// Write the whole block at `index` as a stream delivers it.
    #[deny(clippy::wildcard_enum_match_arm)]
    fn block(
        &mut self,
        out: &mut Out<'_, Completion>,
        index: usize,
        block: ContentBlock,
    ) -> Result<(), ProviderError> {
        match block {
            ContentBlock::Text(text) => push_text(out, index, &text)?,
            // History keeps the cited text, not the citations.
            ContentBlock::CitationsContent(cited) => {
                for content in cited.content.unwrap_or_default() {
                    match content {
                        CitationGeneratedContent::Text(text) => push_text(out, index, &text)?,
                        CitationGeneratedContent::Unknown => skip("cited content"),
                    }
                }
            }
            ContentBlock::ToolUse(call) => {
                open_call(out, index, &call.tool_use_id, &call.name)?;
                out.push(index, &call.input.to_string())?;
            }
            ContentBlock::ReasoningContent(ReasoningContentBlock::ReasoningText(reasoning)) => {
                push_reasoning(out, index, &reasoning.text)?;
                if let Some(signature) = reasoning.signature {
                    push_signature(out, index, signature)?;
                }
            }
            ContentBlock::ReasoningContent(ReasoningContentBlock::RedactedContent(blob)) => {
                self.push_redacted(out, index, &blob.inner)?;
            }
            ContentBlock::Image(image) => {
                return match assistant_image(&image) {
                    Some(image) => out.content(image),
                    None => opaque(out, index, "image"),
                };
            }
            ContentBlock::CachePoint(_) => return opaque(out, index, "cache_point"),
            ContentBlock::Document(_) => return opaque(out, index, "document"),
            ContentBlock::GuardContent(_) => return opaque(out, index, "guard_content"),
            ContentBlock::ToolResult(_) => return opaque(out, index, "tool_result"),
            ContentBlock::Video(_) => return opaque(out, index, "video"),
            ContentBlock::Audio => return opaque(out, index, "audio"),
            ContentBlock::SearchResult => return opaque(out, index, "search_result"),
            ContentBlock::ReasoningContent(ReasoningContentBlock::Unknown)
            | ContentBlock::Unknown => {
                skip("content block");
                return Ok(());
            }
        }
        self.stop(out, index)
    }

    /// The block at `index` finished.
    fn stop(&mut self, out: &mut Out<'_, Completion>, index: usize) -> Result<(), ProviderError> {
        self.flush(out, index)?;
        if out.is_open(index) {
            out.close(index, IfMalformed::Fail)?;
        }
        Ok(())
    }

    /// Hand the writer what this state holds for the block at `index`.
    fn flush(&mut self, out: &mut Out<'_, Completion>, index: usize) -> Result<(), ProviderError> {
        if let Some(block) = self.assembling.remove(&index) {
            return self.block(out, index, block);
        }
        if let Some(bytes) = self.redacted.remove(&index) {
            out.edit(index, |item| {
                *item = serde_json::json!({ "redacted": BASE64_STANDARD.encode(bytes) });
            })?;
        }
        Ok(())
    }

    fn push_redacted(
        &mut self,
        out: &mut Out<'_, Completion>,
        index: usize,
        bytes: &[u8],
    ) -> Result<(), ProviderError> {
        open_once(out, index, Block::Reasoning { redacted: true })?;
        // Encoded once whole: base64 of each chunk would not concatenate.
        self.redacted
            .entry(index)
            .or_default()
            .extend_from_slice(bytes);
        Ok(())
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
                let Some(start) = event.start else {
                    skip("content block start");
                    return Ok(Flow::More);
                };
                let assembled = match start {
                    Start::ToolUse(call) => {
                        open_call(&mut out, index, &call.tool_use_id, &call.name)?;
                        return Ok(Flow::More);
                    }
                    Start::Image(image) => ContentBlock::Image(ImageBlock {
                        format: image.format.into(),
                        source: None,
                    }),
                    Start::ToolResult(result) => ContentBlock::ToolResult(ToolResultBlock {
                        tool_use_id: result.tool_use_id,
                        content: Vec::new(),
                        status: result.status.map(Into::into),
                    }),
                    Start::Unknown { .. } => {
                        skip("content block start");
                        return Ok(Flow::More);
                    }
                    _ => {
                        skip("content block start");
                        return Ok(Flow::More);
                    }
                };
                self.assembling.insert(index, assembled);
            }
            Event::ContentBlockDelta(event) => {
                let index = block_index(event.content_block_index);
                let not_started = || {
                    ProviderError::Response(format!(
                        "Converse sent a delta for block {} it did not start",
                        event.content_block_index
                    ))
                };
                let Some(delta) = event.delta else {
                    skip("content block delta");
                    return Ok(Flow::More);
                };
                match delta {
                    Delta::Text(text) => push_text(&mut out, index, &text)?,
                    Delta::ToolUse(call) => out.push(index, &call.input)?,
                    Delta::ReasoningContent(thought) => match thought {
                        Thought::Text(text) => push_reasoning(&mut out, index, &text)?,
                        Thought::Signature(signature) => {
                            push_signature(&mut out, index, signature)?;
                        }
                        Thought::RedactedContent(blob) => {
                            self.push_redacted(&mut out, index, blob.as_ref())?;
                        }
                        Thought::Unknown { .. } => skip("reasoning delta"),
                        _ => skip("reasoning delta"),
                    },
                    // History keeps the cited text, not the citations.
                    Delta::Citation(_) => {}
                    Delta::Image(image) => {
                        let Some(ContentBlock::Image(assembled)) = self.assembling.get_mut(&index)
                        else {
                            return Err(not_started());
                        };
                        assembled.source = image.source.and_then(|source| source.try_into().ok());
                    }
                    Delta::ToolResult(parts) => {
                        let Some(ContentBlock::ToolResult(assembled)) =
                            self.assembling.get_mut(&index)
                        else {
                            return Err(not_started());
                        };
                        for part in parts {
                            assembled.content.push(match part {
                                ResultDelta::Json(value) => {
                                    ToolResultContentBlock::Json(json::to_value(value))
                                }
                                ResultDelta::Text(text) => ToolResultContentBlock::Text(text),
                                ResultDelta::Unknown { .. } => ToolResultContentBlock::Unknown,
                                _ => ToolResultContentBlock::Unknown,
                            });
                        }
                    }
                    Delta::Unknown { .. } => skip("content block delta"),
                    _ => skip("content block delta"),
                }
            }
            Event::ContentBlockStop(event) => {
                self.stop(&mut out, block_index(event.content_block_index))?
            }
            Event::MessageStart(_) => {}
            Event::MessageStop(event) => {
                self.stop_reason = Some(event.stop_reason.into());
            }
            Event::Metadata(metadata) => {
                // A block the stream never stopped still gets what this
                // state holds for it; the end closes the rest.
                let held: Vec<usize> = self
                    .assembling
                    .keys()
                    .chain(self.redacted.keys())
                    .copied()
                    .collect();
                for index in held {
                    self.flush(&mut out, index)?;
                }
                let native = BedrockStreamingResponse {
                    // The mirror conversion is infallible for `TokenUsage`.
                    usage: metadata
                        .usage
                        .and_then(|usage| TokenUsage::try_from(usage).ok()),
                    stop_reason: self.stop_reason.clone(),
                    provider_request_id: self.provider_request_id.clone(),
                };
                out.raw(serde_json::to_value(&native)?);
                let end = finish(native.usage.as_ref(), native.stop_reason.as_ref());
                return Ok(out.end(end));
            }
            Event::Unknown { .. } => skip("stream event"),
            _ => skip("stream event"),
        }
        Ok(Flow::More)
    }
}

/// An image the model produced, when Converse sent its bytes in a format
/// Rig names.
#[deny(clippy::wildcard_enum_match_arm)]
fn assistant_image(image: &ImageBlock) -> Option<AssistantContent> {
    let ImageSource::Bytes(blob) = image.source.as_ref()? else {
        return None;
    };
    let media_type = match image.format {
        ImageFormat::Gif => ImageMediaType::GIF,
        ImageFormat::Jpeg => ImageMediaType::JPEG,
        ImageFormat::Png => ImageMediaType::PNG,
        ImageFormat::Webp => ImageMediaType::WEBP,
        ImageFormat::Unknown(_) => return None,
    };
    Some(AssistantContent::image_base64(
        BASE64_STANDARD.encode(&blob.inner),
        Some(media_type),
        None,
    ))
}

/// A block with no canonical meaning, kept as a marker naming its `kind`
/// and never sent back. Like pi, rig stores no copy of the SDK's types.
fn opaque(out: &mut Out<'_, Completion>, index: usize, kind: &str) -> Result<(), ProviderError> {
    out.whole(
        index,
        Block::Opaque { replay: false },
        serde_json::json!({ "type": kind }),
        "",
    )
}

fn skip(item: &str) {
    tracing::warn!(
        item,
        "skipping a Converse item this SDK version does not model"
    );
}

/// Open the item at `index` as `block` unless an earlier delta opened it:
/// Converse starts only tool-use blocks explicitly.
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

fn push_text(out: &mut Out<'_, Completion>, index: usize, text: &str) -> Result<(), ProviderError> {
    open_once(out, index, Block::Text)?;
    out.push(index, text)
}

fn push_reasoning(
    out: &mut Out<'_, Completion>,
    index: usize,
    text: &str,
) -> Result<(), ProviderError> {
    open_once(out, index, Block::Reasoning { redacted: false })?;
    out.push(index, text)
}

/// Signature fragments concatenate into the item's `signature`.
fn push_signature(
    out: &mut Out<'_, Completion>,
    index: usize,
    signature: String,
) -> Result<(), ProviderError> {
    open_once(out, index, Block::Reasoning { redacted: false })?;
    out.merge(
        index,
        &serde_json::Map::from_iter([("signature".to_owned(), Value::String(signature))]),
    )
}

fn open_call(
    out: &mut Out<'_, Completion>,
    index: usize,
    id: &str,
    name: &str,
) -> Result<(), ProviderError> {
    let name = ToolName::new(name).map_err(|error| {
        ProviderError::Response(format!("AWS Bedrock returned a tool call: {error}"))
    })?;
    out.open(
        index,
        Block::Call {
            id: CallId::from_wire(id),
            name,
        },
        Value::Null,
    )
}

/// A whole Converse reply, restated as the stream of its blocks.
#[deny(clippy::wildcard_enum_match_arm)]
fn whole(
    state: &mut StreamState,
    output: InternalConverseOutput,
    mut out: Out<'_, Completion>,
) -> Result<Flow, ProviderError> {
    out.raw(serde_json::to_value(&output)?);
    let InternalConverseOutput {
        output: reply,
        stop_reason,
        usage,
        ..
    } = output;
    match reply {
        Some(ConverseOutput::Message(message)) => {
            if message.role != ConversationRole::Assistant {
                return Err(ProviderError::Response(
                    "Converse output message was not an assistant message".to_owned(),
                ));
            }
            for (index, block) in message.content.into_iter().enumerate() {
                state.block(&mut out, index, block)?;
            }
        }
        Some(ConverseOutput::Unknown) => skip("converse output"),
        None => {}
    }
    Ok(out.end(finish(usage.as_ref(), Some(&stop_reason))))
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
            ConverseFrame::Opened { request_id } => {
                self.provider_request_id = request_id;
                Ok(Flow::More)
            }
            ConverseFrame::Whole(output) => whole(self, *output, out),
            ConverseFrame::Event(event) => self.event(event, out),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
