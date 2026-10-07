//! The Converse reassembler: a ConverseStream's events rebuild the
//! `ConverseOutput` a unary Converse call returns.

use std::collections::BTreeMap;

use base64::{Engine, prelude::BASE64_STANDARD};
use rig_core::json_utils::Lenient;
use rig_core::wire::document::Reassemble;
use serde_json::{Map, Value, json};

use super::member;
use crate::completion::ConverseFrame;

/// The padding field every ConverseStream event carries, which a unary
/// reply does not.
const PADDING: &str = "p";

/// One content block as its start and deltas build it.
#[derive(Debug)]
enum Block {
    /// `text` deltas, and the `citation` deltas that make it a
    /// `citationsContent` block.
    Text { text: String, citations: Vec<Value> },
    /// `reasoningContent` deltas: text and signature, or redacted chunks.
    Reasoning {
        text: Option<String>,
        signature: Option<String>,
        redacted: Vec<String>,
    },
    /// A `toolUse` start and its `input` fragments.
    ToolUse {
        fields: Map<String, Value>,
        input: String,
    },
    /// A `toolResult` start and its content deltas.
    ToolResult {
        fields: Map<String, Value>,
        content: Vec<Value>,
    },
    /// An `image` start and its base64 byte chunks.
    Image {
        fields: Map<String, Value>,
        chunks: Vec<String>,
    },
    /// A block this reassembler does not model: its start with its deltas
    /// merged in.
    Other {
        kind: String,
        fields: Map<String, Value>,
    },
}

/// Base64 `chunks` as one encoding of their bytes, since the encodings of
/// chunks do not concatenate. Chunks that are not base64 are joined as
/// they came.
fn joined(chunks: &[String]) -> String {
    let bytes: Result<Vec<Vec<u8>>, _> = chunks
        .iter()
        .map(|chunk| BASE64_STANDARD.decode(chunk))
        .collect();
    match bytes {
        Ok(bytes) => BASE64_STANDARD.encode(bytes.concat()),
        Err(_) => chunks.concat(),
    }
}

/// Merge `fields` into `into`, a `null` only filling an absent key.
fn merge(into: &mut Map<String, Value>, fields: &Map<String, Value>) {
    for (key, value) in fields {
        if !value.is_null() || !into.contains_key(key) {
            into.insert(key.clone(), value.clone());
        }
    }
}

impl Block {
    /// The block a `contentBlockStart` opens.
    fn started(kind: &str, body: &Value) -> Self {
        let fields = body.as_object().cloned().unwrap_or_default();
        match kind {
            "toolUse" => Self::ToolUse {
                fields,
                input: String::new(),
            },
            "toolResult" => Self::ToolResult {
                fields,
                content: Vec::new(),
            },
            "image" => Self::Image {
                fields,
                chunks: Vec::new(),
            },
            kind => Self::Other {
                kind: kind.to_owned(),
                fields,
            },
        }
    }

    /// The block a delta of `kind` opens when no start came first.
    fn opened(kind: &str) -> Self {
        match kind {
            "text" | "citation" => Self::Text {
                text: String::new(),
                citations: Vec::new(),
            },
            "reasoningContent" => Self::Reasoning {
                text: None,
                signature: None,
                redacted: Vec::new(),
            },
            kind => Self::started(kind, &Value::Null),
        }
    }

    /// Fold one delta of `kind` into the block; a delta of another block's
    /// kind is dropped.
    fn absorb(&mut self, kind: &str, body: &Value) {
        match (self, kind) {
            (Self::Text { text, .. }, "text") => text.push_str(body.as_str().unwrap_or_default()),
            (Self::Text { citations, .. }, "citation") => citations.push(body.clone()),
            (
                Self::Reasoning {
                    text,
                    signature,
                    redacted,
                },
                "reasoningContent",
            ) => {
                if let Some(more) = body.str("text") {
                    text.get_or_insert_default().push_str(more);
                }
                if let Some(more) = body.str("signature") {
                    signature.get_or_insert_default().push_str(more);
                }
                if let Some(chunk) = body.str("redactedContent") {
                    redacted.push(chunk.to_owned());
                }
            }
            (Self::ToolUse { input, .. }, "toolUse") => {
                input.push_str(body.str("input").unwrap_or_default());
            }
            (Self::ToolResult { content, .. }, "toolResult") => match body {
                Value::Array(parts) => content.extend(parts.iter().cloned()),
                Value::Null => {}
                part => content.push(part.clone()),
            },
            (Self::Image { chunks, .. }, "image") => {
                if let Some(bytes) = body.at("/source/bytes").and_then(Value::as_str) {
                    chunks.push(bytes.to_owned());
                }
            }
            (Self::Other { kind: own, fields }, kind) if own == kind => {
                if let Some(delta) = body.as_object() {
                    merge(fields, delta);
                }
            }
            _ => {}
        }
    }

    /// The block as a unary reply states it.
    fn finish(self) -> Value {
        match self {
            Self::Text { text, citations } if citations.is_empty() => json!({ "text": text }),
            Self::Text { text, citations } => json!({ "citationsContent": {
                "content": [{ "text": text }],
                "citations": citations,
            } }),
            Self::Reasoning { redacted, .. } if !redacted.is_empty() => {
                json!({ "reasoningContent": { "redactedContent": joined(&redacted) } })
            }
            Self::Reasoning {
                text, signature, ..
            } => {
                let mut reasoning = Map::new();
                reasoning.insert("text".to_owned(), Value::from(text.unwrap_or_default()));
                if let Some(signature) = signature {
                    reasoning.insert("signature".to_owned(), Value::from(signature));
                }
                json!({ "reasoningContent": { "reasoningText": reasoning } })
            }
            Self::ToolUse { mut fields, input } => {
                let input = if input.trim().is_empty() {
                    json!({})
                } else {
                    serde_json::from_str(&input).unwrap_or(Value::String(input))
                };
                fields.insert("input".to_owned(), input);
                json!({ "toolUse": fields })
            }
            Self::ToolResult {
                mut fields,
                content,
            } => {
                fields.insert("content".to_owned(), Value::Array(content));
                json!({ "toolResult": fields })
            }
            Self::Image { mut fields, chunks } => {
                if !chunks.is_empty() {
                    fields.insert("source".to_owned(), json!({ "bytes": joined(&chunks) }));
                }
                json!({ "image": fields })
            }
            Self::Other { kind, fields } => json!({ kind: fields }),
        }
    }
}

/// The `ConverseOutput` a ConverseStream adds up to.
///
/// `messageStart` gives `output.message.role`. Each `contentBlockStart` and
/// `contentBlockDelta`, by `contentBlockIndex`, builds
/// `output.message.content[i]`: text appends, and citations make it a
/// `citationsContent` block; reasoning text and signature append, and
/// redacted chunks join as bytes; a tool use's `input` appends and is
/// parsed as JSON; a tool result's content and an image's bytes collect.
/// `messageStop` gives `stopReason` and `additionalModelResponseFields`,
/// and `metadata` gives `usage`, `metrics`, `trace`, `performanceConfig`
/// and `serviceTier`. The stream's padding is dropped, and an exception
/// adds nothing. A whole reply is its own document.
#[derive(Debug, Default)]
pub struct ConverseOutput {
    role: Option<Value>,
    blocks: BTreeMap<u64, Block>,
    /// What `messageStop` and `metadata` state.
    top: Map<String, Value>,
    whole: Option<Value>,
}

impl rig_core::wire::document::Serves<rig_core::operation::Completion> for ConverseOutput {}

impl Reassemble<ConverseFrame> for ConverseOutput {
    fn absorb(&mut self, frame: &ConverseFrame) {
        let event = match frame {
            ConverseFrame::Whole(document) => {
                self.whole = Some(document.clone());
                return;
            }
            ConverseFrame::Event(event) => event,
        };
        let Some((kind, payload)) = member(event) else {
            return;
        };
        let index = payload.u64("contentBlockIndex").unwrap_or(0);
        match kind {
            "messageStart" => {
                if let Some(role) = payload.get("role") {
                    self.role = Some(role.clone());
                }
            }
            "contentBlockStart" => {
                if let Some((kind, body)) = payload.get("start").and_then(member) {
                    self.blocks.insert(index, Block::started(kind, body));
                }
            }
            "contentBlockDelta" => {
                if let Some((kind, body)) = payload.get("delta").and_then(member) {
                    self.blocks
                        .entry(index)
                        .or_insert_with(|| Block::opened(kind))
                        .absorb(kind, body);
                }
            }
            "messageStop" | "metadata" => {
                if let Some(fields) = payload.as_object() {
                    let mut fields = fields.clone();
                    fields.shift_remove(PADDING);
                    merge(&mut self.top, &fields);
                }
            }
            _ => {}
        }
    }

    fn finish(self) -> Value {
        if let Some(whole) = self.whole {
            return whole;
        }
        let mut message = Map::new();
        if let Some(role) = self.role {
            message.insert("role".to_owned(), role);
        }
        if !self.blocks.is_empty() {
            let content = self.blocks.into_values().map(Block::finish).collect();
            message.insert("content".to_owned(), Value::Array(content));
        }
        let mut document = Map::new();
        if !message.is_empty() {
            document.insert("output".to_owned(), json!({ "message": message }));
        }
        document.extend(self.top);
        if document.is_empty() {
            Value::Null
        } else {
            Value::Object(document)
        }
    }
}
