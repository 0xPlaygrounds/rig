//! The GenerateContent reassembler: a `streamGenerateContent` reply's chunks
//! rebuilt into the `generateContent` body the same turn has unary. The
//! REST, Vertex AI and gRPC wires share it, each over its chunks' REST JSON.
//!
//! ```
//! use rig_core::providers::gemini::streaming::GenerateContentChunk;
//! use rig_core::providers::gemini::streaming::document::GenerateContentResponse;
//! use rig_core::wire::document::Reassemble;
//! use serde_json::json;
//!
//! let mut document = GenerateContentResponse::default();
//! for text in ["Hel", "lo"] {
//!     let chunk = json!({"candidates": [{"content": {"parts": [{"text": text}], "role": "model"}, "index": 0}]});
//!     document.absorb(&GenerateContentChunk(chunk.as_object().cloned().unwrap_or_default()));
//! }
//! assert_eq!(document.finish()["candidates"][0]["content"]["parts"], json!([{"text": "Hello"}]));
//! ```

use serde_json::{Map, Value};

use super::{GenerateContentChunk, merge_part};
use crate::json_utils::Lenient;
use crate::wire::document::{Reassemble, keep, slot};

/// The `generateContent` body a stream of chunks adds up to.
///
/// Candidates are kept by their `index`. Their parts append, and a text
/// part continues the text part before it while `thought` stays the same,
/// as the unary body states a run of text as one part: its text appends,
/// a signature joins it unless both carry one, and its other fields
/// replace. An empty text part with no signature carries nothing. A part
/// that holds only a signature joins the part before it.
/// `citationMetadata.citationSources` append. `promptFeedback` keeps its
/// first value; every other field keeps its last non-null one, and a
/// `null` only fills an absent field. A reply with no chunk has no
/// document.
#[derive(Debug, Default)]
pub struct GenerateContentResponse {
    document: Map<String, Value>,
    /// The `index` of each candidate held, in order.
    indices: Vec<u64>,
    /// Signatures that arrived alone before any part of their candidate,
    /// which the candidate's next part takes.
    signatures: Vec<(u64, String)>,
}

impl GenerateContentResponse {
    /// Fold one streamed candidate into the candidate of its `index`.
    fn candidate(&mut self, position: usize, candidate: &Value) {
        let Value::Object(candidate) = candidate else {
            return;
        };
        let index = candidate
            .get("index")
            .and_then(Value::as_u64)
            .unwrap_or(position as u64);
        let candidates = slot(&mut self.document, "candidates", Value::Array);
        let Value::Array(candidates) = candidates else {
            return;
        };
        let at = match self.indices.iter().position(|held| *held == index) {
            Some(at) => at,
            None => {
                self.indices.push(index);
                candidates.push(Value::Object(Map::new()));
                candidates.len() - 1
            }
        };
        let Some(Value::Object(held)) = candidates.get_mut(at) else {
            return;
        };
        for (key, value) in candidate {
            match (key.as_str(), value) {
                ("content", Value::Object(content)) => {
                    if let Value::Object(held) = slot(held, "content", Value::Object) {
                        content_into(held, content, index, &mut self.signatures);
                    }
                }
                ("citationMetadata", Value::Object(citations)) => {
                    if let Value::Object(held) = slot(held, "citationMetadata", Value::Object) {
                        for (key, value) in citations {
                            match (key.as_str(), held.get_mut(key), value) {
                                (
                                    "citationSources",
                                    Some(Value::Array(sources)),
                                    Value::Array(more),
                                ) => {
                                    sources.extend(more.iter().cloned());
                                }
                                (_, _, value) => keep(held, key, value),
                            }
                        }
                    }
                }
                (_, value) => keep(held, key, value),
            }
        }
    }
}

/// Fold a streamed candidate's `content` into the content held so far.
fn content_into(
    held: &mut Map<String, Value>,
    content: &Map<String, Value>,
    index: u64,
    signatures: &mut Vec<(u64, String)>,
) {
    for (key, value) in content {
        let Value::Array(more) = value else {
            keep(held, key, value);
            continue;
        };
        if key != "parts" {
            keep(held, key, value);
            continue;
        }
        let Value::Array(parts) = slot(held, "parts", Value::Array) else {
            return;
        };
        for part in more {
            part_into(parts, part, index, signatures);
        }
    }
}

/// Append one streamed part to a candidate's parts, joining the part before
/// it where the unary body states them as one.
fn part_into(
    parts: &mut Vec<Value>,
    part: &Value,
    index: u64,
    signatures: &mut Vec<(u64, String)>,
) {
    let signature = super::signature(part).map(str::to_owned);
    if let Some(text) = part.str("text") {
        if text.is_empty() && signature.is_none() {
            return;
        }
        let thought = part.bool("thought") == Some(true);
        if let Some(last) = parts.last_mut()
            && last.str("text").is_some()
            && (last.bool("thought") == Some(true)) == thought
            && !(super::signature(last).is_some() && signature.is_some())
        {
            merge_part(last, part);
            return;
        }
    } else if let Value::Object(fields) = part
        && super::is_bare(fields)
        && let Some(signature) = signature
    {
        match parts.last_mut().and_then(Value::as_object_mut) {
            Some(last) => {
                last.insert("thoughtSignature".to_owned(), Value::String(signature));
            }
            None => signatures.push((index, signature)),
        }
        return;
    }
    let mut part = part.clone();
    if let Some(at) = signatures.iter().position(|(held, _)| *held == index)
        && let Some(fields) = part.as_object_mut()
    {
        let (_, signature) = signatures.remove(at);
        fields
            .entry("thoughtSignature")
            .or_insert(Value::String(signature));
    }
    parts.push(part);
}

impl crate::wire::document::Serves<crate::operation::Completion> for GenerateContentResponse {}

/// Each chunk in REST JSON; the document is `Null` when there were none.
impl Reassemble<GenerateContentChunk> for GenerateContentResponse {
    fn absorb(&mut self, GenerateContentChunk(chunk): &GenerateContentChunk) {
        for (key, value) in chunk {
            match (key.as_str(), value) {
                ("candidates", Value::Array(candidates)) => {
                    for (position, candidate) in candidates.iter().enumerate() {
                        self.candidate(position, candidate);
                    }
                }
                // Candidates of another type fail the decoder; the document
                // keeps the ones it holds.
                ("candidates", _) => {}
                ("promptFeedback", value) => {
                    if self.document.get(key).is_none_or(Value::is_null) {
                        self.document.insert(key.clone(), value.clone());
                    }
                }
                (_, value) => keep(&mut self.document, key, value),
            }
        }
    }

    fn finish(self) -> Value {
        if self.document.is_empty() {
            Value::Null
        } else {
            Value::Object(self.document)
        }
    }
}

#[cfg(test)]
mod tests;
