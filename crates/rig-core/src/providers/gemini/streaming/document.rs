//! The GenerateContent reassembler: a `streamGenerateContent` reply's chunks
//! rebuilt into the `generateContent` body the same turn has unary. The
//! REST, Vertex AI and gRPC wires share it, each over its chunks' REST JSON.
//!
//! ```
//! use rig_core::providers::gemini::streaming::document::GenerateContentResponse;
//! use serde_json::json;
//!
//! let mut document = GenerateContentResponse::default();
//! for text in ["Hel", "lo"] {
//!     let chunk = json!({"candidates": [{"content": {"parts": [{"text": text}], "role": "model"}, "index": 0}]});
//!     document.chunk(chunk.as_object().cloned().unwrap_or_default());
//! }
//! assert_eq!(document.document()["candidates"][0]["content"]["parts"], json!([{"text": "Hello"}]));
//! ```

use serde_json::{Map, Value};

use super::{GenerateContentChunk, GenerateContentDecoder, merge_part};
use crate::json_utils::Lenient;
use crate::wire::document::Reassemble;
use crate::wire::{Decoder, WireEvent, WireFrame};

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
    /// The decoder's classifier.
    classifier: GenerateContentDecoder,
    document: Map<String, Value>,
    /// The `index` of each candidate held, in order.
    indices: Vec<u64>,
    /// Signatures that arrived alone before any part of their candidate,
    /// which the candidate's next part takes.
    signatures: Vec<(u64, String)>,
}

impl GenerateContentResponse {
    /// Absorb one chunk, in REST JSON.
    pub fn chunk(&mut self, chunk: Map<String, Value>) {
        for (key, value) in chunk {
            match (key.as_str(), value) {
                ("candidates", Value::Array(candidates)) => {
                    for (position, candidate) in candidates.into_iter().enumerate() {
                        self.candidate(position, candidate);
                    }
                }
                // Candidates of another type fail the decoder; the document
                // keeps the ones it holds.
                ("candidates", _) => {}
                ("promptFeedback", value) => {
                    if self.document.get(&key).is_none_or(Value::is_null) {
                        self.document.insert(key, value);
                    }
                }
                (_, value) => set(&mut self.document, key, value),
            }
        }
    }

    /// The document the chunks add up to; `Null` when there were none.
    pub fn document(self) -> Value {
        if self.document.is_empty() {
            Value::Null
        } else {
            Value::Object(self.document)
        }
    }

    /// Fold one streamed candidate into the candidate of its `index`.
    fn candidate(&mut self, position: usize, candidate: Value) {
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
                            match (key.as_str(), held.get_mut(&key), value) {
                                (
                                    "citationSources",
                                    Some(Value::Array(sources)),
                                    Value::Array(more),
                                ) => {
                                    sources.extend(more);
                                }
                                (_, _, value) => set(held, key, value),
                            }
                        }
                    }
                }
                (_, value) => set(held, key, value),
            }
        }
    }
}

/// Fold a streamed candidate's `content` into the content held so far.
fn content_into(
    held: &mut Map<String, Value>,
    content: Map<String, Value>,
    index: u64,
    signatures: &mut Vec<(u64, String)>,
) {
    for (key, value) in content {
        let Value::Array(more) = value else {
            set(held, key, value);
            continue;
        };
        if key != "parts" {
            set(held, key, Value::Array(more));
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
    mut part: Value,
    index: u64,
    signatures: &mut Vec<(u64, String)>,
) {
    let signature = part
        .str("thoughtSignature")
        .filter(|signature| !signature.is_empty())
        .map(str::to_owned);
    if let Some(text) = part.str("text") {
        if text.is_empty() && signature.is_none() {
            return;
        }
        let thought = part.bool("thought") == Some(true);
        if let Some(last) = parts.last_mut()
            && last.str("text").is_some()
            && (last.bool("thought") == Some(true)) == thought
            && !(signed(last) && signature.is_some())
        {
            merge_part(last, &part);
            return;
        }
    } else if let Value::Object(fields) = &part {
        let bare = ["thought", "thoughtSignature", "partMetadata"];
        let data = fields.keys().any(|key| !bare.contains(&key.as_str()));
        if !data && let Some(signature) = signature {
            match parts.last_mut().and_then(Value::as_object_mut) {
                Some(last) => {
                    last.insert("thoughtSignature".to_owned(), Value::String(signature));
                }
                None => signatures.push((index, signature)),
            }
            return;
        }
    }
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

/// Whether `part` holds a non-empty signature.
fn signed(part: &Value) -> bool {
    part.str("thoughtSignature")
        .is_some_and(|signature| !signature.is_empty())
}

/// Set `key` to `value`: a non-null value replaces, and `null` only fills
/// an absent key.
fn set(map: &mut Map<String, Value>, key: String, value: Value) {
    if !value.is_null() || !map.contains_key(&key) {
        map.insert(key, value);
    }
}

/// The value at `key`, made an empty `kind` (a list or an object) when it is
/// absent or of another type.
fn slot<'a, T: Default>(
    map: &'a mut Map<String, Value>,
    key: &str,
    kind: fn(T) -> Value,
) -> &'a mut Value {
    let slot = map.entry(key).or_insert(Value::Null);
    if std::mem::discriminant(slot) != std::mem::discriminant(&kind(T::default())) {
        *slot = kind(T::default());
    }
    slot
}

impl Reassemble<WireFrame> for GenerateContentResponse {
    fn absorb(&mut self, frame: &WireFrame) {
        // The driver reports what does not classify.
        if let WireEvent::Known(GenerateContentChunk(chunk)) =
            self.classifier.classify(frame.clone())
        {
            self.chunk(chunk);
        }
    }

    fn finish(self) -> Value {
        self.document()
    }
}

#[cfg(test)]
mod tests;
