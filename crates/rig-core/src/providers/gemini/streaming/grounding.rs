//! A GenerateContent candidate's `groundingMetadata` and `citationMetadata`
//! as citations of the answer text. Both sit beside the content, not in a
//! part, so the decoder tracks where each part's text landed and places
//! each segment in the block holding it.

use serde_json::Value;

use crate::json_utils::Lenient;
use crate::message::{Source, SourceLocation};
use crate::wire::{SpanUnit, WireCitation, WireSpan};

/// The answer text a reply wrote so far: each text block with its text, in
/// order. Thought text is not answer text.
#[derive(Debug, Default)]
pub(super) struct AnswerText {
    blocks: Vec<(usize, String)>,
}

impl AnswerText {
    /// Start the text block at `index` with `text`.
    pub(super) fn open(&mut self, index: usize, text: &str) {
        self.blocks.push((index, text.to_owned()));
    }

    /// Append `text` to the block at `index`, returning the byte offset it
    /// starts at.
    pub(super) fn push(&mut self, index: usize, text: &str) -> Option<usize> {
        let (_, held) = self.blocks.iter_mut().find(|(at, _)| *at == index)?;
        let offset = held.len();
        held.push_str(text);
        Some(offset)
    }

    /// The block and offset in it of byte `at` of the whole answer text, a
    /// span's start.
    fn locate(&self, at: usize) -> Option<(usize, usize)> {
        let mut start = 0;
        for (index, text) in &self.blocks {
            if at < start + text.len() {
                return Some((*index, at - start));
            }
            start += text.len();
        }
        None
    }

    /// The position in the whole answer text of `offset` in block `index`.
    fn absolute(&self, index: usize, offset: usize) -> Option<usize> {
        let mut start = 0;
        for (at, text) in &self.blocks {
            if *at == index {
                return Some(start + offset);
            }
            start += text.len();
        }
        None
    }

    /// Whether block `index` holds `quoted` at `range`.
    fn holds(&self, index: usize, range: std::ops::Range<usize>, quoted: &str) -> bool {
        self.blocks
            .iter()
            .find(|(at, _)| *at == index)
            .and_then(|(_, text)| text.get(range))
            == Some(quoted)
    }

    /// The occurrence of `quoted` nearest `near`, a position in the whole
    /// answer text, as its block and offset.
    fn find(&self, quoted: &str, near: usize) -> Option<(usize, usize)> {
        let mut start = 0;
        let mut best: Option<(usize, usize, usize)> = None;
        for (index, text) in &self.blocks {
            for (offset, _) in text.match_indices(quoted) {
                let distance = (start + offset).abs_diff(near);
                if best.is_none_or(|(_, _, held)| distance < held) {
                    best = Some((*index, offset, distance));
                }
            }
            start += text.len();
        }
        best.map(|(index, offset, _)| (index, offset))
    }
}

/// Each `groundingSupports` segment of `metadata` as a citation of the
/// block holding it. A segment counts bytes within `parts[partIndex]`,
/// which `placed` maps to its block and the byte offset the part starts
/// at; a streamed reply restarts part indices in every chunk, so a segment
/// whose `text` is not there is looked for as a position in the whole
/// answer, then as the occurrence of its text nearest that position.
pub(super) fn grounding(
    metadata: &Value,
    placed: &[Option<(usize, usize)>],
    answer: &AnswerText,
) -> Vec<(usize, WireCitation)> {
    let chunks = metadata.arr("groundingChunks");
    metadata
        .arr("groundingSupports")
        .iter()
        .filter_map(|support| {
            let scores = support.arr("confidenceScores");
            let sources: Vec<Source> = support
                .arr("groundingChunkIndices")
                .iter()
                .enumerate()
                .filter_map(|(at, chunk)| {
                    let chunk = chunks.get(usize::try_from(chunk.as_u64()?).ok()?)?;
                    let source = chunk_source(chunk)?;
                    Some(match scores.get(at).and_then(Value::as_f64) {
                        Some(score) => source.confidence(score as f32),
                        None => source,
                    })
                })
                .collect();
            if sources.is_empty() {
                return None;
            }
            let (index, span) = segment(support.get("segment")?, placed, answer)?;
            Some((index, WireCitation::new(Some(span), sources)))
        })
        .collect()
}

/// The block and span of a `segment`. Proto3 JSON leaves out a zero index.
fn segment(
    segment: &Value,
    placed: &[Option<(usize, usize)>],
    answer: &AnswerText,
) -> Option<(usize, WireSpan)> {
    let offset = |key: &str| usize::try_from(segment.u64(key).unwrap_or(0)).ok();
    let (start, end) = (offset("startIndex")?, offset("endIndex")?);
    let in_part = offset("partIndex")
        .and_then(|part| placed.get(part).copied().flatten())
        .map(|(index, at)| (index, at + start));
    let in_answer = answer.locate(start);
    let quoted = segment.str("text").filter(|text| !text.is_empty());
    let (index, at) = match quoted {
        None => in_part.or(in_answer)?,
        Some(quoted) => [in_part, in_answer]
            .into_iter()
            .flatten()
            .find(|(index, at)| answer.holds(*index, *at..*at + (end - start), quoted))
            .or_else(|| {
                let near = in_part
                    .and_then(|(index, at)| answer.absolute(index, at))
                    .unwrap_or(start);
                answer.find(quoted, near)
            })
            .or(in_part)
            .or(in_answer)?,
    };
    let span = WireSpan::new(
        at as u64,
        (at + end.checked_sub(start)?) as u64,
        SpanUnit::Bytes,
    );
    Some((
        index,
        match quoted {
            Some(quoted) => span.quoted(quoted),
            None => span,
        },
    ))
}

/// The source a grounding chunk names: a web page or a map place by its
/// URI, retrieved context by its URI or else its document.
fn chunk_source(chunk: &Value) -> Option<Source> {
    let (kind, fields) = ["web", "retrievedContext", "maps"]
        .into_iter()
        .find_map(|kind| chunk.get(kind).map(|fields| (kind, fields)))?;
    let location = match (fields.str("uri"), kind) {
        (Some(uri), _) => SourceLocation::Url {
            url: uri.to_owned(),
        },
        (None, "retrievedContext") => SourceLocation::Document {
            index: None,
            id: fields.str("documentName").map(str::to_owned),
            within: None,
        },
        (None, _) => return None,
    };
    let mut source = Source::new(location);
    if let Some(title) = fields.str("title") {
        source = source.title(title);
    }
    if kind == "retrievedContext"
        && let Some(text) = fields.str("text")
    {
        source = source.cited_text(text);
    }
    Some(source)
}

/// Each source of `metadata`, a `citationMetadata`, as a citation. The
/// Gemini API's `citationSources` count bytes of the answer text. Vertex
/// AI's `citations` do not state their unit, so each cites the first text
/// block whole.
pub(super) fn recitations(metadata: &Value, answer: &AnswerText) -> Vec<(usize, WireCitation)> {
    let source = |citation: &Value| {
        let mut source = Source::new(SourceLocation::Url {
            url: citation.str("uri")?.to_owned(),
        });
        if let Some(title) = citation.str("title") {
            source = source.title(title);
        }
        Some(source)
    };
    let gemini = metadata
        .arr("citationSources")
        .iter()
        .filter_map(|citation| {
            let start = usize::try_from(citation.u64("startIndex").unwrap_or(0)).ok()?;
            let end = usize::try_from(citation.u64("endIndex")?).ok()?;
            let (index, at) = answer.locate(start)?;
            let span = WireSpan::new(
                at as u64,
                (at + end.checked_sub(start)?) as u64,
                SpanUnit::Bytes,
            );
            Some((
                index,
                WireCitation::new(Some(span), vec![source(citation)?]),
            ))
        });
    let first = answer.blocks.first().map(|(index, _)| *index);
    let vertex = metadata
        .arr("citations")
        .iter()
        .filter_map(|citation| Some((first?, WireCitation::new(None, vec![source(citation)?]))));
    gemini.chain(vertex).collect()
}

#[cfg(test)]
mod tests;
