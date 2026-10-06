//! What a provider says supports an answer text. A [`Citation`] lives on the
//! [`Text`] it cites, with a byte [`Span`] into that text or none for the
//! whole block, and the [`Source`]s behind it. The list is bound to the text
//! it was resolved against: once the text is edited, [`Text::citations`]
//! reads empty.
//!
//! Decoders never build a span: they hand the fold a
//! [`WireCitation`] whose span names its unit, and
//! the fold resolves it to bytes when the text block closes.
//!
//! ```
//! use rig_core::message::{Citation, Source, SourceLocation, Text};
//!
//! let text = Text::new("Dock Seven is open.");
//! let span = text.span(0..10).ok_or("on a character boundary")?;
//! let mut citation = Citation::new([Source::new(SourceLocation::Url {
//!     url: "https://example.com/docks".to_owned(),
//! })]);
//! citation.span = Some(span);
//! let text = text.with_citations([citation]);
//! assert_eq!(text.cited(&text.citations()[0]), Some("Dock Seven"));
//! # Ok::<(), &str>(())
//! ```

use std::ops::Range;

use serde::{Deserialize, Serialize};

use super::{Fingerprint, Text};
use crate::wire::{SpanUnit, WireCitation, WireSpan};

/// A claim in a text and the sources that support it.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Citation {
    /// The bytes of the text it covers; `None` for the whole block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<Span>,
    /// What supports the claim.
    #[serde(default)]
    pub sources: Vec<Source>,
}

impl Citation {
    /// A citation of the whole block by `sources`.
    pub fn new(sources: impl IntoIterator<Item = Source>) -> Self {
        Self {
            span: None,
            sources: sources.into_iter().collect(),
        }
    }
}

/// A byte range of a text, on character boundaries. Only the fold and
/// [`Text::span`] make one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    start: usize,
    end: usize,
}

impl Span {
    /// The first byte.
    pub fn start(&self) -> usize {
        self.start
    }

    /// The byte after the last.
    pub fn end(&self) -> usize {
        self.end
    }

    /// `start..end`.
    pub fn range(&self) -> Range<usize> {
        self.start..self.end
    }
}

/// One source of a citation.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Source {
    /// Where the source is.
    pub location: SourceLocation,
    /// The source's title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// The passage of the source the provider quoted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cited_text: Option<String>,
    /// The provider's confidence, 0 to 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

impl Source {
    /// A source at `location` with no title, quote or confidence.
    pub fn new(location: SourceLocation) -> Self {
        Self {
            location,
            title: None,
            cited_text: None,
            confidence: None,
        }
    }

    /// Set the title.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the quoted passage of the source.
    pub fn cited_text(mut self, text: impl Into<String>) -> Self {
        self.cited_text = Some(text.into());
        self
    }

    /// Set the provider's confidence.
    pub fn confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }
}

/// Where a cited source is.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SourceLocation {
    /// A document of the request, by position or id, and where in it.
    Document {
        /// The document's position in the request.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        index: Option<u32>,
        /// The document's id.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// The part of the document cited.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        within: Option<DocumentRange>,
    },
    /// A web page.
    Url {
        /// The page's address.
        url: String,
    },
    /// A file the provider stores.
    File {
        /// The provider's file id.
        file_id: String,
        /// The file's name.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        /// The container holding the file.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        container_id: Option<String>,
    },
    /// A search result the request supplied.
    SearchResult {
        /// The result's position in the request.
        index: u32,
        /// The result's source, as the request named it.
        source: String,
        /// The content blocks of the result cited.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        blocks: Option<Range<u32>>,
    },
    /// A tool's output.
    ToolOutput {
        /// The tool call or output id.
        id: String,
    },
}

/// The part of a document a citation covers.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "unit", rename_all = "snake_case")]
pub enum DocumentRange {
    /// Characters of the document's text.
    Chars(Range<u64>),
    /// Pages, from 1, the end exclusive.
    Pages(Range<u32>),
    /// Content blocks or chunks.
    Blocks(Range<u32>),
}

/// The citations of one text and the fingerprint of the text they fit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct Citations {
    fingerprint: Fingerprint,
    list: Vec<Citation>,
}

/// A stored `citations` value, or `None` when it cannot be read.
pub(super) fn lenient<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Citations>, D::Error> {
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value.and_then(|value| Citations::deserialize(value).ok()))
}

fn fingerprint(text: &str) -> Fingerprint {
    Fingerprint::of(&text)
}

/// Whether `citation`'s span lies on character boundaries of `text`.
fn fits(text: &str, citation: &Citation) -> bool {
    citation
        .span
        .is_none_or(|span| text.get(span.range()).is_some())
}

impl Text {
    /// The citations, while `text` is what they were resolved against;
    /// empty once it has been edited.
    pub fn citations(&self) -> &[Citation] {
        match &self.citations {
            Some(citations) if citations.fingerprint == fingerprint(&self.text) => &citations.list,
            _ => &[],
        }
    }

    /// This text with `citations`, bound to its current text. A citation
    /// whose span is off a character boundary or past the end is dropped
    /// with a warning.
    pub fn with_citations(mut self, citations: impl IntoIterator<Item = Citation>) -> Self {
        let list = citations
            .into_iter()
            .filter(|citation| {
                let fits = fits(&self.text, citation);
                if !fits {
                    tracing::warn!(
                        span = ?citation.span,
                        "dropped a citation whose span does not fit its text"
                    );
                }
                fits
            })
            .collect();
        self.bind(list);
        self
    }

    /// Remove every citation.
    pub fn clear_citations(&mut self) {
        self.citations = None;
    }

    /// The text `citation` covers: its span, or the whole block.
    pub fn cited(&self, citation: &Citation) -> Option<&str> {
        match citation.span {
            Some(span) => self.text.get(span.range()),
            None => Some(&self.text),
        }
    }

    /// A span over `range`, in bytes of this text; `None` when either end
    /// is off a character boundary or past the end, or `range` is reversed.
    pub fn span(&self, range: Range<usize>) -> Option<Span> {
        self.text.get(range.clone()).map(|_| Span {
            start: range.start,
            end: range.end,
        })
    }

    fn bind(&mut self, list: Vec<Citation>) {
        self.citations = (!list.is_empty()).then(|| Citations {
            fingerprint: fingerprint(&self.text),
            list,
        });
    }

    /// Stored citations whose spans do not fit the stored text are dropped.
    pub(super) fn checked(mut self) -> Self {
        if let Some(citations) = &mut self.citations {
            citations.list.retain(|citation| fits(&self.text, citation));
            if citations.list.is_empty() {
                self.citations = None;
            }
        }
        self
    }
}

/// Attach `wire` to `text`, the block the provider's item at `index` became:
/// each span resolved to bytes of the text, after `kept`, the citations the
/// block holds already. A span that does not resolve, or covers other text
/// than the provider quoted, drops its citation with a warning; a citation
/// never fails a reply. The completion fold is the only caller.
pub(crate) fn attach(
    text: &mut Text,
    kept: Vec<Citation>,
    wire: Vec<WireCitation>,
    provider: &str,
    index: usize,
) {
    let mut list = kept;
    for citation in wire {
        let WireCitation { span, sources } = citation;
        let span = match span.map(|span| resolve(&text.text, &span)).transpose() {
            Ok(span) => span,
            Err(reason) => {
                tracing::warn!(provider, index, reason, "dropped a citation");
                continue;
            }
        };
        list.push(Citation { span, sources });
    }
    text.bind(list);
}

/// The byte span `span` names in `text`.
fn resolve(text: &str, span: &WireSpan) -> Result<Span, &'static str> {
    let wire = |offset: u64| usize::try_from(offset).ok();
    let (Some(start), Some(end)) = (wire(span.start), wire(span.end)) else {
        return Err("the span is past the end of the text");
    };
    let byte = |offset: usize| match span.unit {
        SpanUnit::Bytes => Some(offset),
        SpanUnit::Chars => char_offset(text, offset),
        SpanUnit::Utf16 => utf16_offset(text, offset),
    };
    let (Some(start), Some(end)) = (byte(start), byte(end)) else {
        return Err("the span is past the end of the text or splits a character");
    };
    let Some(covered) = text.get(start..end) else {
        return Err("the span is reversed, past the end of the text or splits a character");
    };
    if span
        .quoted
        .as_deref()
        .is_some_and(|quoted| quoted != covered)
    {
        return Err("the span covers other text than the provider quoted");
    }
    Ok(Span { start, end })
}

/// The byte offset of the `chars`th character of `text`.
fn char_offset(text: &str, chars: usize) -> Option<usize> {
    text.char_indices()
        .map(|(at, _)| at)
        .chain(std::iter::once(text.len()))
        .nth(chars)
}

/// The byte offset of the `units`th UTF-16 code unit of `text`, if it
/// starts a character.
fn utf16_offset(text: &str, units: usize) -> Option<usize> {
    let mut seen = 0;
    for (at, character) in text.char_indices() {
        if seen == units {
            return Some(at);
        }
        if seen > units {
            return None;
        }
        seen += character.len_utf16();
    }
    (seen == units).then_some(text.len())
}

#[cfg(test)]
mod tests;
