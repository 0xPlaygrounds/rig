//! Citations as a provider states them, before the completion fold resolves
//! their spans. A decoder hands each one to
//! [`Out::cite`](crate::wire::Out::cite) with the unit its provider counts
//! offsets in, so no decoder converts a unit or builds a byte span itself.

use crate::message::Source;

/// A citation as the provider stated it.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct WireCitation {
    /// The part of the text it covers; `None` for the whole block.
    pub span: Option<WireSpan>,
    /// What supports the claim.
    pub sources: Vec<Source>,
}

impl WireCitation {
    /// A citation of `span` (or, with `None`, the whole block) by `sources`.
    pub fn new(span: Option<WireSpan>, sources: Vec<Source>) -> Self {
        Self { span, sources }
    }
}

/// Offsets into a text block as the provider counts them.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WireSpan {
    /// The first offset.
    pub start: u64,
    /// The offset after the last.
    pub end: u64,
    /// What the offsets count.
    pub unit: SpanUnit,
    /// The text the provider says the span covers, which the resolved span
    /// must match.
    pub quoted: Option<String>,
}

impl WireSpan {
    /// `start..end` counted in `unit`, with no quoted text.
    pub fn new(start: u64, end: u64, unit: SpanUnit) -> Self {
        Self {
            start,
            end,
            unit,
            quoted: None,
        }
    }

    /// Set the text the provider says the span covers.
    pub fn quoted(mut self, text: impl Into<String>) -> Self {
        self.quoted = Some(text.into());
        self
    }
}

/// What a provider's offsets count.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpanUnit {
    /// UTF-8 bytes.
    Bytes,
    /// Unicode scalar values.
    Chars,
    /// UTF-16 code units.
    Utf16,
}
